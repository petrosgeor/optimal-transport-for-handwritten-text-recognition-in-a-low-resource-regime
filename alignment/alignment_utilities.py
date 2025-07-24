"""Utility functions for aligning dataset instances to unique words."""

from typing import Optional, Tuple, List, Sequence
from pathlib import Path

import os
import random
import warnings
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from omegaconf import OmegaConf
import editdistance
from torch.utils.data import DataLoader, Subset
from alignment.ctc_utils import greedy_ctc_decode, beam_search_ctc_decode

cfg = OmegaConf.load("alignment/alignment_configs/trainer_config.yaml")

# count calls to :func:`align_more_instances`
_ALIGN_CALL_COUNT = 0

from htr_base.utils.htr_dataset import HTRDataset, FusedHTRDataset
from htr_base.models import HTRNet
from htr_base.utils.vocab import load_vocab



def harvest_backbone_features(
    dataset: HTRDataset | FusedHTRDataset,
    backbone: HTRNet,
    *,
    batch_size: int = 64,
    num_workers: int = 0,
    device: torch.device = torch.device(cfg.device),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return image descriptors and alignment info for the whole dataset.

    Dataset augmentations are temporarily disabled while collecting
    descriptors. The backbone is run in evaluation mode on ``device``.

    Parameters
    ----------
    dataset : HTRDataset
        Dataset providing images and alignment information.
    backbone : HTRNet
        Visual encoder used to extract per-image descriptors.
    batch_size : int, optional
        Mini-batch size when forwarding the dataset.
    num_workers : int, optional
        Worker processes used by the ``DataLoader``.
    device : torch.device | str, optional
        Device on which the backbone runs.

    Returns
    -------
    torch.Tensor
        Tensor of descriptors with shape ``(N, D)`` where ``N`` is the
        dataset size.
    torch.Tensor
        Alignment tensor of shape ``(N,)`` copied from the dataset.
    """

    device = torch.device(device)
    backbone = backbone.to(device).eval()

    orig_transforms = getattr(dataset, "transforms", None)
    dataset.transforms = None

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    feats_buf: list[torch.Tensor] = []
    align_buf: list[torch.Tensor] = []

    with torch.no_grad():
        for imgs, _txt, aligned in loader:
            imgs = imgs.to(device)
            output = backbone(imgs, return_feats=True)
            if backbone.phoc_head is not None:
                feats = output[-2]
            else:
                feats = output[-1]
            if feats.dim() != 2:
                raise RuntimeError(
                    f"Expected (B, feat_dim) descriptors, got {feats.shape}"
                )
            feats_buf.append(feats.cpu())
            align_buf.append(aligned.cpu())

    dataset.transforms = orig_transforms

    feats_all = torch.cat(feats_buf, dim=0)
    aligned_all = torch.cat(align_buf, dim=0)

    return feats_all, aligned_all


def select_uncertain_instances(
    m: int,
    *,
    transport_plan: Optional[np.ndarray] = None,
    dist_matrix: Optional[np.ndarray] = None,
    metric: str = "gap",
) -> np.ndarray:
    """Return indices of the ``m`` most uncertain instances.

    Parameters
    ----------
    m : int
        Number of indices to return.
    transport_plan : np.ndarray, optional
        OT plan of shape ``(N, V)``. Required for ``metric='entropy'``.
    dist_matrix : np.ndarray, optional
        Pre-computed pairwise distances ``(N, V)``. Required for
        ``metric='gap'``.
    metric : str, optional
        Either ``'gap'`` or ``'entropy'`` selecting the uncertainty measure.

    Returns
    -------
    np.ndarray
        Array of ``m`` indices sorted by decreasing uncertainty.
    """
    if metric not in {"gap", "entropy"}:
        raise ValueError("metric must be 'gap' or 'entropy'")

    if m <= 0:
        raise ValueError("m must be positive")

    if metric == "gap":
        if dist_matrix is None:
            raise ValueError("dist_matrix required for metric='gap'")
        assert dist_matrix.ndim == 2, "dist_matrix must be 2D"
        n = dist_matrix.shape[0]
        assert m <= n, "m cannot exceed number of dataset instances"
        sorted_d = np.sort(dist_matrix, axis=1)
        gaps = sorted_d[:, 1] - sorted_d[:, 0]
        order = np.argsort(gaps)
    else:
        if transport_plan is None:
            raise ValueError("transport_plan required for metric='entropy'")
        assert transport_plan.ndim == 2, "transport_plan must be 2D"
        n = transport_plan.shape[0]
        assert m <= n, "m cannot exceed number of dataset instances"
        row_sum = transport_plan.sum(axis=1, keepdims=True)
        p = np.divide(transport_plan, row_sum, out=np.zeros_like(transport_plan), where=row_sum != 0)
        ent = -np.sum(p * np.log(p + 1e-12), axis=1)
        order = np.argsort(-ent)

    return order[:m]


class ProjectionAligner:
    """Helper class implementing projection-based pseudo-labelling."""

    def __init__(
        self,
        dataset: HTRDataset,
        backbone: HTRNet,
        projectors: Sequence[nn.Module],
        *,
        batch_size: int = 512,
        device: str = cfg.device,
        k: int = 0,
        metric: str = "gap",
        agree_threshold: int = 1,
    ) -> None:
        """Initialise the projection aligner.

        Args:
            dataset (HTRDataset): Dataset providing images and alignment info.
            backbone (HTRNet): Visual backbone network.
            projectors (Sequence[nn.Module]): Projector ensemble.
            batch_size (int): Mini-batch size during descriptor harvesting.
            device (str): Device used for alignment.
            k (int): Number of least-moved descriptors to pseudo-label.
            metric (str): Certainty metric ('gap', 'variance', 'closest').
            agree_threshold (int): Minimum number of agreeing projectors.

        Returns:
            None
        """
        if metric == "entropy" or metric not in {"gap", "variance", "closest"}:
            raise ValueError("metric must be 'gap', 'variance' or 'closest'")
        self.dataset = dataset
        self.backbone = backbone
        self.projectors = (
            projectors if isinstance(projectors, Sequence) else [projectors]
        )
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.k = k
        self.metric = metric
        self.agree_threshold = agree_threshold

        assert agree_threshold <= len(self.projectors), (
            "agree_threshold bigger than ensemble size"
        )
        assert hasattr(dataset, "unique_word_embeddings"), (
            "Dataset missing unique_word_embeddings"
        )
        assert (
            dataset.unique_word_embeddings.ndim == 2
            and dataset.unique_word_embeddings.size(0) > 0
        ), "Word-embedding matrix empty or wrong shape"
        self.word_embs = dataset.unique_word_embeddings.to(self.device)

    # ------------------------------------------------------------------
    def _get_projector_outputs(self):
        """Run projectors on all descriptors and compute ensemble statistics.

        Returns:
            dict: Dictionary containing the OT plan, mean projected features,
            distance matrix, moved distances, nearest word indices,
            per-sample variance scores and the ``aligned`` vector from the
            dataset.
        """
        # collect image descriptors from the backbone
        feats_all, aligned_all = harvest_backbone_features(
            self.dataset,
            self.backbone,
            batch_size=self.batch_size,
            num_workers=0,
            device=self.device,
        )

        assert len(self.dataset) == feats_all.size(0) == aligned_all.size(0), (
            "Feature count mismatch"
        )

        Z_list: list[torch.Tensor] = []
        dist_list: list[torch.Tensor] = []
        nearest_list: list[torch.Tensor] = []

        # process each projector in the ensemble
        for proj in self.projectors:
            proj.eval().to(self.device)
            proj_feats = proj(feats_all.to(self.device)).cpu()
            assert proj_feats.shape[1] == self.word_embs.shape[1], (
                "Embedding dim mismatch"
            )
            assert torch.isfinite(proj_feats).all(), "Non-finite projector output"

            dist_matrix = torch.cdist(proj_feats, self.word_embs.cpu())
            nearest_word = dist_matrix.argmin(dim=1)
            Z_list.append(proj_feats)
            dist_list.append(dist_matrix)
            nearest_list.append(nearest_word)

        Z = torch.stack(Z_list)
        var_scores = Z.var(dim=0, unbiased=False).sum(dim=1)
        dist_matrix = torch.stack(dist_list).mean(dim=0)
        moved_dist = torch.zeros(dist_matrix.size(0))
        preds = torch.stack(nearest_list)
        nearest_word, _ = preds.mode(dim=0)
        counts = (preds == nearest_word.unsqueeze(0)).sum(dim=0)
        proj_feats_mean = Z.mean(dim=0)

        return {
            "plan": torch.empty(0),
            "proj_feats_mean": proj_feats_mean,
            "dist_matrix": dist_matrix,
            "moved_dist": moved_dist,
            "nearest_word": nearest_word,
            "counts": counts,
            "var_scores": var_scores,
            "aligned_all": aligned_all,
        }

    # ------------------------------------------------------------------
    def _select_closest_per_word(
        self,
        counts: torch.Tensor,
        dist_matrix: torch.Tensor,
        aligned_all: torch.Tensor,
    ) -> torch.Tensor:
        """Select one unaligned sample for each word embedding.

        Args:
            counts (torch.Tensor): Ensemble agreement counts.
            dist_matrix (torch.Tensor): Mean Euclidean distances ``(N, V)``.
            aligned_all (torch.Tensor): Current alignment flags ``(N,)``.

        Returns:
            torch.Tensor: Tensor containing the chosen dataset indices.
        """
        N, V = dist_matrix.shape
        taken = aligned_all.clone()
        chosen: list[int] = []
        top_k_words = 100
        if top_k_words > 0:
            probs = np.asarray(self.dataset.unique_word_probs)  # Convert to np for sorting
            top_indices = np.argsort(-probs)[:top_k_words]  # Top-K descending
        else:
            top_indices = range(V)  # All words (current behavior)
        
        for w in top_indices:  # Changed: Loop over top_indices instead of range(V)
            if (taken == w).any():
                continue
            mask = taken == -1
            if not mask.any():
                break
            dist_col = dist_matrix[:, w].clone()
            dist_col[~mask] = float("inf")
            idx = int(dist_col.argmin().item())
            if not mask[idx]:
                continue
            if counts[idx].item() >= self.agree_threshold:
                chosen.append(idx)
                taken[idx] = w
        return torch.tensor(chosen, dtype=torch.long)

    # ------------------------------------------------------------------
    def _select_candidates(
        self,
        counts: torch.Tensor,
        dist_matrix: torch.Tensor,
        aligned_all: torch.Tensor,
        var_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Choose which dataset indices to pseudo-label this round.

        Args:
            counts (torch.Tensor): Agreement counts from the projector ensemble.
            dist_matrix (torch.Tensor): Pairwise distances to word embeddings.
            aligned_all (torch.Tensor): Current alignment vector ``(N,)``.
            var_scores (torch.Tensor): Per-sample variance scores.

        Returns:
            torch.Tensor: 1-D tensor of selected dataset indices.
        """
        assert self.k >= 0, "k must be non-negative"

        if self.metric == "closest":
            return self._select_closest_per_word(counts, dist_matrix, aligned_all)

        # rank samples by uncertainty according to the chosen metric
        if self.metric == "variance":
            order_unc = torch.argsort(var_scores).cpu().numpy()
        else:
            order_unc = select_uncertain_instances(
                m=len(self.dataset),
                dist_matrix=dist_matrix.numpy() if self.metric == "gap" else None,
                metric=self.metric,
            )
        order_cert = order_unc[::-1]
        mask_new = (aligned_all == -1).numpy()
        assert self.agree_threshold > 0, "agree_threshold must be positive"
        assert order_cert.size > 0, "No candidates produced"

        # keep samples that are still unlabelled and meet the agreement threshold
        chosen: list[int] = []
        for idx in order_cert:
            if mask_new[idx] and counts[idx].item() >= self.agree_threshold:
                chosen.append(idx)
                if len(chosen) == min(self.k, mask_new.sum()):
                    break
        return torch.tensor(chosen, dtype=torch.long)

    # ------------------------------------------------------------------
    def _update_dataset(self, chosen: torch.Tensor, nearest_word: torch.Tensor) -> None:
        """Write newly aligned indices back into ``dataset.aligned``.

        Args:
            chosen (torch.Tensor): Dataset indices selected for labelling.
            nearest_word (torch.Tensor): Predicted word indices for each sample.

        Returns:
            None
        """
        if chosen.numel() > 0:
            prev_vals = self.dataset.aligned[chosen]
            # ensure we only label previously unaligned samples
            assert (prev_vals == -1).all(), "About to overwrite existing labels!"
            self.dataset.aligned[chosen] = nearest_word[chosen].to(torch.int32)

    # ------------------------------------------------------------------
    def _log_results(
        self,
        chosen: torch.Tensor,
        nearest_word: torch.Tensor,
        moved_dist: torch.Tensor,
        dist_matrix: torch.Tensor,
        var_scores: torch.Tensor,
    ) -> None:
        """Print alignment statistics for the current round.

        Args:
            chosen (torch.Tensor): Newly aligned dataset indices.
            nearest_word (torch.Tensor): Predicted word ids for each sample.
            moved_dist (torch.Tensor): Distances moved by descriptors (always zero).
            dist_matrix (torch.Tensor): Pairwise distances after projection.
            var_scores (torch.Tensor): Variance scores for every dataset sample.

        Returns:
            None
        """
        # report statistics for the newly pseudo-labelled samples
        chosen_list = chosen.tolist()
        if self.k > 0 and chosen_list:
            correct_new = sum(
                self.dataset.unique_words[self.dataset.aligned[i].item()] ==
                self.dataset.transcriptions[i].strip()
                for i in chosen_list
            )
            print(
                f"[Align] newly pseudo-labelled {len(chosen_list)} items; "
                f"[Align] round accuracy {correct_new}/{len(chosen_list)} "
                f"({correct_new/len(chosen_list):.1%})"
            )

            for idx in random.sample(chosen_list, min(10, len(chosen_list))):
                gt = self.dataset.transcriptions[idx].strip()
                pred = self.dataset.unique_words[self.dataset.aligned[idx].item()]
                print(f"[Align] sample: '{gt}' → '{pred}'")

            md = moved_dist[chosen]
            print(
                f"[Align] mean moved distance: {md.mean():.4f} ± {md.std(unbiased=False):.4f}"
            )

            vs = var_scores[chosen]
            print(
                f"[Align] mean var score: {vs.mean():.4f} ± {vs.std(unbiased=False):.4f}"
            )

            if self.metric == "gap":
                top2 = torch.topk(dist_matrix[chosen], 2, largest=False).values
                gap_vals = top2[:, 1] - top2[:, 0]
                print(
                    f"[Align] mean NN-gap: {gap_vals.mean():.4f} ± {gap_vals.std(unbiased=False):.4f}"
                )

        # track running accuracy over all pseudo-labelled samples
        aligned_idx = torch.nonzero(self.dataset.aligned != -1, as_tuple=True)[0]
        if aligned_idx.numel():
            correct_tot = sum(
                self.dataset.unique_words[self.dataset.aligned[i].item()] ==
                self.dataset.transcriptions[i].strip()
                for i in aligned_idx.tolist()
            )
            print(
                f"[Align] cumulative accuracy: {correct_tot}/{aligned_idx.numel()} "
                f"({correct_tot/aligned_idx.numel():.1%}) – "
                f"{aligned_idx.numel()} of {len(self.dataset)} samples aligned"
            )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate_pseudo_labels(
        self,
        *,
        edit_threshold: int = 3,
        batch_size: int = 256,
        decode_cfg: dict = None,
        num_workers: int = 0,
    ) -> int:
        """Drop unreliable pseudo-labels based on the backbone output.

        Counts how many removed labels differ from the dataset transcriptions.

        Parameters
        ----------
        edit_threshold : int, optional
            Maximum allowed Levenshtein distance between prediction and label.
        batch_size : int, optional
            Mini-batch size for the evaluation pass.
        decode_cfg : dict | None, optional
            Decoding configuration forwarded to the decoding helper.
        num_workers : int, optional
            Workers used by the ``DataLoader``.

        Returns
        -------
        int
            Number of samples that were un-aligned because the distance
            exceeded ``edit_threshold``.
        """
        self.backbone.eval()
        device = self.device

        aligned_idx = torch.nonzero(self.dataset.aligned != -1, as_tuple=True)[0]
        if not aligned_idx.numel():
            return 0

        loader = DataLoader(
            Subset(self.dataset, aligned_idx.tolist()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        removed = 0
        correct_removed = 0
        ptr = 0
        _, i2c = load_vocab()
        for imgs, *_ in loader:
            imgs = imgs.to(device, non_blocking=True)
            logits, *_ = self.backbone(imgs, return_feats=False)

            if decode_cfg and decode_cfg.get("method") == "beam":
                preds = beam_search_ctc_decode(
                    logits.cpu(), i2c, beam_width=decode_cfg.get("beam_width", 3)
                )
            else:
                preds = greedy_ctc_decode(logits.cpu(), i2c)

            for i, pred in enumerate(preds):
                idx = aligned_idx[ptr + i].item()
                gold_word = self.dataset.unique_words[self.dataset.aligned[idx].item()]
                if editdistance.eval(pred, gold_word) > edit_threshold:
                    self.dataset.aligned[idx] = -1
                    removed += 1
                    if hasattr(self.dataset, "transcriptions") and (
                        self.dataset.transcriptions[idx] != gold_word
                    ):
                        correct_removed += 1
            ptr += len(preds)

        print(
            f"[Validate] removed {removed} unreliable pseudo-labels "
            f"(threshold = {edit_threshold}). {correct_removed} mismatched ground truth."
        )
        return removed

    # ------------------------------------------------------------------
    def align(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform one OT pseudo-labelling iteration.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                Transport plan, mean projected features and per-sample moved distance.
        """
        # gather features and OT statistics for the current dataset state
        out = self._get_projector_outputs()

        chosen = self._select_candidates(
            out["counts"],
            out["dist_matrix"],
            out["aligned_all"],
            out["var_scores"],
        )

        self._update_dataset(chosen, out["nearest_word"])
        self._log_results(
            chosen,
            out["nearest_word"],
            out["moved_dist"],
            out["dist_matrix"],
            out["var_scores"],
        )

        # switch networks back to training mode
        for proj in self.projectors:
            proj.train()
        self.backbone.train()

        assert torch.isfinite(out["moved_dist"]).all(), "Non-finite moved_dist"
        return out["plan"], out["proj_feats_mean"], out["moved_dist"]

@torch.no_grad()
def align_more_instances(
    dataset: HTRDataset,
    backbone: HTRNet,
    projectors: Sequence[nn.Module],
    *,
    batch_size: int = 512,
    device: str = cfg.device,
    reg: float = 0.1,
    unbalanced: bool = False,
    reg_m: float = 1.0,
    sinkhorn_kwargs: Optional[dict] = None,
    k: int = 0,
    metric: str = "gap",
    agree_threshold: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Wrapper over :class:`ProjectionAligner` for backward compatibility.

    When ``cfg.pseudo_label_validation.enable`` is ``True``, this function
    invokes :meth:`ProjectionAligner.validate_pseudo_labels` once the number of
    calls to :func:`align_more_instances` reaches the configured
    ``start_iteration``.
    """
    global _ALIGN_CALL_COUNT
    _ALIGN_CALL_COUNT += 1

    if (
        reg != 0.1 or unbalanced or reg_m != 1.0 or (sinkhorn_kwargs is not None and sinkhorn_kwargs)
    ):
        warnings.warn(
            "OT-related parameters are deprecated and ignored",
            DeprecationWarning,
        )

    aligner = ProjectionAligner(
        dataset,
        backbone,
        projectors,
        batch_size=batch_size,
        device=device,
        k=k,
        metric=metric,
        agree_threshold=agree_threshold,
    )
    plan, proj_feats, moved = aligner.align()

    # --- conditional pseudo-label validation ---
    vcfg = getattr(cfg, "pseudo_label_validation", None)
    if (
        vcfg
        and vcfg.enable
        and _ALIGN_CALL_COUNT >= int(getattr(vcfg, "start_iteration", 0))
    ):
        aligner.validate_pseudo_labels(
            edit_threshold=int(vcfg.edit_distance),
            batch_size=batch_size,
            decode_cfg=getattr(cfg, "decode_config", None),
        )

    return plan, proj_feats, moved
