"""Utility functions for aligning dataset instances to unique words."""

from typing import Optional, Tuple, List, Sequence
from pathlib import Path

import os
import random
import torch
import torch.nn as nn
import numpy as np
import ot
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial import distance
from omegaconf import OmegaConf
import editdistance
from torch.utils.data import DataLoader, Subset
from alignment.ctc_utils import greedy_ctc_decode, beam_search_ctc_decode

cfg = OmegaConf.load("alignment/alignment_configs/trainer_config.yaml")

# count calls to :func:`align_more_instances`
_ALIGN_CALL_COUNT = 0

from htr_base.utils.htr_dataset import HTRDataset
from htr_base.models import HTRNet
from htr_base.utils.vocab import load_vocab



def harvest_backbone_features(
    dataset: HTRDataset,
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

def calculate_ot_projections(
    pa: np.ndarray,
    X: np.ndarray,
    pb: np.ndarray,
    Y: np.ndarray,
    reg: float = 0.1,
    *,
    unbalanced: bool = False,
    reg_m: float = 1.0,
    sinkhorn_kwargs: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute OT projections of ``X`` onto ``Y``.

    Parameters
    ----------
    pa : np.ndarray
        Source distribution over ``X`` of shape ``(N,)``.
    X : np.ndarray
        Source features of shape ``(N, D)``.
    pb : np.ndarray
        Target distribution over ``Y`` of shape ``(M,)``.
    Y : np.ndarray
        Target features of shape ``(M, D)``.
    reg : float, optional
        Entropic regularisation.
    unbalanced : bool, optional
        Use unbalanced OT formulation if ``True``.
    reg_m : float, optional
        Unbalanced mass regularisation.
    sinkhorn_kwargs : dict, optional
        Additional arguments for the OT solver.

    Returns
    -------
    projections : np.ndarray
        ``X`` projected in the space of ``Y`` (``(N, D)``).
    plan : np.ndarray
        Optimal transport plan of shape ``(N, M)``.
    """
    if sinkhorn_kwargs is None:
        sinkhorn_kwargs = {}

    M = distance.cdist(X, Y)

    if unbalanced:
        T = ot.unbalanced.sinkhorn_unbalanced(
            pa, pb, M, reg, reg_m, **sinkhorn_kwargs
        )
    else:
        T = ot.sinkhorn(pa, pb, M, reg=reg, **sinkhorn_kwargs)

    row_sum = T.sum(axis=1, keepdims=True)
    inv_row_sum = np.divide(1.0, row_sum, out=np.zeros_like(row_sum), where=row_sum != 0)
    projections = inv_row_sum * (T @ Y)
    assert np.all(np.isfinite(projections)), "Non-finite values in projections"

    return projections, T


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


class OTAligner:
    """Helper class implementing the OT pseudo-labelling routine."""

    def __init__(
        self,
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
        metric: str = "entropy",
        agree_threshold: int = 1,
    ) -> None:
        if metric not in {"gap", "entropy", "variance"}:
            raise ValueError("metric must be 'gap', 'entropy' or 'variance'")
        self.dataset = dataset
        self.backbone = backbone
        self.projectors = (
            projectors if isinstance(projectors, Sequence) else [projectors]
        )
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.reg = reg
        self.unbalanced = unbalanced
        self.reg_m = reg_m
        self.sinkhorn_kwargs = sinkhorn_kwargs or {}
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
    def _calculate_ot(self, proj_feats: torch.Tensor) -> tuple[torch.Tensor, np.ndarray]:
        """Compute the OT projection of projector outputs onto word embeddings.

        Args:
            proj_feats (torch.Tensor): Projector features of shape ``(N, D)``.

        Returns:
            tuple[torch.Tensor, np.ndarray]:
                The projected features as a tensor and the OT plan as a NumPy array.
        """
        N, V = proj_feats.size(0), self.word_embs.size(0)
        # uniform distribution over descriptors
        a = np.full((N,), 1.0 / N, dtype=np.float64)
        if getattr(self.dataset, "unique_word_probs", None):
            b = np.asarray(self.dataset.unique_word_probs, dtype=np.float64)
            if not self.unbalanced:
                b /= b.sum()
        else:
            # uniform distribution over vocabulary
            b = np.full((V,), 1.0 / V, dtype=np.float64)

        proj_np, plan = calculate_ot_projections(
            a,
            proj_feats.detach().numpy(),
            b,
            self.word_embs.cpu().numpy(),
            self.reg,
            unbalanced=self.unbalanced,
            reg_m=self.reg_m,
            sinkhorn_kwargs=self.sinkhorn_kwargs,
        )

        assert np.isfinite(proj_np).all(), "NaNs in OT projections"
        assert plan.shape == (N, V), "OT plan shape wrong"
        # convert OT projections back to torch
        proj_feats_ot = torch.from_numpy(proj_np).float()
        return proj_feats_ot, plan

    # ------------------------------------------------------------------
    def _get_projector_outputs(self):
        """Run projectors on all descriptors and compute OT projections.

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
        moved_list: list[torch.Tensor] = []
        plan: np.ndarray | None = None

        # process each projector in the ensemble
        for proj in self.projectors:
            proj.eval().to(self.device)
            proj_feats = proj(feats_all.to(self.device)).cpu()
            assert proj_feats.shape[1] == self.word_embs.shape[1], (
                "Embedding dim mismatch"
            )
            assert torch.isfinite(proj_feats).all(), "Non-finite projector output"

            # project descriptors onto the word embedding space using OT
            proj_feats_ot, plan = self._calculate_ot(proj_feats)

            moved_dist = (proj_feats_ot - proj_feats).norm(p=2, dim=1)
            dist_matrix = torch.cdist(proj_feats_ot, self.word_embs.cpu())
            nearest_word = dist_matrix.argmin(dim=1)

            Z_list.append(proj_feats_ot)
            dist_list.append(dist_matrix)
            nearest_list.append(nearest_word)
            moved_list.append(moved_dist)

        Z = torch.stack(Z_list)
        var_scores = Z.var(dim=0, unbiased=False).sum(dim=1)
        dist_matrix = torch.stack(dist_list).mean(dim=0)
        moved_dist = torch.stack(moved_list).mean(dim=0)
        preds = torch.stack(nearest_list)
        nearest_word, _ = preds.mode(dim=0)
        counts = (preds == nearest_word.unsqueeze(0)).sum(dim=0)

        plan_torch = (
            torch.from_numpy(plan).float() if plan is not None else torch.empty(0)
        )
        proj_feats_mean = Z.mean(dim=0)

        return {
            "plan": plan_torch,
            "proj_feats_mean": proj_feats_mean,
            "dist_matrix": dist_matrix,
            "moved_dist": moved_dist,
            "nearest_word": nearest_word,
            "counts": counts,
            "var_scores": var_scores,
            "aligned_all": aligned_all,
        }

    # ------------------------------------------------------------------
    def _select_candidates(
        self,
        counts: torch.Tensor,
        dist_matrix: torch.Tensor,
        plan: torch.Tensor,
        aligned_all: torch.Tensor,
        var_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Choose which dataset indices to pseudo-label this round.

        Args:
            counts (torch.Tensor): Agreement counts from the projector ensemble.
            dist_matrix (torch.Tensor): Pairwise distances to word embeddings.
            plan (torch.Tensor): Optimal transport plan from OT alignment.
            aligned_all (torch.Tensor): Current alignment vector ``(N,)``.
            var_scores (torch.Tensor): Per-sample variance scores.

        Returns:
            torch.Tensor: 1-D tensor of selected dataset indices.
        """
        assert self.k >= 0, "k must be non-negative"
        # rank samples by uncertainty according to the chosen metric
        if self.metric == "variance":
            order_unc = torch.argsort(var_scores).cpu().numpy()
        else:
            order_unc = select_uncertain_instances(
                m=len(self.dataset),
                transport_plan=plan.numpy() if self.metric == "entropy" else None,
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
        plan: torch.Tensor,
        var_scores: torch.Tensor,
    ) -> None:
        """Print alignment statistics for the current round.

        Args:
            chosen (torch.Tensor): Newly aligned dataset indices.
            nearest_word (torch.Tensor): Predicted word ids for each sample.
            moved_dist (torch.Tensor): Distances moved by descriptors during OT.
            dist_matrix (torch.Tensor): Pairwise distances after projection.
            plan (torch.Tensor): Optimal transport plan used for alignment.
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
            else:
                row = plan[chosen_list] / plan[chosen_list].sum(axis=1, keepdims=True)
                ent = -(row * np.log(row + 1e-12)).sum(axis=1)
                print(
                    f"[Align] mean row entropy: {ent.mean():.4f} ± {ent.std():.4f}"
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
            out["plan"],
            out["aligned_all"],
            out["var_scores"],
        )

        self._update_dataset(chosen, out["nearest_word"])
        self._log_results(
            chosen,
            out["nearest_word"],
            out["moved_dist"],
            out["dist_matrix"],
            out["plan"].numpy() if out["plan"].numel() else np.empty((0, 0)),
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
    metric: str = "entropy",          # 'gap', 'entropy' or 'variance' (assignment certainty)
    agree_threshold: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Wrapper over :class:`OTAligner` for backward compatibility.

    When ``cfg.pseudo_label_validation.enable`` is ``True``, this function
    invokes :meth:`OTAligner.validate_pseudo_labels` once the number of
    calls to :func:`align_more_instances` reaches the configured
    ``start_iteration``.
    """
    global _ALIGN_CALL_COUNT
    _ALIGN_CALL_COUNT += 1

    aligner = OTAligner(
        dataset,
        backbone,
        projectors,
        batch_size=batch_size,
        device=device,
        reg=reg,
        unbalanced=unbalanced,
        reg_m=reg_m,
        sinkhorn_kwargs=sinkhorn_kwargs,
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


def compute_ot_responsibilities(
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
    ensemble: str = "mean",
    topk: Optional[int] = None,
) -> torch.Tensor:
    """Compute soft responsibilities over words from the OT transport plan.

    Purpose:
        Runs the existing harvesting → projector → OT pipeline, collects the
        entropic OT plan(s) and row‑normalises them into a row‑stochastic
        responsibility matrix ``r`` of shape ``(N, V)``. If multiple
        projectors are provided, per‑projector responsibilities are fused
        by averaging. Optionally keeps only the top‑K entries per row and
        renormalises.

    Args:
        dataset (HTRDataset): Dataset exposing ``unique_word_embeddings`` and
            optional ``unique_word_probs`` used as the OT target marginal.
        backbone (HTRNet): Visual encoder used to extract per‑image descriptors.
        projectors (Sequence[nn.Module]): One or more projectors mapping
            descriptors to the word‑embedding space.
        batch_size (int, optional): Mini‑batch size for feature harvesting.
        device (str, optional): Device used when running ``backbone``.
        reg (float, optional): Entropic OT regularisation parameter.
        unbalanced (bool, optional): Whether to use unbalanced OT.
        reg_m (float, optional): Mass regularisation weight for unbalanced OT.
        sinkhorn_kwargs (dict, optional): Extra args forwarded to the solver.
        ensemble (str, optional): How to fuse per‑projector responsibilities;
            currently only ``"mean"`` is supported.
        topk (int | None, optional): If set, keep the K largest entries per
            row and renormalise; the rest are zeroed for sparsity.

    Returns:
        torch.Tensor: CPU tensor ``(N, V)`` with non‑negative entries and rows
        summing to 1 (up to numerical tolerance). If ``topk`` is set, at most
        K entries per row are non‑zero.
    """
    if not isinstance(projectors, Sequence) or isinstance(projectors, (nn.Module,)):
        projectors = [projectors]  # type: ignore[list-item]

    device_t = torch.device(device)
    # 1) Harvest descriptors once
    feats_all, _ = harvest_backbone_features(
        dataset,
        backbone,
        batch_size=batch_size,
        num_workers=0,
        device=device_t,
    )

    # Word embeddings and priors
    if not hasattr(dataset, "unique_word_embeddings"):
        raise AttributeError("dataset must provide unique_word_embeddings")
    word_embs = dataset.unique_word_embeddings
    if word_embs.ndim != 2 or word_embs.size(0) == 0:
        raise ValueError("unique_word_embeddings must be (V, E) with V>0")
    V, E = word_embs.size(0), word_embs.size(1)

    # Prepare target marginal b (word priors)
    if getattr(dataset, "unique_word_probs", None):
        b = np.asarray(dataset.unique_word_probs, dtype=np.float64)
        if not unbalanced:
            b = b / b.sum()
    else:
        b = np.full((V,), 1.0 / V, dtype=np.float64)

    resp_list: list[np.ndarray] = []
    for proj in projectors:
        proj.eval().to(device_t)
        proj_feats = proj(feats_all.to(device_t)).detach().cpu()
        if proj_feats.ndim != 2:
            raise RuntimeError(f"Projector output must be (N, E), got {proj_feats.shape}")
        if proj_feats.size(1) != E:
            raise RuntimeError(
                f"Embedding dimension mismatch: got {proj_feats.size(1)} vs {E}"
            )

        N = proj_feats.size(0)
        a = np.full((N,), 1.0 / N, dtype=np.float64)

        _proj_np, plan = calculate_ot_projections(
            a,
            proj_feats.numpy(),
            b,
            word_embs.cpu().numpy(),
            reg,
            unbalanced=unbalanced,
            reg_m=reg_m,
            sinkhorn_kwargs=sinkhorn_kwargs,
        )
        # Row‑normalise plan safely
        row_sum = plan.sum(axis=1, keepdims=True)
        r = np.divide(plan, row_sum, out=np.zeros_like(plan), where=row_sum != 0)
        resp_list.append(r.astype(np.float32, copy=False))

    if not resp_list:
        raise RuntimeError("No projectors provided")

    if ensemble != "mean":
        raise ValueError("Only ensemble='mean' is supported at the moment")
    R = np.mean(resp_list, axis=0)

    # Optional top‑K sparsification per row
    if topk is not None:
        if topk <= 0:
            raise ValueError("topk must be a positive integer")
        K = min(int(topk), V)
        for i in range(R.shape[0]):
            row = R[i]
            if K < V:
                idx = np.argpartition(row, -K)[-K:]
                mask = np.zeros_like(row, dtype=bool)
                mask[idx] = True
                row = np.where(mask, row, 0.0)
            s = row.sum()
            if s > 0:
                row = row / s
            R[i] = row

    # Final checks and return
    if not np.all(np.isfinite(R)):
        raise RuntimeError("Non‑finite values in responsibilities")
    if R.shape != (feats_all.size(0), V):
        raise RuntimeError("Responsibility shape mismatch")

    return torch.from_numpy(R.astype(np.float32))
