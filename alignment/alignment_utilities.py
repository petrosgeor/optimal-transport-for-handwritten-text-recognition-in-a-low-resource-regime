"""Utility functions for aligning dataset instances to unique words."""

from typing import Optional, Tuple
from pathlib import Path

import os
import random
import torch
import torch.nn as nn
import numpy as np
import ot
from scipy.spatial import distance
from omegaconf import OmegaConf
import editdistance
from torch.utils.data import DataLoader, Subset
from alignment.utils.ctc import greedy_ctc_decode, beam_search_ctc_decode


def _resolve_trainer_config_path() -> Path:
    """Resolve trainer config path with env override and stable fallback.

    Returns:
        Path: Absolute path to trainer YAML used by alignment utilities.
    """
    env_cfg = os.environ.get("ALIGNMENT_TRAINER_CONFIG")
    if env_cfg:
        return Path(env_cfg).resolve()
    return (Path(__file__).resolve().parent / "alignment_configs" / "trainer_config.yaml").resolve()


cfg = OmegaConf.load(_resolve_trainer_config_path())

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
    descriptors. The backbone is run in evaluation mode on device.

    Args:
        dataset (HTRDataset): Dataset providing images and alignment metadata.
        backbone (HTRNet): Visual encoder used to extract descriptors.
        batch_size (int): Mini-batch size used during feature harvesting.
        num_workers (int): Number of DataLoader workers.
        device (torch.device | str): Device where the backbone runs.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Feature descriptors with shape (N, D)
        and aligned ids with shape (N,).
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
                raise RuntimeError(f"Expected (B, feat_dim) descriptors, got {feats.shape}")
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
    """Compute OT projections of X onto Y.

    Args:
        pa (np.ndarray): Source marginal with shape (N,).
        X (np.ndarray): Source feature matrix with shape (N, D).
        pb (np.ndarray): Target marginal with shape (M,).
        Y (np.ndarray): Target feature matrix with shape (M, D).
        reg (float): Entropic regularization coefficient.
        unbalanced (bool): Whether to use unbalanced OT.
        reg_m (float): Mass regularization for unbalanced OT.
        sinkhorn_kwargs (Optional[dict]): Extra keyword args for the solver.

    Returns:
        tuple[np.ndarray, np.ndarray]: Projected source features with shape
            (N, D) and transport plan with shape (N, M).
    """
    if sinkhorn_kwargs is None:
        sinkhorn_kwargs = {}

    M = distance.cdist(X, Y)

    if unbalanced:
        T = ot.unbalanced.sinkhorn_unbalanced(pa, pb, M, reg, reg_m, **sinkhorn_kwargs)
    else:
        T = ot.sinkhorn(pa, pb, M, reg=reg, **sinkhorn_kwargs)

    row_sum = T.sum(axis=1, keepdims=True)
    inv_row_sum = np.divide(1.0, row_sum, out=np.zeros_like(row_sum), where=row_sum != 0)
    projections = inv_row_sum * (T @ Y)
    assert np.all(np.isfinite(projections)), "Non-finite values in projections"
    return projections, T


class OTAligner:
    """Helper class implementing the OT pseudo-labelling routine."""

    def __init__(
        self,
        dataset: HTRDataset,
        backbone: HTRNet,
        projector: nn.Module,
        *,
        batch_size: int = 512,
        device: str = cfg.device,
        reg: float = 0.1,
        unbalanced: bool = False,
        reg_m: float = 1.0,
        sinkhorn_kwargs: Optional[dict] = None,
        k: int = 0,
    ) -> None:
        """Initialize one OT pseudo-labeling helper.

        Args:
            dataset (HTRDataset): Dataset with transcriptions, embeddings, and
                alignment ids.
            backbone (HTRNet): Backbone used to harvest image descriptors with
                shape (N, D).
            projector (nn.Module): Projector module
                mapping descriptors to the word-embedding space.
            batch_size (int): Full-dataset batching size for descriptor harvest.
            device (str): Device identifier such as "cpu" or "cuda".
            reg (float): Entropic regularization for Sinkhorn OT.
            unbalanced (bool): Whether to use unbalanced OT.
            reg_m (float): Mass regularization coefficient for unbalanced OT.
            sinkhorn_kwargs (Optional[dict]): Extra keyword arguments for OT.
            k (int): Maximum number of samples to pseudo-label in one round.
        """
        self.dataset = dataset
        self.backbone = backbone
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.projector = projector.to(self.device)
        self.reg = reg
        self.unbalanced = unbalanced
        self.reg_m = reg_m
        self.sinkhorn_kwargs = sinkhorn_kwargs or {}
        self.k = k

        assert hasattr(dataset, "unique_word_embeddings"), "Dataset missing unique_word_embeddings"
        assert (
            dataset.unique_word_embeddings.ndim == 2 and dataset.unique_word_embeddings.size(0) > 0
        ), "Word-embedding matrix empty or wrong shape"
        self.word_embs = dataset.unique_word_embeddings.to(self.device)

    # ------------------------------------------------------------------
    def _calculate_ot(self, proj_feats: torch.Tensor) -> tuple[torch.Tensor, np.ndarray]:
        """Compute the OT projection of projector outputs onto word embeddings.

        Args:
            proj_feats (torch.Tensor): Projector features of shape (N, D).

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
    def _get_projector_outputs(self) -> dict[str, torch.Tensor]:
        """Run the projector on all descriptors and compute OT projections.

        Returns:
            dict[str, torch.Tensor]: Tensors required by the alignment step:
                transport plan, projected features, distance matrix,
                moved-distance scores, nearest-word ids, and the current
                aligned vector.
        """
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
        self.projector.eval()
        proj_feats = self.projector(feats_all.to(self.device)).cpu()
        assert proj_feats.shape[1] == self.word_embs.shape[1], "Embedding dim mismatch"
        assert torch.isfinite(proj_feats).all(), "Non-finite projector output"
        proj_feats_ot, plan_np = self._calculate_ot(proj_feats)

        moved_dist = (proj_feats_ot - proj_feats).norm(p=2, dim=1)
        dist_matrix = torch.cdist(proj_feats_ot, self.word_embs.cpu())
        nearest_word = dist_matrix.argmin(dim=1)
        plan_torch = torch.from_numpy(plan_np).float()

        return {
            "plan": plan_torch,
            "proj_feats": proj_feats_ot,
            "dist_matrix": dist_matrix,
            "moved_dist": moved_dist,
            "nearest_word": nearest_word,
            "aligned_all": aligned_all,
        }

    # ------------------------------------------------------------------
    def _select_candidates(
        self,
        plan: torch.Tensor,
        aligned_all: torch.Tensor,
    ) -> torch.Tensor:
        """Choose which dataset indices to pseudo-label this round.

        Args:
            plan (torch.Tensor): Transport plan with shape (N, V).
            aligned_all (torch.Tensor): Current alignment vector (N,).

        Returns:
            torch.Tensor: Selected dataset indices with shape (K,).
        """
        assert self.k >= 0, "k must be non-negative"
        plan_np = plan.numpy()
        row_sum = plan_np.sum(axis=1, keepdims=True)
        probs = np.divide(
            plan_np,
            row_sum,
            out=np.zeros_like(plan_np),
            where=row_sum != 0,
        )
        entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
        # Lower entropy indicates higher confidence.
        order_cert = np.argsort(entropy)

        mask_new = (aligned_all == -1).numpy()
        assert order_cert.size > 0, "No candidates produced"
        target_k = min(self.k, int(mask_new.sum()))

        chosen: list[int] = []
        for idx in order_cert:
            if mask_new[idx]:
                chosen.append(idx)
                if len(chosen) == target_k:
                    break
        return torch.tensor(chosen, dtype=torch.long)

    # ------------------------------------------------------------------
    def _update_dataset(self, chosen: torch.Tensor, nearest_word: torch.Tensor) -> None:
        """Write newly aligned indices back into dataset.aligned.

        Args:
            chosen (torch.Tensor): Selected dataset indices with shape (K,).
            nearest_word (torch.Tensor): Predicted word ids with shape (N,).
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
        moved_dist: torch.Tensor,
        plan: torch.Tensor,
    ) -> None:
        """Print alignment statistics for the current round.

        Args:
            chosen (torch.Tensor): Selected dataset indices with shape (K,).
            moved_dist (torch.Tensor): OT movement distances with shape (N,).
            plan (torch.Tensor): Transport plan with shape (N, V).
        """
        # report statistics for the newly pseudo-labelled samples
        chosen_list = chosen.tolist()
        if self.k > 0 and chosen_list:
            correct_new = sum(
                self.dataset.unique_words[self.dataset.aligned[i].item()]
                == self.dataset.transcriptions[i].strip()
                for i in chosen_list
            )
            print(
                f"[Align] newly pseudo-labelled {len(chosen_list)} items; "
                f"[Align] round accuracy {correct_new}/{len(chosen_list)} "
                f"({correct_new / len(chosen_list):.1%})"
            )

            for idx in random.sample(chosen_list, min(10, len(chosen_list))):
                gt = self.dataset.transcriptions[idx].strip()
                pred = self.dataset.unique_words[self.dataset.aligned[idx].item()]
                print(f"[Align] sample: '{gt}' → '{pred}'")

            md = moved_dist[chosen]
            print(f"[Align] mean moved distance: {md.mean():.4f} ± {md.std(unbiased=False):.4f}")

            row = plan[chosen_list]
            row_sum = row.sum(axis=1, keepdims=True)
            probs = np.divide(row, row_sum, out=np.zeros_like(row), where=row_sum != 0)
            ent = -(probs * np.log(probs + 1e-12)).sum(axis=1)
            print(f"[Align] mean row entropy: {ent.mean():.4f} ± {ent.std():.4f}")

        # track running accuracy over all pseudo-labelled samples
        aligned_idx = torch.nonzero(self.dataset.aligned != -1, as_tuple=True)[0]
        if aligned_idx.numel():
            correct_tot = sum(
                self.dataset.unique_words[self.dataset.aligned[i].item()]
                == self.dataset.transcriptions[i].strip()
                for i in aligned_idx.tolist()
            )
            print(
                f"[Align] cumulative accuracy: {correct_tot}/{aligned_idx.numel()} "
                f"({correct_tot / aligned_idx.numel():.1%}) – "
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

        Args:
            edit_threshold (int): Maximum allowed edit distance between decoded
                prediction and current pseudo-label.
            batch_size (int): Validation mini-batch size.
            decode_cfg (dict | None): Optional decoder settings.
            num_workers (int): Number of DataLoader workers.

        Returns:
            int: Number of aligned samples reset back to -1.
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
        dataset_name = self.dataset.get_dataset_name()
        _, i2c = load_vocab(dataset_name=dataset_name)
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
                pred_norm = str(pred).strip()
                gold_norm = str(gold_word).strip()
                if editdistance.eval(pred_norm, gold_norm) > edit_threshold:
                    self.dataset.aligned[idx] = -1
                    removed += 1
                    if hasattr(self.dataset, "transcriptions") and (
                        str(self.dataset.transcriptions[idx]).strip() != gold_norm
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
                Transport plan with shape (N, V), projected features with
                shape (N, D), and per-sample moved distance with shape (N,).
        """
        # gather features and OT statistics for the current dataset state
        out = self._get_projector_outputs()

        chosen = self._select_candidates(
            out["plan"],
            out["aligned_all"],
        )

        self._update_dataset(chosen, out["nearest_word"])
        self._log_results(
            chosen,
            out["moved_dist"],
            out["plan"].numpy() if out["plan"].numel() else np.empty((0, 0)),
        )

        # switch networks back to training mode
        self.projector.train()
        self.backbone.train()

        assert torch.isfinite(out["moved_dist"]).all(), "Non-finite moved_dist"
        return out["plan"], out["proj_feats"], out["moved_dist"]


@torch.no_grad()
def align_more_instances(
    dataset: HTRDataset,
    backbone: HTRNet,
    projector: nn.Module,
    *,
    batch_size: int = 512,
    device: str = cfg.device,
    reg: float = 0.1,
    unbalanced: bool = False,
    reg_m: float = 1.0,
    sinkhorn_kwargs: Optional[dict] = None,
    k: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align additional dataset instances with OT-based pseudo-labeling.

    This is a backward-compatible wrapper around :class:`OTAligner`. It runs one
    alignment round, updates ``dataset.aligned`` for the selected instances, and
    returns the transport-plan outputs required by the training loop.

    The function also maintains a module-level call counter. When
    ``cfg.pseudo_label_validation.enable`` is ``True`` and the current call count
    reaches ``cfg.pseudo_label_validation.start_iteration``, it triggers
    :meth:`OTAligner.validate_pseudo_labels`.

    Args:
        dataset (HTRDataset): Dataset containing image samples, word prototypes,
            and mutable alignment state.
        backbone (HTRNet): Backbone used to extract image features before
            projection and OT matching.
        projector (nn.Module): Projector module used to
            map backbone features to the word-embedding space.
        batch_size (int): Batch size used for feature harvesting and optional
            pseudo-label validation.
        device (str): Device identifier used during alignment (for example
            ``"cpu"`` or ``"cuda"``).
        reg (float): Entropic regularization coefficient for Sinkhorn OT.
        unbalanced (bool): If ``True``, use unbalanced Sinkhorn OT.
        reg_m (float): Mass regularization used in unbalanced OT.
        sinkhorn_kwargs (Optional[dict]): Extra keyword arguments forwarded to
            the OT solver.
        k (int): Number of high-confidence instances to relabel in this call.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - ``plan``: OT transport plan with shape (N, V) where ``N`` is
              dataset size and ``V`` is vocabulary size.
            - ``proj_feats``: Projected features with shape (N, D).
            - ``moved``: Per-instance movement scores used for uncertainty
              ranking, shape (N,).
    """
    global _ALIGN_CALL_COUNT
    _ALIGN_CALL_COUNT += 1

    aligner = OTAligner(
        dataset,
        backbone,
        projector,
        batch_size=batch_size,
        device=device,
        reg=reg,
        unbalanced=unbalanced,
        reg_m=reg_m,
        sinkhorn_kwargs=sinkhorn_kwargs,
        k=k,
    )
    plan, proj_feats, moved = aligner.align()

    # --- conditional pseudo-label validation ---
    vcfg = getattr(cfg, "pseudo_label_validation", None)
    if vcfg and vcfg.enable and _ALIGN_CALL_COUNT >= int(getattr(vcfg, "start_iteration", 0)):
        aligner.validate_pseudo_labels(
            edit_threshold=int(vcfg.edit_distance),
            batch_size=batch_size,
            decode_cfg=getattr(cfg, "decode_config", None),
        )

    return plan, proj_feats, moved
