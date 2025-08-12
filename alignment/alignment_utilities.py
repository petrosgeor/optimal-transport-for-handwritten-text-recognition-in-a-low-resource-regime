"""Utility functions for aligning dataset instances to unique words."""

from typing import Optional, Tuple, List, Sequence
from contextlib import ExitStack
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
from torch.utils.data import DataLoader, Subset
# (CTC decode utilities not needed after removing pseudo-labelling)

cfg = OmegaConf.load("alignment/alignment_configs/trainer_config.yaml")

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
    descriptors. The backbone is temporarily switched to evaluation mode on
    ``device`` and restored to its original training/eval mode afterwards.

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
    # Preserve original training/eval mode and temporarily switch to eval
    was_training = backbone.training
    backbone = backbone.to(device)
    backbone.eval()

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

    # Restore original module mode (train/eval)
    backbone.train(was_training)

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
    # 1) Harvest descriptors once (harvester already eval+no_grad)
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

    # 3) Force projectors to eval, remember original modes
    _proj_was_training = [p.training for p in projectors]
    for p in projectors:
        p.eval().to(device_t)

    try:
        # 4) Ensure no gradients or autograd graphs are created
        with torch.inference_mode():
            resp_list: list[np.ndarray] = []
            for proj in projectors:
                proj_feats = proj(feats_all.to(device_t)).cpu()
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

    finally:
        # 5) Restore original projector modes
        for p, was_tr in zip(projectors, _proj_was_training):
            p.train(was_tr)

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
class IndexedSubset(Subset):
    """Subset wrapper that appends the original dataset index to each item.

    Returns tuples like ``(..., orig_index)`` to enable slicing global tensors
    (e.g., responsibilities) stored in dataset order.
    """

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        orig_index = self.indices[idx]
        if not isinstance(data, tuple):
            data = (data,)
        return (*data, orig_index)
