"""Utility functions for aligning dataset instances to unique words."""

from typing import Optional, Tuple, List, Sequence, Union
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
    dataset: Union[HTRDataset, FusedHTRDataset],
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
    dataset : HTRDataset | FusedHTRDataset
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




class ProjectionAligner:
    """Helper class implementing projection-based pseudo-labelling."""

    def __init__(
        self,
        dataset: FusedHTRDataset,
        backbone: HTRNet,
        projectors: Sequence[nn.Module],
        *,
        batch_size: int = 512,
        device: str = cfg.device,
        k: int = 0,
    ) -> None:
        """Initialise the projection aligner.

        Args:
            dataset (HTRDataset | FusedHTRDataset): Dataset providing images and alignment info.
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

        self.dataset = dataset
        self.backbone = backbone
        self.projectors = projectors
        self.batch_size = batch_size
        self.device = device
        self.k = k

        self.word_embeddings = dataset.unique_word_embeddings.to(self.device)
        self.real_word_indices = dataset.real_word_indices
        self.synth_word_indices = dataset.synth_word_indices
        self.is_real = dataset._is_real
        self.is_syn = ~self.is_real


    def _get_projector_features(self):
        """Harvests backbone features and computes projector outputs.

        This method runs the backbone on the entire dataset to get visual
        descriptors, then passes these descriptors through each projector in
        the ensemble to obtain their corresponding embeddings in the word
        embedding space.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - A 3D tensor of projected features with shape
              (n_projectors, n_samples, embedding_dim).
            - A 1D tensor of alignment information for each sample.
        """
        # 1. Harvest visual descriptors from the backbone for the entire dataset.
        feats_all, aligned_all = harvest_backbone_features(
            self.dataset,
            self.backbone,
            batch_size=self.batch_size,
            device=self.device,
        )

        word_emb_dim = self.projectors[0].output_dim
        n_samles = feats_all.shape[0]

        # 2. Initialize a tensor to store the outputs of each projector.
        # The shape is (num_projectors, num_samples, word_embedding_dim).
        proj_feats = torch.zeros((len(self.projectors), n_samles, word_emb_dim))
        
        # 3. Pass the descriptors through each projector to get embeddings.
        with torch.no_grad():
            for i, proj in enumerate(self.projectors):
                proj_feats[i] = proj(feats_all.to(self.device)).cpu()
        
        return proj_feats, aligned_all


    @torch.no_grad()
    def align(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pseudo‑label up to ``self.k`` still‑unaligned samples by nearest‑
        neighbour assignment in word‑embedding space and update
        ``self.dataset.aligned`` *in‑place*.

        Returns
        -------
        proj_feats_mean : torch.Tensor
            Mean projector output for every dataset item (``(N, E)`` on *CPU*).
        moved : torch.Tensor
            1‑D tensor (length ``N``) whose *i*‑th entry is the distance
            between sample *i* and its assigned embedding (``inf`` if the
            sample is still un‑aligned).
        """
        # ── 1. Ensemble projections for the whole dataset ─────────────────
        proj_feats, aligned_all = self._get_projector_features()          # (P, N, E), (N,)
        proj_feats_mean = proj_feats.mean(dim=0)                          # (N, E)

        # ── 2. Work on *unaligned* real samples only ----------------------
        unaligned_mask = aligned_all == -1                                # (N,)
        if unaligned_mask.sum() == 0:                                     # nothing to do
            moved = torch.full((proj_feats_mean.size(0),), float("inf"))
            return torch.empty(0), proj_feats_mean.cpu(), moved

        proj_feats_acceptable = proj_feats_mean[unaligned_mask].to(self.device)  # (U, E)

        # ── 3. Distance to every word embedding (exclude synthetic‑only) ──
        dist_matrix = torch.cdist(proj_feats_acceptable,
                                  self.word_embeddings, p=2)              # (U, V)
        dist_matrix[:, self.synth_word_indices.to(self.device)] = float("inf")

        min_dists, pred_word_idx = dist_matrix.min(dim=1)                 # (U,)

        # ── 4. Pick the *k* most confident predictions (smallest distance) ─
        k = self.k if self.k > 0 else min_dists.numel()
        k = min(k, min_dists.numel())
        topk_local = torch.argsort(min_dists)[:k]                         # (k,)

        # ── 5. Map *local* rows back to *global* dataset indices ─────────
        unaligned_global = torch.nonzero(unaligned_mask, as_tuple=True)[0]  # (U,)
        chosen_global   = unaligned_global[topk_local]                    # (k,)
        chosen_words    = pred_word_idx[topk_local].cpu().to(
                             dtype=self.dataset.aligned.dtype)            # (k,)

        # ── 6. Update the dataset alignment tensor in‑place --------------
        self.dataset.aligned[chosen_global] = chosen_words                # ✔ global mapping

        # ── 7. Build "moved" vector for the whole dataset ----------------
        moved = torch.full((proj_feats_mean.size(0),), float("inf"),
                           device=min_dists.device)
        moved[unaligned_global] = min_dists
        moved = moved.cpu()

        # API compatibility: return dummy transport projections + moved
        proj_feats_mean.cpu(), moved


        




@torch.no_grad()
def align_more_instances(
    dataset: FusedHTRDataset,
    backbone: HTRNet,
    projectors: Sequence[nn.Module],
    *,
    batch_size: int = 512,
    device: str = cfg.device,
    k: int = 0,
    metric: str = "gap",
    agree_threshold: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pseudo-label new dataset instances using the projection ensemble.

    Args:
        dataset (HTRDataset | FusedHTRDataset): Dataset to be aligned.
        backbone (HTRNet): Visual backbone network.
        projectors (Sequence[nn.Module]): Ensemble of projector networks.
        batch_size (int): Mini-batch size for descriptor extraction.
        device (str): Device used during alignment.
        k (int): Number of samples to pseudo-label.
        metric (str): Selection metric ('gap', 'variance', 'closest').
        agree_threshold (int): Minimum projector agreement for a label.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            Dummy transport plan, mean projector features and zeros for moved distance.

    When ``cfg.pseudo_label_validation.enable`` is ``True`` this function
    calls :meth:`ProjectionAligner.validate_pseudo_labels` after the alignment
    round once ``align_more_instances`` has been invoked at least
    ``start_iteration`` times.
    """
    global _ALIGN_CALL_COUNT
    _ALIGN_CALL_COUNT += 1



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
    proj_feats, moved = aligner.align()

    return proj_feats, moved
