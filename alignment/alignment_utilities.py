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
import ot

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
        debug_checks: bool = True,
    ) -> None:
        """Initialise the projection aligner.

        Args:
            dataset (HTRDataset | FusedHTRDataset): Dataset providing images and alignment info.
            backbone (HTRNet): Visual backbone network.
            projectors (Sequence[nn.Module]): Projector ensemble.
            batch_size (int): Mini-batch size during descriptor harvesting.
            device (str): Device used for alignment.
            k (int): Number of least-moved descriptors to pseudo-label.
            debug_checks (bool): Whether to run sanity checks during alignment.

        Returns:
            None
        """

        self.dataset = dataset
        self.backbone = backbone
        self.projectors = projectors
        self.batch_size = batch_size
        self.device = device
        self.k = k
        self.debug_checks = debug_checks

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

        # set projectors in evaluation mode
        for proj in self.projectors: 
            proj.eval()
        
        # 3. Pass the descriptors through each projector to get embeddings.
        with torch.no_grad():
            for i, proj in enumerate(self.projectors):
                proj_feats[i] = proj(feats_all.to(self.device)).cpu()
        
        for proj in self.projectors: 
            proj.train()

        return proj_feats, aligned_all

    @torch.no_grad()
    def calculate_ot_projections(
        self,
        pa: torch.Tensor,
        X: torch.Tensor,
        pb: torch.Tensor,
        Y: torch.Tensor,
        reg: float = 0.1,
        *,
        unbalanced: bool = False,
        reg_m: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the barycentric projections of source features X onto target features Y
        by solving an entropic-regularised optimal transport (OT) problem entirely in PyTorch.

        This variant:
        - Uses the input mass vectors pa and pb as-provided (no internal ℓ¹ normalisation).
        - Computes projections for all samples, even if a sample receives zero total mass.

        Parameters
        ----------
        pa : torch.Tensor, shape (N,)
            Source mass distribution over X. 
        X : torch.Tensor, shape (N, D)
            Source feature matrix.
        pb : torch.Tensor, shape (M,)
            Target mass distribution over Y.
        Y : torch.Tensor, shape (M, D)
            Target feature matrix.
        reg : float, default 0.1
            Entropic regularisation strength.
        unbalanced : bool, default False
            If True, uses the unbalanced OT formulation (requires  reg_m).
        reg_m : float, default 1.0
            Mass regularisation parameter when unbalanced=True.

        Returns
        -------
        projections : torch.Tensor, shape (N, D)
            Barycentric projections of X in the space of Y:
                projection_i = (∑_j T_ij * Y_j) / (∑_j T_ij)
            (division by zero yields NaN/Inf as appropriate).
        plan : torch.Tensor, shape (N, M)
            Optimal transport plan solving the regularised OT problem.
        """
        # Bring pa, pb to same device and ensure contiguity
        pa = pa.to(X.device, non_blocking=True).contiguous()
        pb = pb.to(Y.device, non_blocking=True).contiguous()

        # Squared-Euclidean cost matrix M_{i,j} = ||x_i - y_j||^2
        M = torch.cdist(X, Y, p=2).pow(2)  # (N, M)

        # Solve OT in PyTorch backend
        if unbalanced:
            plan = ot.unbalanced.sinkhorn_unbalanced(
                pa, pb, M, reg=reg, reg_m=reg_m)                                # (N, M)
        else:
            plan = ot.sinkhorn(pa, pb, M, reg=reg)  # (N, M)

        # Barycentric projections for all samples
        row_sum = plan.sum(dim=1, keepdim=True)  # (N, 1)
        projections = (plan @ Y) / row_sum       # (N, D)

        return projections, plan


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
        topk_local = torch.argsort(min_dists)[:k].cpu()                         # (k,)

        # ── 5. Map *local* rows back to *global* dataset indices ─────────
        unaligned_global = torch.nonzero(unaligned_mask, as_tuple=True)[0]  # (U,)
        chosen_global   = unaligned_global[topk_local]                    # (k,)
        chosen_words    = pred_word_idx[topk_local].cpu().to(
                             dtype=self.dataset.aligned.dtype)            # (k,)

        if self.debug_checks:
            prev_aligned = self.dataset.aligned.clone()
            prev_real_vocab = self.real_word_indices.clone()
            prev_syn_vocab = self.synth_word_indices.clone()
            vocab_size_before = len(self.dataset.unique_words)

        # ── 6. Update the dataset alignment tensor in‑place --------------
        self.dataset.aligned[chosen_global] = chosen_words                # ✔ global mapping

        if self.debug_checks:
            self._assert_alignment_invariants(
                prev_aligned, prev_real_vocab, prev_syn_vocab, vocab_size_before
            )

        self._log_results(chosen_global)

        # ── 7. Build "moved" vector for the whole dataset ----------------
        moved = torch.full((proj_feats_mean.size(0),), float("inf"),
                           device=min_dists.device)
        moved[unaligned_global] = min_dists
        moved = moved.cpu()

        # Return mean projector features and moved distance
        return proj_feats_mean.cpu(), moved

    def _assert_alignment_invariants(
    self,
    prev_aligned: torch.Tensor,
    prev_real_vocab: torch.Tensor,
    prev_syn_vocab: torch.Tensor,
    vocab_size_before: int,
) -> None:
        """Check that this alignment step did not break any invariants.

        Parameters
        ----------
        prev_aligned : torch.Tensor
            Copy of ``dataset.aligned`` *before* the current update.
        prev_real_vocab : torch.Tensor
            Cached indices of real-word vocabulary entries.
        prev_syn_vocab : torch.Tensor
            Cached indices of synthetic-only vocabulary entries.
        vocab_size_before : int
            Length of ``dataset.unique_words`` before the update.

        Returns
        -------
        None
        """
        # 1. Already-aligned samples must never be overwritten.
        changed = (prev_aligned != -1) & (self.dataset.aligned != prev_aligned)
        assert not changed.any(), "Aligned samples were modified"

        # 2. Isolate **only** the rows that were un-aligned and are now labelled.
        mask_new = (prev_aligned == -1) & (self.dataset.aligned != -1)
        new_ids = self.dataset.aligned[mask_new]

        # 3. Every fresh label must refer to an existing real-word embedding.
        assert torch.isin(new_ids, self.real_word_indices).all(), (
            "New labels must come from real vocabulary"
        )

        # 4. The vocabulary itself and its cached partitions must stay frozen.
        assert len(self.dataset.unique_words) == vocab_size_before, (
            "Vocabulary size changed during alignment"
        )
        assert torch.equal(self.real_word_indices, prev_real_vocab), (
            "Real-word indices changed"
        )
        assert torch.equal(self.synth_word_indices, prev_syn_vocab), (
            "Synthetic-word indices changed"
        )

        # 5. No more than ``k`` new samples should have been pseudo-labelled.
        if self.k > 0:
            assert new_ids.numel() <= self.k, "More than k new labels assigned"

    def _log_results(self, new_indices: torch.Tensor) -> None:
        """
        Print pseudo‑label accuracy **for this round** *and* the
        **cumulative** accuracy achieved so far ‒ both restricted to
        **real** images (synthetic samples are ignored).

        Parameters
        ----------
        new_indices : torch.Tensor
            1‑D CPU tensor with the *global* dataset indices that have been
            pseudo‑labelled in the current call to :meth:`align`.
            Every index refers to ``self.dataset``.

        Notes
        -----
        • A prediction is considered *correct* when the word obtained from
          ``self.dataset.unique_words[self.dataset.aligned[i]]`` matches
          the ground‑truth transcription in
          ``self.dataset.transcriptions[i]`` (case‑ and whitespace‑
          insensitive).

        • The method has **no side‑effects** apart from its printouts.
        """
        # ── 0. Early exit when nothing new was labelled ────────────────
        if new_indices.numel() == 0:
            print("[Align] No new pseudo‑labels this round.")
            return

        # Helper that returns True when prediction equals ground truth
        def _is_correct(idx: int) -> bool:
            word_id = int(self.dataset.aligned[idx])
            if word_id == -1:            # still unlabelled ‒ ignore
                return False
            pred = self.dataset.unique_words[word_id].strip().lower()
            gt   = self.dataset.transcriptions[idx].strip().lower()
            return pred == gt

        # ── 1. Accuracy for the *newly* pseudo‑labelled real images ────
        correct_now, total_now = 0, 0
        for idx in new_indices.tolist():
            if not bool(self.is_real[idx]):      # skip synthetic images
                continue
            total_now += 1
            correct_now += int(_is_correct(idx))
        acc_now = correct_now / total_now if total_now else float("nan")

        # ── 2. Cumulative accuracy on *all* aligned real images ────────
        real_and_aligned = torch.nonzero(
            self.is_real & (self.dataset.aligned != -1), as_tuple=True
        )[0].tolist()
        total_cum = len(real_and_aligned)
        correct_cum = sum(int(_is_correct(i)) for i in real_and_aligned)
        acc_cum = correct_cum / total_cum if total_cum else float("nan")

        # ── 3. Display the results ─────────────────────────────────────
        if total_now:
            print(
                f"[Align] Round accuracy (real): "
                f"{correct_now}/{total_now} correct ({acc_now:.2%}) "
            )
            print(
                f"[Align] Cumulative accuracy (real): "
                f"{correct_cum}/{total_cum} ({acc_cum:.2%}) "
            )

            # ── 4. Print 5 random predictions and ground truths ─────────
            # Filter new real indices for sampling
            new_real_indices = [idx for idx in new_indices.tolist() if self.is_real[idx]]
            if new_real_indices:
                # Sample up to 5 indices
                sampled_indices = random.sample(new_real_indices, min(5, len(new_real_indices)))
                print("[Align] Sample predictions (index: prediction -> ground truth):")
                for idx in sampled_indices:
                    word_id = int(self.dataset.aligned[idx])
                    pred_word = self.dataset.unique_words[word_id] if word_id != -1 else "<UNLABELLED>"
                    true_word = self.dataset.transcriptions[idx]
                    print(f"  {idx}: '{pred_word}' -> '{true_word}'")


        




@torch.no_grad()
def align_more_instances(
    dataset: FusedHTRDataset,
    backbone: HTRNet,
    projectors: Sequence[nn.Module],
    *,
    batch_size: int = 512,
    device: str = cfg.device,
    k: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pseudo-label new dataset instances using the projection ensemble.

    Args:
        dataset (HTRDataset | FusedHTRDataset): Dataset to be aligned.
        backbone (HTRNet): Visual backbone network.
        projectors (Sequence[nn.Module]): Ensemble of projector networks.
        batch_size (int): Mini-batch size for descriptor extraction.
        device (str): Device used during alignment.
        k (int): Number of samples to pseudo-label.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            Mean projector features and moved distance for each sample.

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
    )
    proj_feats, moved = aligner.align()

    return proj_feats, moved
