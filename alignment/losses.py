"""Loss functions used by the alignment and pretraining pipelines.

This module contains:
- a CTC helper wrapper used by backbone training routines,
- an OT-based projection loss with an optional supervised anchor term, and
- a soft contrastive loss that builds text-space similarities from edit distance.
"""

import torch
import torch.nn.functional as F
import ot  # POT - Python Optimal Transport
from geomloss.samples_loss import SamplesLoss
from rapidfuzz import process, distance
import numpy as np


def _ctc_loss_fn(
    logits: torch.Tensor,
    targets: torch.IntTensor,
    inp_lens: torch.IntTensor,
    tgt_lens: torch.IntTensor,
) -> torch.Tensor:
    """Compute CTC loss from raw logits.

    Args:
    - logits (torch.Tensor): Unnormalized class scores with shape (T, B, C).
    - targets (torch.IntTensor): Flattened encoded targets with shape (sum(L),).
    - inp_lens (torch.IntTensor): Input sequence lengths per sample with shape (B,).
    - tgt_lens (torch.IntTensor): Target lengths per sample with shape (B,).

    Returns:
    - (torch.Tensor): Scalar loss tensor on the same device as logits.
    """
    log_probs = F.log_softmax(logits, dim=2)
    loss = F.ctc_loss(
        log_probs,
        targets,
        inp_lens,
        tgt_lens,
        reduction="mean",
        zero_infinity=True,
    )
    return loss.to(logits.device)


class ProjectionLoss(torch.nn.Module):
    """Optimal-transport projection loss with optional supervised anchors.

    The loss has two parts:
    - Sinkhorn OT term that moves projected descriptors toward word embeddings.
    - Optional supervised MSE term on samples with known alignment ids.

    Args:
    - reg (float): Entropic regularization strength for Sinkhorn.
    - unbalanced (bool): Whether to use unbalanced Sinkhorn with mass relaxation.
    - reg_m (float): Mass regularization used only in unbalanced mode.
    - supervised_weight (float): Weight applied to the supervised MSE term.
    - sinkhorn_kwargs (dict): Extra keyword args passed to GeomLoss SamplesLoss.

    """
    def __init__(
        self,
        reg: float = 0.1,
        *,
        unbalanced: bool = False,
        reg_m: float = 1.0,
        supervised_weight: float = 1.0,
        **sinkhorn_kwargs,
    ):
        """Initialize OT loss components.

        Args:
        - reg (float): Entropic regularization for Sinkhorn.
        - unbalanced (bool): Toggle unbalanced OT formulation.
        - reg_m (float): Unbalanced mass regularization parameter.
        - supervised_weight (float): Weight for supervised MSE anchor term.
        - sinkhorn_kwargs (dict): Extra options forwarded to SamplesLoss.
        """
        super().__init__()
        self.reg = reg
        self.unbalanced = unbalanced
        self.reg_m = reg_m
        self.supervised_weight = supervised_weight
        self.sinkhorn_kwargs = sinkhorn_kwargs

        if self.unbalanced:
            self.unbalanced_ot_loss = SamplesLoss(
                loss="sinkhorn", p=2, blur=self.reg, reach=self.reg_m, debias=True, **self.sinkhorn_kwargs
            )
        else:
            self.balanced_ot_loss = SamplesLoss(
                loss="sinkhorn", p=2, blur=self.reg, debias=True, **self.sinkhorn_kwargs
            )

    def _validate_projection_inputs(
        self,
        descriptors: torch.Tensor,
        word_embeddings: torch.Tensor,
        aligned: torch.Tensor,
        tgt_probs: torch.Tensor,
    ) -> tuple[int, int, int]:
        """Validate projection-loss inputs and return key shape dimensions.

        Args:
        - descriptors (torch.Tensor): Source descriptors expected with shape (N, D).
        - word_embeddings (torch.Tensor): Target embeddings expected with shape (M, D).
        - aligned (torch.Tensor): Alignment indices expected with shape (N,).
        - tgt_probs (torch.Tensor): Target marginal probabilities expected with shape (M,).

        Returns:
        - (tuple[int, int, int]): The tuple (N, M, D) from validated tensor shapes.

        Raises:
        - ValueError: If shapes, dtypes, or finiteness checks fail.
        """
        if descriptors.ndim != 2:
            raise ValueError(
                f"descriptors must have shape (N, D); got descriptors.shape={tuple(descriptors.shape)}"
            )
        if word_embeddings.ndim != 2:
            raise ValueError(
                "word_embeddings must have shape (M, D); "
                f"got word_embeddings.shape={tuple(word_embeddings.shape)}"
            )
        n_samples, feat_dim = descriptors.shape
        n_words, emb_dim = word_embeddings.shape
        if feat_dim != emb_dim:
            raise ValueError(
                f"descriptor and embedding dimensions must match; got D={feat_dim} and D'={emb_dim}"
            )
        if aligned.shape != (n_samples,):
            raise ValueError(
                f"aligned must have shape ({n_samples},); got aligned.shape={tuple(aligned.shape)}"
            )
        if tgt_probs.shape != (n_words,):
            raise ValueError(
                f"tgt_probs must have shape ({n_words},); got tgt_probs.shape={tuple(tgt_probs.shape)}"
            )

        if aligned.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"aligned must be int32 or int64; got aligned.dtype={aligned.dtype}"
            )
        if not tgt_probs.dtype.is_floating_point:
            raise ValueError(
                f"tgt_probs must be a floating-point tensor; got tgt_probs.dtype={tgt_probs.dtype}"
            )

        for tensor, name in (
            (descriptors, "descriptors"),
            (word_embeddings, "word_embeddings"),
            (tgt_probs, "tgt_probs"),
        ):
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains NaN/Inf")

        return n_samples, n_words, feat_dim

    def forward(
        self,
        descriptors: torch.Tensor,
        word_embeddings: torch.Tensor,
        aligned: torch.Tensor,
        tgt_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute projection loss for a batch of descriptors.

        Args:
        - descriptors (torch.Tensor): Source descriptors with shape (N, D).
        - word_embeddings (torch.Tensor): Target embeddings with shape (M, D).
        - aligned (torch.Tensor): Alignment ids with shape (N,). Use -1 for unknown.
        - tgt_probs (torch.Tensor): Target distribution weights with shape (M,).

        Returns:
        - (torch.Tensor): Scalar loss = OT term + weighted supervised MSE term.

        Raises:
        - ValueError: If inputs are invalid, if balanced-mode tgt_probs sums to
          non-positive values, or if OT loss becomes non-finite.
        """
        N, M, _ = self._validate_projection_inputs(
            descriptors=descriptors,
            word_embeddings=word_embeddings,
            aligned=aligned,
            tgt_probs=tgt_probs,
        )

        # Build source and target marginals. Balanced mode renormalizes tgt_probs.
        a = torch.full((N,), 1.0 / N, dtype=descriptors.dtype, device=descriptors.device)
        if not self.unbalanced:
            tgt_sum = tgt_probs.sum()
            if tgt_sum.item() <= 0:
                raise ValueError("tgt_probs must sum to > 0 in balanced mode")
            b = tgt_probs / tgt_sum
        else:
            b = tgt_probs

        # Compute OT loss.
        if self.unbalanced:
            ot_loss = self.unbalanced_ot_loss(a, descriptors, b, word_embeddings)
        else:
            ot_loss = self.balanced_ot_loss(a, descriptors, b, word_embeddings)
        if not torch.isfinite(ot_loss).all():
            raise ValueError("OT loss is NaN/Inf")

        # Compute supervised anchor term.
        aligned_indices = torch.where(aligned != -1)[0]
        if aligned_indices.numel() == 0:
            distance_loss = torch.tensor(0.0, device=descriptors.device, dtype=descriptors.dtype)
        else:
            aligned_descriptors = descriptors[aligned_indices]
            corresp_word_embeddings = word_embeddings[aligned[aligned_indices]]
            distance_loss = F.mse_loss(aligned_descriptors, corresp_word_embeddings)
        return ot_loss + self.supervised_weight * distance_loss


# ------------------------------------------------------------------
# Soft Contrastive / InfoNCE with continuous positives (Euclidean)
# ------------------------------------------------------------------
from .ctc_utils import _unflatten_targets


class SoftContrastiveLoss(torch.nn.Module):
    """Soft InfoNCE-style contrastive loss using text edit distance.

    The image-space similarity comes from pairwise feature distances, while
    text-space similarity is computed from pairwise Levenshtein distances
    between decoded target sequences.

    Args:
    - tau (float): Temperature for image-space similarities.
    - T_txt (float): Temperature for text-space similarities.
    - eps (float): Small constant for numerical stability.

    """
    def __init__(self, tau: float = .07, T_txt: float = 1.0, eps: float = 1e-8):
        """Initialize contrastive-loss temperatures.

        Args:
        - tau (float): Temperature for image-space pair similarities.
        - T_txt (float): Temperature for text-space pair similarities.
        - eps (float): Numerical-stability constant for division and log.
        """
        super().__init__()
        self.tau   = tau
        self.T_txt = T_txt
        self.eps   = eps

    def forward(self, feats: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Compute soft contrastive loss for one minibatch.

        Args:
        - feats (torch.Tensor): Feature matrix with shape (B, D).
        - targets (torch.Tensor): Flattened encoded CTC targets with shape (sum(L),).
        - lengths (torch.Tensor): Target lengths with shape (B,).

        Returns:
        - (torch.Tensor): Scalar contrastive loss. Returns 0 when B < 2.
        """
        B = len(lengths)
        if B < 2:
            return torch.tensor(0.0, device=feats.device, dtype=feats.dtype)

        # ---- 1. pairwise Euclidean distances in descriptor space ----
        dists = torch.cdist(feats, feats, p=2)        # (B, B)
        sim_img = torch.exp(-dists / self.tau)        # convert to similarity

        # ---- 2. edit-distance matrix (vectorized) --------------------
        with torch.no_grad():
            # Convert flattened targets to list of lists
            unflattened_targets = _unflatten_targets(targets, lengths)
            # Use rapidfuzz for fast, parallelized pairwise edit distance.
            edit_np = process.cdist(
                unflattened_targets, unflattened_targets,
                scorer=distance.Levenshtein.distance, workers=-1
            ).astype(np.float32)
            edit = torch.from_numpy(edit_np).to(device=feats.device)
        sim_txt = torch.exp(-edit / self.T_txt)       # soft positives

        # ---- 3. InfoNCE objective ------------------------------------
        # exclude diagonal (self‑similarity) from both sums
        eye = torch.eye(B, device=feats.device)
        sim_img = sim_img * (1 - eye)
        sim_txt = sim_txt * (1 - eye)

        numerator   = (sim_txt * sim_img).sum(dim=1)          # (B,)
        denominator = sim_img.sum(dim=1) + self.eps           # (B,)
        loss = -torch.log((numerator + self.eps) / denominator).mean()
        return loss
