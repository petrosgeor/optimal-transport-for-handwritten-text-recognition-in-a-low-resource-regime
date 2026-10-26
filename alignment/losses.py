"""Loss functions used by the alignment and pretraining pipelines.

This module contains:
- a CTC helper wrapper used by backbone training routines, and
- an OT-based projection loss with an optional supervised anchor term.
"""

import torch
import torch.nn.functional as F
from geomloss.samples_loss import SamplesLoss


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
    - supervised_weight (float): Weight applied to the supervised MSE term.
    - sinkhorn_kwargs (dict): Extra keyword args passed to GeomLoss SamplesLoss.

    """

    def __init__(
        self,
        reg: float = 0.1,
        *,
        supervised_weight: float = 1.0,
        **sinkhorn_kwargs,
    ):
        """Initialize OT loss components.

        Args:
        - reg (float): Entropic regularization for Sinkhorn.
        - supervised_weight (float): Weight for supervised MSE anchor term.
        - sinkhorn_kwargs (dict): Extra options forwarded to SamplesLoss.
        """
        super().__init__()
        self.reg = reg
        self.supervised_weight = supervised_weight
        self.sinkhorn_kwargs = sinkhorn_kwargs

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
            raise ValueError(f"aligned must be int32 or int64; got aligned.dtype={aligned.dtype}")
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
        - ValueError: If inputs are invalid, if tgt_probs sums to non-positive
          values, or if OT loss becomes non-finite.
        """
        N, M, _ = self._validate_projection_inputs(
            descriptors=descriptors,
            word_embeddings=word_embeddings,
            aligned=aligned,
            tgt_probs=tgt_probs,
        )

        # Build source and target marginals; tgt_probs is renormalized.
        a = torch.full((N,), 1.0 / N, dtype=descriptors.dtype, device=descriptors.device)
        tgt_sum = tgt_probs.sum()
        if tgt_sum.item() <= 0:
            raise ValueError("tgt_probs must sum to > 0")
        b = tgt_probs / tgt_sum

        # Compute OT loss.
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
