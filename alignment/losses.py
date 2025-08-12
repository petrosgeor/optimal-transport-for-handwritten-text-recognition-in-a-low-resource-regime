import torch
import torch.nn.functional as F
import ot  # POT - Python Optimal Transport
from geomloss.samples_loss import SamplesLoss
from rapidfuzz import process, distance
import numpy as np
import math
from typing import Optional, List, Dict

def _ctc_loss_fn(
    logits: torch.Tensor,
    targets: torch.IntTensor,
    inp_lens: torch.IntTensor,
    tgt_lens: torch.IntTensor,
) -> torch.Tensor:
    """A thin wrapper around `torch.nn.functional.ctc_loss` that takes *logits*."""
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
    """
    Entropic‑regularised optimal‑transport projection loss with optional
    unbalanced formulation.

    Parameters
    ----------
    reg : float, optional
        Entropic regularisation strength passed to ``ot.sinkhorn2`` or
        ``ot.unbalanced.sinkhorn_unbalanced2``.  Default = 0.1.
    unbalanced : bool, optional
        If ``True`` use the unbalanced OT formulation
        ``ot.unbalanced.sinkhorn_unbalanced2`` (requires an additional
        ``reg_m`` parameter that can be passed through ``sinkhorn_kwargs``);
        otherwise use the balanced formulation ``ot.sinkhorn2``.
        Default = ``False`` (balanced).
    supervised_weight : float, optional
        Scale for the supervised descriptor distance term.  Default = 1.0.
    sinkhorn_kwargs : dict, optional
        Extra keyword arguments forwarded to the solver (e.g. max_iter, tol,
        log, reg_m for the unbalanced case).

    Forward Inputs
    --------------
    descriptors : torch.Tensor                # shape (N, d)
        Source features extracted by the network.
    word_embeddings : torch.Tensor            # shape (M, d)
        2‑D “vocabulary” embedding coordinates (target support).
    aligned : torch.Tensor                    # shape (N,)
        For each descriptor, the index of its aligned word in
        ``word_embeddings`` (‒1 if no alignment available).
    tgt_probs : torch.Tensor                  # shape (M,)
        Marginal probabilities for the target distribution.  For the balanced
        version it will be renormalised to sum to 1.

    Returns
    -------
    loss : torch.Tensor
        Scalar loss = OT loss + supervised alignment loss.
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

    def forward(
        self,
        descriptors: torch.Tensor,
        word_embeddings: torch.Tensor,
        aligned: torch.Tensor,
        tgt_probs: torch.Tensor,
    ) -> torch.Tensor:
        # sanity checks on shapes
        assert descriptors.ndim == 2, "descriptors must be 2-D (N, d)"
        assert word_embeddings.ndim == 2, "word_embeddings must be 2-D (M, d)"
        N, d = descriptors.shape
        M, d2 = word_embeddings.shape
        assert d == d2, "descriptor and embedding dimensions differ"
        assert aligned.shape == (N,), "aligned must have shape (N,)"
        assert tgt_probs.shape == (M,), "tgt_probs must have shape (M,)"

        # check for NaN or Inf values
        for tensor, name in [
            (descriptors, "descriptors"),
            (word_embeddings, "word_embeddings"),
            (tgt_probs, "tgt_probs"),
        ]:
            assert not torch.isnan(tensor).any(), f"{name} contains NaN values"
            assert not torch.isinf(tensor).any(), f"{name} contains infinite values"
        # create uniform source distribution
        n = descriptors.shape[0]
        a = torch.full((n,), 1.0 / n, dtype=descriptors.dtype, device=descriptors.device)
        # target distribution
        if not self.unbalanced:
            b = tgt_probs / tgt_probs.sum()
        else:
            b = tgt_probs

        # compute OT loss depending on balanced/unbalanced mode
        if self.unbalanced:
            ot_loss = self.unbalanced_ot_loss(a, descriptors, b, word_embeddings)
        else:
            ot_loss = self.balanced_ot_loss(a, descriptors, b, word_embeddings)
        assert not torch.isnan(ot_loss).any(), "OT loss is NaN"
        assert not torch.isinf(ot_loss).any(), "OT loss is infinite"

        # compute supervised alignment loss when alignments are provided
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
from .ctc_utils import _unflatten_targets, ctc_target_probability


class SoftContrastiveLoss(torch.nn.Module):
    """
    InfoNCE‑style loss that pulls together image descriptors whose
    transcripts have small Levenshtein distance.

    Parameters
    ----------
    tau   : float  – temperature in image space (distance → similarity).
    T_txt : float  – temperature in transcript space (controls softness).
    eps   : float  – numeric stability.
    """
    def __init__(self, tau: float = .07, T_txt: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.tau   = tau
        self.T_txt = T_txt
        self.eps   = eps

    def forward(self, feats: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        feats        : (B, D) float tensor (output of HTRNet, 3ʳᵈ element)
        targets      : (sum(L),) int tensor of encoded labels
        lengths      : (B,) int tensor of label lengths
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


def _em_word_loss_for_batch(
    logits_btc: torch.Tensor,
    batch_indices: List[int],
    R_full: Optional[torch.Tensor],
    unique_words: List[str],
    c2i_map: Dict[str, int],
    k_top: int = 5,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Expected NLL over top‑K candidate words for an unaligned mini‑batch.

    Args:
        logits_btc (torch.Tensor): Network logits with shape ``(T, B, C)``.
        batch_indices (list[int]): Dataset indices corresponding to the B items.
        R_full (torch.Tensor | None): Responsibility matrix of shape ``(N, V)``
            on CPU; pass ``None`` to disable the loss.
        unique_words (list[str]): Vocabulary tokens indexed by columns of ``R_full``.
        c2i_map (dict[str,int]): Character‑to‑index map for CTC encoding.
        k_top (int): How many highest‑weight candidates to keep per row.
        eps (float): Minimum probability used inside the logarithm for stability.

    Returns:
        torch.Tensor: Scalar tensor with the averaged EM loss for this batch.
    """
    if R_full is None:
        return torch.tensor(0.0, device=logits_btc.device)
    T, B, C = logits_btc.shape
    losses = []
    for b, idx in enumerate(batch_indices):
        row = R_full[idx].cpu().numpy()
        if row.ndim != 1 or row.size == 0:
            continue
        # select top‑K entries by weight
        ksel = min(int(max(1, k_top)), row.size)
        cand_idx = np.argpartition(row, -ksel)[-ksel:]
        weights = row[cand_idx]
        s = weights.sum()
        if s <= 0:
            continue
        weights = weights / s
        logits_one = logits_btc[:, b, :]
        nll = 0.0
        for j, w_idx in enumerate(cand_idx.tolist()):
            w = unique_words[w_idx]
            p = float(ctc_target_probability(logits_one, f" {w} ", c2i_map))
            nll += -weights[j] * math.log(max(p, eps))
        losses.append(nll)
    if not losses:
        return torch.tensor(0.0, device=logits_btc.device)
    return torch.tensor(sum(losses) / len(losses), device=logits_btc.device)


def expected_phoc_from_responsibilities(
    R_batch: torch.Tensor,
    phoc_vocab: torch.Tensor,
) -> torch.Tensor:
    """
    Compute expected PHOC targets given responsibilities and a PHOC matrix.

    Args:
        R_batch (torch.Tensor): Responsibilities for a mini‑batch of size ``B``
            over a vocabulary of size ``V``; shape ``(B, V)``. Rows are assumed
            to be non‑negative and sum to 1 (row‑stochastic), but the function
            does not enforce nor require exact normalisation.
        phoc_vocab (torch.Tensor): PHOC descriptor matrix for the vocabulary,
            shape ``(V, P)`` where ``P`` is the PHOC dimensionality, typically
            built with ``build_phoc_description(unique_words, c2i, levels)``.

    Returns:
        torch.Tensor: Expected PHOC targets of shape ``(B, P)`` computed as
        ``R_batch @ phoc_vocab``. The result is on the same device and dtype as
        the inputs after standard matmul broadcasting rules.
    """
    if R_batch.ndim != 2 or phoc_vocab.ndim != 2:
        raise ValueError("R_batch and phoc_vocab must be 2-D tensors")
    if R_batch.size(1) != phoc_vocab.size(0):
        raise ValueError("Incompatible shapes: R is (B,V) and PHOC is (V,P)")
    return R_batch @ phoc_vocab
