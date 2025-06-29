import torch
import torch.nn.functional as F
import ot  # POT - Python Optimal Transport
from geomloss.samples_loss import SamplesLoss


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
        targets.cpu(),
        inp_lens.cpu(),
        tgt_lens.cpu(),
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

    def forward(self, feats: torch.Tensor, transcripts: list[str]) -> torch.Tensor:
        """
        feats        : (B, D) float tensor (output of HTRNet, 3ʳᵈ element)
        transcripts  : list of B padded strings (" word ")
        """
        # ---- 1. pairwise Euclidean distances in descriptor space ----
        dists = torch.cdist(feats, feats, p=2)        # (B, B)
        sim_img = torch.exp(-dists / self.tau)        # convert to similarity

        # ---- 2. edit‑distance matrix on CPU (small batches) ----------
        import editdistance
        with torch.no_grad():
            B = len(transcripts)
            edit = torch.zeros(B, B, dtype=torch.float32, device=feats.device)
            for i in range(B):
                for j in range(i + 1, B):
                    e = editdistance.eval(transcripts[i].strip(),
                                           transcripts[j].strip())
                    edit[i, j] = edit[j, i] = float(e)
        sim_txt = torch.exp(-edit / self.T_txt)       # soft positives

        # ---- 3. InfoNCE objective ------------------------------------
        # exclude diagonal (self‑similarity) from both sums
        eye = torch.eye(sim_img.size(0), device=feats.device)
        sim_img = sim_img * (1 - eye)
        sim_txt = sim_txt * (1 - eye)

        numerator   = (sim_txt * sim_img).sum(dim=1)          # (B,)
        denominator = sim_img.sum(dim=1) + self.eps           # (B,)
        loss = -torch.log((numerator + self.eps) / denominator).mean()
        return loss