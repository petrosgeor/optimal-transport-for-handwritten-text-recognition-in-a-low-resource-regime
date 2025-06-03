import torch
import torch.nn.functional as F
import ot  # POT - Python Optimal Transport


class ProjectionLoss(torch.nn.Module):
    """
    Entropic-regularised OT (Sinkhorn-2) projection loss.

    Parameters
    ----------
    reg : float, optional
        Entropic regularisation strength passed to ``ot.sinkhorn2``.  Default = 0.1.
    sinkhorn_kwargs : dict, optional
        Extra keyword arguments forwarded to ``ot.sinkhorn2`` (e.g. max_iter, tol).

    Forward Inputs
    --------------
    descriptors : torch.Tensor                # shape (N, d)
        Source features extracted by the network.
    word_embeddings : torch.Tensor            # shape (M, d)
        2-D “vocabulary” embedding coordinates (target support).
    aligned : torch.Tensor                    # shape (N,)
        For each descriptor, the index of its aligned word in ``word_embeddings``  
        (‒1 if no alignment available).
    tgt_probs : torch.Tensor                  # shape (M,)
        Marginal probabilities for the target distribution (must sum to 1).

    Returns
    -------
    loss : torch.Tensor
        Scalar OT loss (differentiable w.r.t. `descriptors`).
    """

    def __init__(self, reg: float = 0.1, *, unbalanced: bool = False, reg_m: float = 1.0, **sinkhorn_kwargs):
        super().__init__()
        self.reg = reg
        self.unbalanced = unbalanced
        self.reg_m = reg_m
        self.sinkhorn_kwargs = sinkhorn_kwargs

    def forward(
        self,
        descriptors: torch.Tensor,
        word_embeddings: torch.Tensor,
        aligned: torch.Tensor,
        tgt_probs: torch.Tensor,
    ) -> torch.Tensor:
        # create uniform source distribution
        n = descriptors.shape[0]
        a = torch.full((n,), 1.0 / n, dtype=descriptors.dtype, device=descriptors.device)

        # target distribution
        if not self.unbalanced:
            b = tgt_probs / tgt_probs.sum()
        else:
            b = tgt_probs

        # cost matrix between descriptors and word embeddings (euclidean distance)
        C = torch.cdist(descriptors, word_embeddings, p=2)

        # compute OT loss depending on balanced/unbalanced mode
        if self.unbalanced:
            ot_loss = ot.unbalanced.sinkhorn_unbalanced2(
                a,
                b,
                C,
                reg=self.reg,
                reg_m=self.reg_m,
                **self.sinkhorn_kwargs,
            )
        else:
            ot_loss = ot.sinkhorn2(
                a,
                b,
                C,
                reg=self.reg,
                **self.sinkhorn_kwargs,
            )

        # compute supervised alignment loss when alignments are provided
        aligned_indices = torch.where(aligned != -1)[0]
        if aligned_indices.numel() == 0:
            distance_loss = torch.tensor(0.0, device=descriptors.device, dtype=descriptors.dtype)
        else:
            aligned_descriptors = descriptors[aligned_indices]
            corresp_word_embeddings = word_embeddings[aligned[aligned_indices]]
            distance_loss = F.mse_loss(aligned_descriptors, corresp_word_embeddings)

        return ot_loss + distance_loss
