import torch
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

    def __init__(self, reg: float = 0.1, **sinkhorn_kwargs):
        super().__init__()
        self.reg = reg
        self.sinkhorn_kwargs = sinkhorn_kwargs

    def forward(
        self,
        descriptors: torch.Tensor,
        word_embeddings: torch.Tensor,
        aligned: torch.Tensor,
        tgt_probs: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: 1) build the cost matrix  C  (e.g. squared Euclidean distances)
        #       2) define source/target marginals  a  and  b
        #       3) call  ot.sinkhorn2(a, b, C, reg=self.reg, **self.sinkhorn_kwargs)
        #       4) optionally add supervised alignment terms
        #
        # For now we only lay out the API; we'll implement the body next.
        raise NotImplementedError(
            "ProjectionLoss forward pass not yet implemented — "
            "let’s fill this in together!"
        )
