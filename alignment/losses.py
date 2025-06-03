import torch
import ot  # POT - Python Optimal Transport


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

    def __init__(self, *, reg: float = 0.1, unbalanced: bool = False, **sinkhorn_kwargs):
        super().__init__()
        self.reg = reg
        self.unbalanced = unbalanced
        # we keep kwargs mutable copy so we can safely pop items
        self.sinkhorn_kwargs = dict(sinkhorn_kwargs)

    def forward(
        self,
        descriptors: torch.Tensor,
        word_embeddings: torch.Tensor,
        aligned: torch.Tensor,
        tgt_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the OT loss plus supervised alignment loss."""
        if descriptors.ndim != 2 or word_embeddings.ndim != 2:
            raise ValueError("descriptors and word_embeddings must be 2‑D tensors")

        device = descriptors.device
        dtype = descriptors.dtype

        n, _ = descriptors.shape
        m, _ = word_embeddings.shape

        # --- source & target marginals -------------------------------------------------
        src_probs = torch.full((n,), 1.0 / n, device=device, dtype=dtype)  # uniform

        if self.unbalanced:
            # keep original target probs (can be unnormalised)
            b = tgt_probs.to(device=device, dtype=dtype)
        else:
            # balanced OT requires histograms to sum to 1
            if tgt_probs.sum() <= 0:
                raise ValueError("tgt_probs must have positive mass")
            b = (tgt_probs / tgt_probs.sum()).to(device=device, dtype=dtype)

        # --- cost matrix ---------------------------------------------------------------
        # Euclidean distance between every descriptor and every word embedding
        C = torch.cdist(descriptors, word_embeddings, p=2)  # shape (n, m)

        # --- OT loss -------------------------------------------------------------------
        if self.unbalanced:
            # sinkhorn_kwargs may contain reg_m; if not provided use reg
            kwargs = dict(self.sinkhorn_kwargs)  # copy
            reg_m = kwargs.pop("reg_m", self.reg)
            ot_loss = ot.unbalanced.sinkhorn_unbalanced2(
                src_probs, b, C, self.reg, reg_m, **kwargs
            )
        else:
            ot_loss = ot.sinkhorn2(src_probs, b, C, self.reg, **self.sinkhorn_kwargs)

        # sinkhorn2 returns cost or (cost, log) depending on `log`
        if isinstance(ot_loss, tuple):
            ot_loss = ot_loss[0]

        # ensure scalar tensor
        if not torch.is_tensor(ot_loss):
            ot_loss = torch.tensor(ot_loss, device=device, dtype=dtype)

        # --- supervised alignment loss -------------------------------------------------
        aligned_indices = torch.where(aligned != -1)[0]  # N_aligned

        if aligned_indices.numel() > 0:
            aligned_descriptors = descriptors[aligned_indices]
            corresp_word_embeddings = word_embeddings[aligned[aligned_indices]]
            distance_loss = torch.nn.functional.mse_loss(
                aligned_descriptors, corresp_word_embeddings
            )
        else:
            distance_loss = torch.tensor(0.0, device=device, dtype=dtype)

        return ot_loss + distance_loss


