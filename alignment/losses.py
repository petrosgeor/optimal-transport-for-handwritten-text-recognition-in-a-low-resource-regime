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
        logits (torch.Tensor): Unnormalized class scores with shape (T, B, C).
        targets (torch.IntTensor): Flattened encoded targets with shape (sum(L),).
        inp_lens (torch.IntTensor): Input sequence lengths per sample with shape (B,).
        tgt_lens (torch.IntTensor): Target lengths per sample with shape (B,).

    Returns:
        torch.Tensor: Scalar loss tensor on the same device as `logits`.
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


def compute_sinkhorn_coupling(
    a: torch.Tensor,
    b: torch.Tensor,
    cost: torch.Tensor,
    beta: float,
    *,
    outer_iters: int,
    inner_iters: int,
    tol: float = 1e-6,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a balanced OT coupling with iterative Sinkhorn-style normalization.

    This implements the algorithm requested in the planning notes:
    build the Gibbs kernel ``A = exp(-C / beta)``, repeatedly reweight the
    current plan, and normalize rows/columns with scaling vectors until the
    marginals are matched.

    Args:
        a (torch.Tensor): Source marginal vector with shape ``(n,)``.
        b (torch.Tensor): Target marginal vector with shape ``(m,)``.
        cost (torch.Tensor): Ground-cost matrix with shape ``(n, m)``.
        beta (float): Entropic regularization parameter. Must be positive.
        outer_iters (int): Number of outer reweighting iterations.
        inner_iters (int): Number of inner marginal-normalization updates.
        tol (float): Early-stop threshold on max marginal residual.
        eps (float): Small positive constant used in safe division.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - ``T``: Transport coupling with shape ``(n, m)``.
            - ``D_w``: Wasserstein cost ``(cost * T).sum()``.

    Raises:
        ValueError: If shapes, dtypes, iteration counts, regularization, or
            finiteness checks fail. Also raised when ``tol > 0`` and the final
            coupling does not satisfy the requested marginal tolerance.
    """
    if cost.ndim != 2:
        raise ValueError(f"cost must have shape (n, m); got cost.shape={tuple(cost.shape)}")
    n, m = cost.shape
    if a.shape != (n,):
        raise ValueError(f"a must have shape ({n},); got a.shape={tuple(a.shape)}")
    if b.shape != (m,):
        raise ValueError(f"b must have shape ({m},); got b.shape={tuple(b.shape)}")
    if not (cost.dtype.is_floating_point and a.dtype.is_floating_point and b.dtype.is_floating_point):
        raise ValueError("a, b, and cost must all be floating-point tensors")
    if beta <= 0:
        raise ValueError(f"beta must be > 0; got beta={beta}")
    if outer_iters < 1:
        raise ValueError(f"outer_iters must be >= 1; got outer_iters={outer_iters}")
    if inner_iters < 1:
        raise ValueError(f"inner_iters must be >= 1; got inner_iters={inner_iters}")
    if tol < 0:
        raise ValueError(f"tol must be >= 0; got tol={tol}")
    if eps <= 0:
        raise ValueError(f"eps must be > 0; got eps={eps}")

    a = a.to(device=cost.device, dtype=cost.dtype)
    b = b.to(device=cost.device, dtype=cost.dtype)

    if not torch.isfinite(cost).all():
        raise ValueError("cost contains NaN/Inf")
    if not torch.isfinite(a).all():
        raise ValueError("a contains NaN/Inf")
    if not torch.isfinite(b).all():
        raise ValueError("b contains NaN/Inf")
    if (a < 0).any():
        raise ValueError("a contains negative entries")
    if (b < 0).any():
        raise ValueError("b contains negative entries")

    a_sum = a.sum()
    b_sum = b.sum()
    if a_sum.item() <= 0:
        raise ValueError("a must sum to > 0")
    if b_sum.item() <= 0:
        raise ValueError("b must sum to > 0")
    a = a / a_sum
    b = b / b_sum

    A = torch.exp(-cost / beta)
    T = torch.ones((n, m), device=cost.device, dtype=cost.dtype)
    sigma = torch.full((m,), 1.0 / float(m), device=cost.device, dtype=cost.dtype)

    for _ in range(outer_iters):
        Q = A * T
        delta = torch.full((n,), 1.0 / float(n), device=cost.device, dtype=cost.dtype)
        for _ in range(inner_iters):
            delta = a / (Q @ sigma).clamp_min(eps)
            sigma = b / (Q.transpose(0, 1) @ delta).clamp_min(eps)
        T = delta[:, None] * Q * sigma[None, :]

        if tol > 0:
            row_err = (T.sum(dim=1) - a).abs().max()
            col_err = (T.sum(dim=0) - b).abs().max()
            if torch.maximum(row_err, col_err) <= tol:
                break

    D_w = (cost * T).sum()
    row_err = (T.sum(dim=1) - a).abs().max()
    col_err = (T.sum(dim=0) - b).abs().max()

    if not torch.isfinite(T).all():
        raise ValueError("computed coupling contains NaN/Inf")
    if not torch.isfinite(D_w).all():
        raise ValueError("computed Wasserstein cost is NaN/Inf")
    if tol > 0 and torch.maximum(row_err, col_err).item() > tol:
        raise ValueError(
            "Sinkhorn solver did not converge to requested tolerance; "
            f"max residual={torch.maximum(row_err, col_err).item():.4e}, tol={tol:.4e}"
        )

    return T, D_w


def _validate_gw_shapes_and_params(
    a: torch.Tensor,
    b: torch.Tensor,
    x_feats: torch.Tensor,
    y_feats: torch.Tensor,
    beta: float,
    gw_iters: int,
    sinkhorn_outer_iters: int,
    sinkhorn_inner_iters: int,
    tol: float,
    eps: float,
) -> None:
    """Validate GW tensor shapes and scalar hyperparameters.

    Args:
        a (torch.Tensor): Source probability vector expected with shape ``(n,)``.
        b (torch.Tensor): Target probability vector expected with shape ``(m,)``.
        x_feats (torch.Tensor): Source features expected with shape ``(n, d_x)``.
        y_feats (torch.Tensor): Target features expected with shape ``(m, d_y)``.
        beta (float): Entropic regularization for the inner OT solver.
        gw_iters (int): Number of GW fixed-point iterations.
        sinkhorn_outer_iters (int): Number of outer Sinkhorn iterations.
        sinkhorn_inner_iters (int): Number of inner Sinkhorn iterations.
        tol (float): Convergence tolerance for the GW fixed-point loop.
        eps (float): Small positive constant for numerical safeguards.

    Raises:
        ValueError: If any tensor shape, dtype, or scalar parameter is invalid.
    """
    if x_feats.ndim != 2:
        raise ValueError(
            f"x_feats must have shape (n, d_x); got x_feats.shape={tuple(x_feats.shape)}"
        )
    if y_feats.ndim != 2:
        raise ValueError(
            f"y_feats must have shape (m, d_y); got y_feats.shape={tuple(y_feats.shape)}"
        )

    n, _ = x_feats.shape
    m, _ = y_feats.shape

    if a.shape != (n,):
        raise ValueError(f"a must have shape ({n},); got a.shape={tuple(a.shape)}")
    if b.shape != (m,):
        raise ValueError(f"b must have shape ({m},); got b.shape={tuple(b.shape)}")
    if not (x_feats.dtype.is_floating_point and y_feats.dtype.is_floating_point):
        raise ValueError("x_feats and y_feats must be floating-point tensors")
    if not (a.dtype.is_floating_point and b.dtype.is_floating_point):
        raise ValueError("a and b must be floating-point tensors")
    if beta <= 0:
        raise ValueError(f"beta must be > 0; got beta={beta}")
    if gw_iters < 1:
        raise ValueError(f"gw_iters must be >= 1; got gw_iters={gw_iters}")
    if sinkhorn_outer_iters < 1:
        raise ValueError(
            "sinkhorn_outer_iters must be >= 1; "
            f"got sinkhorn_outer_iters={sinkhorn_outer_iters}"
        )
    if sinkhorn_inner_iters < 1:
        raise ValueError(
            "sinkhorn_inner_iters must be >= 1; "
            f"got sinkhorn_inner_iters={sinkhorn_inner_iters}"
        )
    if tol < 0:
        raise ValueError(f"tol must be >= 0; got tol={tol}")
    if eps <= 0:
        raise ValueError(f"eps must be > 0; got eps={eps}")


def _prepare_gw_inputs(
    a: torch.Tensor,
    b: torch.Tensor,
    x_feats: torch.Tensor,
    y_feats: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move GW inputs onto a common device and dtype.

    Args:
        a (torch.Tensor): Source marginal vector.
        b (torch.Tensor): Target marginal vector.
        x_feats (torch.Tensor): Source feature matrix.
        y_feats (torch.Tensor): Target feature matrix.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            Prepared tensors ``(a, b, x_feats, y_feats)`` on a shared device and dtype.
    """
    x_feats = x_feats.to(dtype=x_feats.dtype)
    y_feats = y_feats.to(device=x_feats.device, dtype=x_feats.dtype)
    a = a.to(device=x_feats.device, dtype=x_feats.dtype)
    b = b.to(device=x_feats.device, dtype=x_feats.dtype)

    return a, b, x_feats, y_feats


def _validate_and_normalize_gw_marginals(
    a: torch.Tensor,
    b: torch.Tensor,
    x_feats: torch.Tensor,
    y_feats: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate GW tensor values and normalize the source/target marginals.

    Args:
        a (torch.Tensor): Source marginal vector on the GW working device/dtype.
        b (torch.Tensor): Target marginal vector on the GW working device/dtype.
        x_feats (torch.Tensor): Source feature matrix on the GW working device/dtype.
        y_feats (torch.Tensor): Target feature matrix on the GW working device/dtype.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Normalized marginals ``(a, b)``.

    Raises:
        ValueError: If tensors contain non-finite values, negative mass, or
            non-positive total mass.
    """
    for tensor, name in ((x_feats, "x_feats"), (y_feats, "y_feats"), (a, "a"), (b, "b")):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN/Inf")
    if (a < 0).any():
        raise ValueError("a contains negative entries")
    if (b < 0).any():
        raise ValueError("b contains negative entries")

    a_sum = a.sum()
    b_sum = b.sum()
    if a_sum.item() <= 0:
        raise ValueError("a must sum to > 0")
    if b_sum.item() <= 0:
        raise ValueError("b must sum to > 0")

    return a / a_sum, b / b_sum


def _compute_gw_distance_matrices(
    x_feats: torch.Tensor,
    y_feats: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute intra-domain distance matrices for the GW objective.

    Args:
        x_feats (torch.Tensor): Source feature matrix with shape ``(n, d_x)``.
        y_feats (torch.Tensor): Target feature matrix with shape ``(m, d_y)``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            Pairwise Euclidean distance matrices ``(D_x, D_y)``.

    Raises:
        ValueError: If either distance matrix contains non-finite values.
    """
    D_x = torch.cdist(x_feats, x_feats, p=2)
    D_y = torch.cdist(y_feats, y_feats, p=2)
    if not torch.isfinite(D_x).all():
        raise ValueError("D_x contains NaN/Inf")
    if not torch.isfinite(D_y).all():
        raise ValueError("D_y contains NaN/Inf")

    return D_x, D_y


def compute_gromov_wasserstein_coupling(
    a: torch.Tensor,
    b: torch.Tensor,
    x_feats: torch.Tensor,
    y_feats: torch.Tensor,
    beta: float,
    *,
    gw_iters: int,
    sinkhorn_outer_iters: int,
    sinkhorn_inner_iters: int,
    tol: float = 1e-6,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a balanced Gromov-Wasserstein coupling from source/target features.

    The routine follows the linearization scheme from Algorithm 2:
    construct intra-domain relation matrices, build a pseudo-cost matrix from
    the current plan, and repeatedly solve the induced OT subproblem.

    Args:
        a (torch.Tensor): Source probability vector with shape ``(n,)``.
        b (torch.Tensor): Target probability vector with shape ``(m,)``.
        x_feats (torch.Tensor): Source feature matrix with shape ``(n, d_x)``.
        y_feats (torch.Tensor): Target feature matrix with shape ``(m, d_y)``.
        beta (float): Entropic regularization for the inner OT solver.
        gw_iters (int): Maximum number of outer GW fixed-point iterations.
        sinkhorn_outer_iters (int): Outer iterations passed to
            ``compute_sinkhorn_coupling``.
        sinkhorn_inner_iters (int): Inner normalization iterations passed to
            ``compute_sinkhorn_coupling``.
        tol (float): Convergence tolerance on max plan change per GW step.
        eps (float): Small positive constant for numerical safeguards.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            -``T``: Final coupling matrix with shape ``(n, m)``.
            -``D_gw``: Final GW objective value ``(L_final * T).sum()``.

    Raises:
        ValueError: If inputs are invalid, contain non-finite values, or if
            the outer GW fixed-point iterations do not converge when ``tol > 0``.
    """
    _validate_gw_shapes_and_params(
        a=a,
        b=b,
        x_feats=x_feats,
        y_feats=y_feats,
        beta=beta,
        gw_iters=gw_iters,
        sinkhorn_outer_iters=sinkhorn_outer_iters,
        sinkhorn_inner_iters=sinkhorn_inner_iters,
        tol=tol,
        eps=eps,
    )
    a, b, x_feats, y_feats = _prepare_gw_inputs(a=a, b=b, x_feats=x_feats, y_feats=y_feats)
    a, b = _validate_and_normalize_gw_marginals(a=a, b=b, x_feats=x_feats, y_feats=y_feats)

    # Build intra-domain distance matrices D_x in R^(n x n) and D_y in R^(m x m).
    D_x, D_y = _compute_gw_distance_matrices(x_feats=x_feats, y_feats=y_feats)

    # Build the separable GW scaffold M in R^(n x m) and the initial coupling T in R^(n x m).
    mx = D_x.pow(2) @ a
    my = D_y.pow(2) @ b
    M = mx[:, None] + my[None, :]
    T = a[:, None] * b[None, :]

    converged = False
    # Alternate between linearizing the GW objective and solving the induced OT problem.
    for _ in range(gw_iters):
        # L is the current GW pseudo-cost matrix in R^(n x m).
        L = M - 2.0 * (D_x @ T @ D_y.transpose(0, 1))
        if not torch.isfinite(L).all():
            raise ValueError("GW pseudo-cost contains NaN/Inf")
        T_next, _ = compute_sinkhorn_coupling(
            a,
            b,
            L,
            beta,
            outer_iters=sinkhorn_outer_iters,
            inner_iters=sinkhorn_inner_iters,
            tol=tol,
            eps=eps,
        )
        delta = (T_next - T).abs().max()
        T = T_next
        if tol > 0 and delta.item() <= tol:
            converged = True
            break
        if tol == 0:
            converged = True

    # Re-evaluate the final pseudo-cost and objective with the converged coupling.
    L_final = M - 2.0 * (D_x @ T @ D_y.transpose(0, 1))
    D_gw = (L_final * T).sum()

    if not torch.isfinite(T).all():
        raise ValueError("GW coupling contains NaN/Inf")
    if not torch.isfinite(D_gw).all():
        raise ValueError("GW objective contains NaN/Inf")
    if tol > 0 and not converged:
        raise ValueError(
            "GW solver did not converge to requested tolerance; "
            f"tol={tol:.4e} after gw_iters={gw_iters}"
        )
    return T, D_gw


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
            reg (float): Entropic regularization for Sinkhorn.
            unbalanced (bool): Toggle unbalanced OT formulation.
            reg_m (float): Unbalanced mass regularization parameter.
            supervised_weight (float): Weight for supervised MSE anchor term.
            **sinkhorn_kwargs: Extra options forwarded to `SamplesLoss`.
        """
        super().__init__()
        self.reg = reg
        self.unbalanced = unbalanced
        self.reg_m = reg_m
        self.supervised_weight = supervised_weight
        self.sinkhorn_kwargs = sinkhorn_kwargs

        if self.unbalanced:
            self.unbalanced_ot_loss = SamplesLoss(
                loss="sinkhorn",
                p=2,
                blur=self.reg,
                reach=self.reg_m,
                debias=True,
                **self.sinkhorn_kwargs,
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
            descriptors (torch.Tensor): Source descriptors expected with shape (N, D).
            word_embeddings (torch.Tensor): Target embeddings expected with shape (M, D).
            aligned (torch.Tensor): Alignment indices expected with shape (N,).
            tgt_probs (torch.Tensor): Target marginal probabilities expected with shape (M,).

        Returns:
            tuple[int, int, int]: The tuple (N, M, D) from validated tensor shapes.

        Raises:
            ValueError: If shapes, dtypes, or finiteness checks fail.
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
            descriptors (torch.Tensor): Source descriptors with shape (N, D).
            word_embeddings (torch.Tensor): Target embeddings with shape (M, D).
            aligned (torch.Tensor): Alignment ids with shape (N,). Use -1 for unknown.
            tgt_probs (torch.Tensor): Target distribution weights with shape (M,).

        Returns:
            torch.Tensor: Scalar loss equal to OT term plus weighted supervised MSE term.

        Raises:
            ValueError: If inputs are invalid, if balanced-mode `tgt_probs` sums
                to a non-positive value, or if OT loss becomes non-finite.
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
from .utils.ctc import _unflatten_targets


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

    def __init__(self, tau: float = 0.07, T_txt: float = 1.0, eps: float = 1e-8):
        """Initialize contrastive-loss temperatures.

        Args:
            tau (float): Temperature for image-space pair similarities.
            T_txt (float): Temperature for text-space pair similarities.
            eps (float): Numerical-stability constant for division and log.
        """
        super().__init__()
        self.tau = tau
        self.T_txt = T_txt
        self.eps = eps

    def forward(
        self, feats: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Compute soft contrastive loss for one minibatch.

        Args:
            feats (torch.Tensor): Feature matrix with shape (B, D).
            targets (torch.Tensor): Flattened encoded CTC targets with shape (sum(L),).
            lengths (torch.Tensor): Target lengths with shape (B,).

        Returns:
            torch.Tensor: Scalar contrastive loss. Returns 0 when B < 2.
        """
        B = len(lengths)
        if B < 2:
            return torch.tensor(0.0, device=feats.device, dtype=feats.dtype)

        # ---- 1. pairwise Euclidean distances in descriptor space ----
        dists = torch.cdist(feats, feats, p=2)  # (B, B)
        sim_img = torch.exp(-dists / self.tau)  # convert to similarity

        # ---- 2. edit-distance matrix (vectorized) --------------------
        with torch.no_grad():
            # Convert flattened targets to list of lists
            unflattened_targets = _unflatten_targets(targets, lengths)
            # Use rapidfuzz for fast, parallelized pairwise edit distance.
            edit_np = process.cdist(
                unflattened_targets,
                unflattened_targets,
                scorer=distance.Levenshtein.distance,
                workers=-1,
            ).astype(np.float32)
            edit = torch.from_numpy(edit_np).to(device=feats.device)
        sim_txt = torch.exp(-edit / self.T_txt)  # soft positives

        # ---- 3. InfoNCE objective ------------------------------------
        # exclude diagonal (self‑similarity) from both sums
        eye = torch.eye(B, device=feats.device)
        sim_img = sim_img * (1 - eye)
        sim_txt = sim_txt * (1 - eye)

        numerator = (sim_txt * sim_img).sum(dim=1)  # (B,)
        denominator = sim_img.sum(dim=1) + self.eps  # (B,)
        loss = -torch.log((numerator + self.eps) / denominator).mean()
        return loss
