"""Unit tests for the standalone Sinkhorn coupling solver."""

from __future__ import annotations

import numpy as np
import ot
import pytest
import torch

from alignment.losses import compute_sinkhorn_coupling


def _make_toy_problem() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a small balanced OT problem with explicit marginals.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            Source marginal ``a``, target marginal ``b``, and cost matrix ``C``.
    """
    a = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    b = torch.tensor([0.4, 0.35, 0.25], dtype=torch.float64)
    cost = torch.tensor(
        [
            [0.1, 0.7, 1.2],
            [0.6, 0.2, 0.8],
            [1.0, 0.5, 0.3],
        ],
        dtype=torch.float64,
    )
    return a, b, cost


def _make_param_problem(
    n_rows: int,
    n_cols: int,
    *,
    dtype: torch.dtype,
    uniform: bool,
    cost_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a deterministic balanced OT problem for parity tests.

    Args:
        n_rows (int): Number of source atoms.
        n_cols (int): Number of target atoms.
        dtype (torch.dtype): Floating dtype for returned tensors.
        uniform (bool): Whether to use uniform marginals.
        cost_scale (float): Positive scalar applied to the cost matrix.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            Source marginal ``a``, target marginal ``b``, and cost matrix ``C``.
    """
    row_idx = torch.arange(n_rows, dtype=dtype)
    col_idx = torch.arange(n_cols, dtype=dtype)
    cost = (
        (row_idx[:, None] - col_idx[None, :]).abs()
        + 0.15 * row_idx[:, None]
        + 0.25 * col_idx[None, :]
        + 0.05 * (row_idx[:, None] + 1) * (col_idx[None, :] + 1)
    ) * cost_scale

    if uniform:
        a = torch.full((n_rows,), 1.0 / float(n_rows), dtype=dtype)
        b = torch.full((n_cols,), 1.0 / float(n_cols), dtype=dtype)
    else:
        a = torch.arange(1, n_rows + 1, dtype=dtype)
        b = torch.arange(n_cols, 0, -1, dtype=dtype)
        a = a / a.sum()
        b = b / b.sum()
    return a, b, cost


def _run_pot_reference(
    a: torch.Tensor,
    b: torch.Tensor,
    cost: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute POT's balanced Sinkhorn reference solution for one problem.

    Args:
        a (torch.Tensor): Source marginal with shape ``(n,)``.
        b (torch.Tensor): Target marginal with shape ``(m,)``.
        cost (torch.Tensor): Cost matrix with shape ``(n, m)``.
        beta (float): Entropic regularization strength.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            POT transport plan and its Frobenius transport cost.
    """
    pot_plan = ot.sinkhorn(
        a.detach().cpu().numpy(),
        b.detach().cpu().numpy(),
        cost.detach().cpu().numpy(),
        reg=beta,
        numItermax=5000,
        stopThr=1e-12,
    )
    pot_plan_t = torch.from_numpy(np.asarray(pot_plan)).to(dtype=cost.dtype)
    pot_cost = (cost * pot_plan_t).sum()
    return pot_plan_t, pot_cost


def _parity_tolerances(dtype: torch.dtype) -> tuple[float, float, float, float]:
    """Return marginal, cost, mean-plan, and max-plan tolerances for a dtype.

    Args:
        dtype (torch.dtype): Floating dtype used in the parity problem.

    Returns:
        tuple[float, float, float, float]:
            Tolerances for marginal residual, cost difference, mean absolute
            plan difference, and max absolute plan difference.
    """
    if dtype == torch.float64:
        return 1e-6, 1e-6, 1e-4, 1e-3
    return 1e-4, 1e-4, 1e-3, 1e-2


STABLE_PARITY_CASES = [
    ((3, 3), True, torch.float64, 1.0, 400, 1e-15),
    ((3, 3), True, torch.float32, 1.0, 600, 1e-12),
    ((4, 3), False, torch.float64, 1.0, 400, 1e-15),
    ((4, 3), False, torch.float32, 1.0, 600, 1e-12),
    ((3, 3), False, torch.float64, 0.1, 3200, 1e-15),
    ((3, 3), False, torch.float32, 0.1, 3200, 1e-12),
    ((4, 3), True, torch.float64, 0.1, 3200, 1e-15),
    ((4, 3), True, torch.float32, 0.1, 3200, 1e-12),
]


def test_compute_sinkhorn_coupling_shape_nonnegative_and_scalar_cost() -> None:
    """Solver should return a nonnegative coupling with expected shape and scalar cost."""
    a, b, cost = _make_toy_problem()
    plan, wasserstein = compute_sinkhorn_coupling(
        a,
        b,
        cost,
        beta=0.1,
        outer_iters=15,
        inner_iters=30,
        tol=0.0,
    )

    assert plan.shape == cost.shape
    assert (plan >= 0).all()
    assert wasserstein.ndim == 0


def test_compute_sinkhorn_coupling_matches_target_marginals() -> None:
    """Solver should recover marginals within numerical tolerance."""
    a, b, cost = _make_toy_problem()
    plan, _ = compute_sinkhorn_coupling(
        a,
        b,
        cost,
        beta=0.1,
        outer_iters=20,
        inner_iters=50,
        tol=1e-10,
        eps=1e-15,
    )

    assert torch.allclose(plan.sum(dim=1), a, atol=5e-4, rtol=5e-4)
    assert torch.allclose(plan.sum(dim=0), b, atol=5e-4, rtol=5e-4)


def test_compute_sinkhorn_coupling_cost_matches_frobenius_inner_product() -> None:
    """Reported Wasserstein cost should equal ``(C * T).sum()``."""
    a, b, cost = _make_toy_problem()
    plan, wasserstein = compute_sinkhorn_coupling(
        a,
        b,
        cost,
        beta=0.2,
        outer_iters=10,
        inner_iters=20,
        tol=0.0,
    )

    expected_cost = (cost * plan).sum()
    assert torch.allclose(wasserstein, expected_cost, atol=1e-12, rtol=1e-12)


def test_compute_sinkhorn_coupling_uniform_marginals() -> None:
    """Uniform marginals should stay approximately uniform after normalization."""
    cost = torch.tensor(
        [
            [0.0, 0.2, 0.4, 0.6],
            [0.3, 0.1, 0.2, 0.5],
        ],
        dtype=torch.float64,
    )
    a = torch.full((2,), 0.5, dtype=torch.float64)
    b = torch.full((4,), 0.25, dtype=torch.float64)

    plan, _ = compute_sinkhorn_coupling(
        a,
        b,
        cost,
        beta=0.15,
        outer_iters=15,
        inner_iters=40,
        tol=0.0,
    )

    assert torch.allclose(plan.sum(dim=1), a, atol=5e-4, rtol=5e-4)
    assert torch.allclose(plan.sum(dim=0), b, atol=5e-4, rtol=5e-4)


def test_compute_sinkhorn_coupling_close_to_pot_on_small_problem() -> None:
    """Single outer loop should closely match classic Sinkhorn from POT."""
    a, b, cost = _make_toy_problem()
    beta = 0.1

    plan, wasserstein = compute_sinkhorn_coupling(
        a,
        b,
        cost,
        beta=beta,
        outer_iters=1,
        inner_iters=400,
        tol=0.0,
        eps=1e-15,
    )

    pot_plan = ot.sinkhorn(
        a.detach().cpu().numpy(),
        b.detach().cpu().numpy(),
        cost.detach().cpu().numpy(),
        reg=beta,
        numItermax=5000,
        stopThr=1e-12,
    )
    pot_plan_t = torch.from_numpy(np.asarray(pot_plan)).to(dtype=cost.dtype)
    pot_cost = (cost * pot_plan_t).sum()

    assert torch.allclose(plan, pot_plan_t, atol=1e-3, rtol=1e-3)
    assert torch.allclose(wasserstein, pot_cost, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("shape", "uniform", "dtype", "beta", "inner_iters", "eps"),
    STABLE_PARITY_CASES,
)
def test_compute_sinkhorn_coupling_matches_pot_on_stable_reference_cases(
    shape: tuple[int, int],
    uniform: bool,
    dtype: torch.dtype,
    beta: float,
    inner_iters: int,
    eps: float,
) -> None:
    """Solver should stay close to POT on stable balanced reference problems."""
    n_rows, n_cols = shape
    a, b, cost = _make_param_problem(n_rows, n_cols, dtype=dtype, uniform=uniform)
    marginal_tol, cost_tol, plan_l1_tol, plan_linf_tol = _parity_tolerances(dtype)

    plan, wasserstein = compute_sinkhorn_coupling(
        a,
        b,
        cost,
        beta=beta,
        outer_iters=1,
        inner_iters=inner_iters,
        tol=0.0,
        eps=eps,
    )
    pot_plan, pot_cost = _run_pot_reference(a, b, cost, beta)

    ours_row_err = (plan.sum(dim=1) - a).abs().max().item()
    ours_col_err = (plan.sum(dim=0) - b).abs().max().item()
    pot_row_err = (pot_plan.sum(dim=1) - a).abs().max().item()
    pot_col_err = (pot_plan.sum(dim=0) - b).abs().max().item()
    cost_diff = (wasserstein - pot_cost).abs().item()
    plan_diff = (plan - pot_plan).abs()
    plan_l1 = plan_diff.mean().item()
    plan_linf = plan_diff.max().item()

    assert ours_row_err < marginal_tol
    assert ours_col_err < marginal_tol
    assert pot_row_err < marginal_tol
    assert pot_col_err < marginal_tol
    assert cost_diff < cost_tol
    assert plan_l1 < plan_l1_tol
    assert plan_linf < plan_linf_tol


@pytest.mark.parametrize(
    (
        "shape",
        "uniform",
        "dtype",
        "beta",
        "inner_iters",
        "eps",
        "cost_scale",
        "max_row_err",
        "max_col_err",
        "max_cost_diff",
    ),
    [
        ((6, 6), True, torch.float64, 0.1, 3200, 1e-15, 1.0, 1e-4, 1e-6, 5e-4),
        ((6, 6), True, torch.float32, 0.1, 3200, 1e-12, 1.0, 1e-4, 1e-6, 2e-4),
        ((6, 6), False, torch.float32, 0.1, 3200, 1e-12, 1.0, 3e-1, 3e-1, 2.0),
        ((3, 3), True, torch.float64, 0.01, 400, 1e-15, 1.0, 4e-1, 4e-1, 2.0),
        ((3, 3), True, torch.float32, 0.01, 600, 1e-12, 1.0, 4e-1, 4e-1, 2.0),
        ((5, 4), False, torch.float64, 1e-3, 800, 1e-15, 10.0, 1.0, 1.0, 10.0),
        ((5, 4), False, torch.float32, 1e-3, 800, 1e-12, 10.0, 1.0, 1.0, 10.0),
    ],
)
def test_compute_sinkhorn_coupling_handles_harder_cost_over_beta_regimes(
    shape: tuple[int, int],
    uniform: bool,
    dtype: torch.dtype,
    beta: float,
    inner_iters: int,
    eps: float,
    cost_scale: float,
    max_row_err: float,
    max_col_err: float,
    max_cost_diff: float,
) -> None:
    """Harder regimes should still compare to POT without catastrophic failure."""
    n_rows, n_cols = shape
    a, b, cost = _make_param_problem(
        n_rows,
        n_cols,
        dtype=dtype,
        uniform=uniform,
        cost_scale=cost_scale,
    )

    plan, wasserstein = compute_sinkhorn_coupling(
        a,
        b,
        cost,
        beta=beta,
        outer_iters=1,
        inner_iters=inner_iters,
        tol=0.0,
        eps=eps,
    )
    pot_plan, pot_cost = _run_pot_reference(a, b, cost, beta)

    row_err = (plan.sum(dim=1) - a).abs().max().item()
    col_err = (plan.sum(dim=0) - b).abs().max().item()
    cost_diff = (wasserstein - pot_cost).abs().item()

    assert torch.isfinite(plan).all()
    assert torch.isfinite(wasserstein).all()
    assert torch.isfinite(pot_plan).all()
    assert torch.isfinite(pot_cost).all()
    assert row_err < max_row_err
    assert col_err < max_col_err
    assert cost_diff < max_cost_diff


def test_compute_sinkhorn_coupling_rejects_invalid_inputs() -> None:
    """Solver should fail fast on invalid shapes and non-finite or invalid masses."""
    a, b, cost = _make_toy_problem()

    with pytest.raises(ValueError, match="beta must be > 0"):
        compute_sinkhorn_coupling(a, b, cost, beta=0.0, outer_iters=3, inner_iters=2)
    with pytest.raises(ValueError, match="outer_iters must be >= 1"):
        compute_sinkhorn_coupling(a, b, cost, beta=0.1, outer_iters=0, inner_iters=2)
    with pytest.raises(ValueError, match="inner_iters must be >= 1"):
        compute_sinkhorn_coupling(a, b, cost, beta=0.1, outer_iters=2, inner_iters=0)
    with pytest.raises(ValueError, match="a must have shape"):
        compute_sinkhorn_coupling(a[:2], b, cost, beta=0.1, outer_iters=2, inner_iters=2)
    with pytest.raises(ValueError, match="a contains negative entries"):
        bad_a = a.clone()
        bad_a[0] = -0.1
        compute_sinkhorn_coupling(bad_a, b, cost, beta=0.1, outer_iters=2, inner_iters=2)
    with pytest.raises(ValueError, match="b must sum to > 0"):
        compute_sinkhorn_coupling(a, torch.zeros_like(b), cost, beta=0.1, outer_iters=2, inner_iters=2)
    with pytest.raises(ValueError, match="cost contains NaN/Inf"):
        bad_cost = cost.clone()
        bad_cost[0, 0] = float("nan")
        compute_sinkhorn_coupling(a, b, bad_cost, beta=0.1, outer_iters=2, inner_iters=2)
    with pytest.raises(ValueError, match="did not converge"):
        compute_sinkhorn_coupling(
            a,
            b,
            cost,
            beta=0.1,
            outer_iters=1,
            inner_iters=1,
            tol=1e-15,
        )
