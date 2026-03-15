"""Numerical sanity-check helpers used across alignment modules."""

from __future__ import annotations

import torch
import torch.nn as nn


def _assert_finite(
    t: torch.Tensor,
    where: str,
    *,
    message_prefix: str = "Non-finite values in",
) -> None:
    """Assert that a tensor contains no NaN or Inf values.

    Args:
        t (torch.Tensor): Tensor to validate with any shape.
        where (str): Human-readable tensor location for error messages.
        message_prefix (str): Prefix used in the assertion message.
    """
    assert torch.isfinite(t).all(), f"{message_prefix} {where}"


def _assert_grad_finite(
    model: nn.Module,
    name: str,
    *,
    include_param_names: bool = False,
) -> None:
    """Ensure all parameter gradients in a model are finite.

    Args:
        model (nn.Module): Model whose gradients are checked.
        name (str): Identifier used in non-detailed error messages.
        include_param_names (bool): If True, raise with the offending parameter
            name. If False, use a single aggregate assertion message.
    """
    if include_param_names:
        for param_name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"grad NaN/Inf in {param_name}"
        return

    assert all(
        param.grad is None or torch.isfinite(param.grad).all() for param in model.parameters()
    ), f"Gradient explosion in {name}"
