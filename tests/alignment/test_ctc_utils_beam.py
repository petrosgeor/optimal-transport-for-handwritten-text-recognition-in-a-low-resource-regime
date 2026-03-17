"""Unit tests for CTC beam-search decoding."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("pyctcdecode")

from alignment.utils.ctc import beam_search_ctc_decode, greedy_ctc_decode


def _build_time_first_logits(
    argmax_indices: list[int],
    num_classes: int,
) -> torch.Tensor:
    """Create deterministic logits with a fixed argmax sequence.

    Args:
    - argmax_indices (list[int]): Class index selected at each time step.
    - num_classes (int): Total number of CTC classes including blank.

    Returns:
    - (torch.Tensor): Logits tensor shaped (T, B, C) with B=1.
    """
    time_steps = len(argmax_indices)
    logits = torch.full((time_steps, 1, num_classes), -20.0, dtype=torch.float32)
    for time_idx, cls_idx in enumerate(argmax_indices):
        logits[time_idx, 0, cls_idx] = 20.0
    return logits


def test_beam_matches_greedy_on_unambiguous_logits() -> None:
    """Beam decode should match greedy decode when logits are unambiguous."""
    i2c = {1: "a", 2: "b", 3: " "}
    logits = _build_time_first_logits([0, 1, 1, 0, 2, 2], num_classes=4)
    greedy = greedy_ctc_decode(logits, i2c, blank_id=0, time_first=True)
    beam = beam_search_ctc_decode(
        logits,
        i2c,
        beam_width=5,
        blank_id=0,
        time_first=True,
    )
    assert beam == greedy
