"""Unit tests for CTC encoding and greedy decoding helpers."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("pyctcdecode")

from alignment.ctc_utils import encode_for_ctc, greedy_ctc_decode


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


def test_encode_for_ctc_strips_boundaries() -> None:
    """Encode should strip boundaries before mapping chars to ids."""
    c2i = {"a": 1, "b": 2, " ": 3}
    targets, lengths = encode_for_ctc(["  ab  "], c2i)
    assert lengths.tolist() == [2]
    assert targets.tolist() == [1, 2]


def test_encode_for_ctc_rejects_empty_after_strip() -> None:
    """Encode should raise on labels that become empty after stripping."""
    c2i = {"a": 1, "b": 2, " ": 3}
    with pytest.raises(ValueError):
        encode_for_ctc(["   "], c2i)


def test_greedy_decode_removes_blanks_and_collapses_repeats() -> None:
    """Greedy decode should collapse repeats and ignore blank ids."""
    i2c = {1: "a", 2: "b", 3: " "}
    logits = _build_time_first_logits([0, 1, 1, 0, 2, 2], num_classes=4)
    decoded = greedy_ctc_decode(logits, i2c, blank_id=0, time_first=True)
    assert decoded == ["ab"]


def test_greedy_decode_strips_boundary_spaces_but_keeps_internal_space() -> None:
    """Greedy decode should trim outer spaces and preserve inner separators."""
    i2c = {1: "a", 2: "b", 3: " "}
    logits = _build_time_first_logits([3, 1, 3, 2, 3], num_classes=4)
    decoded = greedy_ctc_decode(logits, i2c, blank_id=0, time_first=True)
    assert decoded == ["a b"]
    assert not decoded[0].startswith(" ")
    assert not decoded[0].endswith(" ")


def test_greedy_decode_supports_batch_first_layout() -> None:
    """Greedy decode should return the same text for batch-first logits."""
    i2c = {1: "a", 2: "b", 3: " "}
    logits_tf = _build_time_first_logits([0, 1, 1, 0, 2, 2], num_classes=4)
    logits_bf = logits_tf.permute(1, 0, 2).contiguous()
    decoded = greedy_ctc_decode(logits_bf, i2c, blank_id=0, time_first=False)
    assert decoded == ["ab"]
