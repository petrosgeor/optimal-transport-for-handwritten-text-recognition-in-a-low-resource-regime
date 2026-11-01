"""Unit tests for CER accumulation and scoring semantics."""

from __future__ import annotations

import pytest

from htr_base.utils.metrics import CER


def test_cer_exact_value_for_single_pair() -> None:
    """CER should be 0.0 for exact match and 0.5 for one deletion in len-2."""
    metric = CER()
    metric.update("ab", "ab")
    assert metric.score() == pytest.approx(0.0)

    metric.reset()
    metric.update("a", "ab")
    assert metric.score() == pytest.approx(0.5)


def test_cer_accumulates_over_multiple_updates() -> None:
    """CER should equal total edit distance divided by total target length."""
    metric = CER()
    metric.update("ab", "ab")  # dist=0, len=2
    metric.update("a", "ab")  # dist=1, len=2
    expected = (0.0 + 1.0) / (2 + 2)
    assert metric.score() == pytest.approx(expected)


def test_cer_score_raises_if_no_updates() -> None:
    """CER.score should currently raise ZeroDivisionError when empty."""
    metric = CER()
    with pytest.raises(ZeroDivisionError):
        _ = metric.score()
