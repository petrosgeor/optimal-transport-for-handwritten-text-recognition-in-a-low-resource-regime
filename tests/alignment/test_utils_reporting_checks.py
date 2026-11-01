"""Regression tests for extracted reporting and numeric-check helpers.

These tests ensure the helper extraction out of ``alignment.trainer`` did not
change file formats, auto-path behavior, or finite-value safety checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

from alignment.utils.checks import _assert_finite, _assert_grad_finite
from alignment.utils.reporting import log_pseudo_labels, log_round_metrics


@dataclass
class _DummyDataset:
    """Minimal dataset-like container used by reporting tests."""

    aligned: torch.Tensor
    unique_words: list[str]
    transcriptions: list[str]


def test_log_pseudo_labels_writes_expected_rows(tmp_path) -> None:
    """Pseudo-label logging should preserve the exported row format.

    This protects the reporting refactor so moving the helper does not change
    which rows are written or how predicted and ground-truth words are paired.
    """
    dataset = _DummyDataset(
        aligned=torch.tensor([-1, 1, 0], dtype=torch.int64),
        unique_words=["zero", "one"],
        transcriptions=["gt0", "gt1", "gt2"],
    )
    out_dir = tmp_path / "pseudo"
    log_pseudo_labels(
        torch.tensor([0, 1, 2], dtype=torch.int64),
        dataset,  # type: ignore[arg-type]
        round_idx=3,
        out_dir=str(out_dir),
    )
    target_file = out_dir / "pseudo_labels_round_3.txt"
    assert target_file.is_file()
    assert target_file.read_text(encoding="utf-8").splitlines() == [
        "1\tone\tgt1",
        "2\tzero\tgt2",
    ]


def test_log_round_metrics_auto_mode_overwrites_then_appends(tmp_path, monkeypatch) -> None:
    """Auto-path metrics logging should preserve overwrite and append behavior.

    This catches regressions where the extracted helper would change the
    original round-1 reset behavior or the resolved auto output path.
    """
    monkeypatch.chdir(tmp_path)
    basefolder = "DATASETS/CVL/processed_words"

    log_round_metrics(
        round_idx=1,
        correct_pseudo=5,
        cer_test=0.1234,
        wer_test=10.5,
        out_file="auto",
        basefolder=basefolder,
        n_aligned=50,
        seed=3,
    )
    auto_file = tmp_path / "results" / "CVL" / "cvl_50samples.txt"
    assert auto_file.is_file()
    assert auto_file.read_text(encoding="utf-8").splitlines() == ["1\t5\t0.1234\t10.5000\t3"]

    log_round_metrics(
        round_idx=1,
        correct_pseudo=7,
        cer_test=0.2000,
        wer_test=20.0,
        out_file="auto",
        basefolder=basefolder,
        n_aligned=50,
        seed=3,
    )
    # Round 1 should replace the previous contents rather than append to them.
    assert auto_file.read_text(encoding="utf-8").splitlines() == ["1\t7\t0.2000\t20.0000\t3"]

    log_round_metrics(
        round_idx=2,
        correct_pseudo=9,
        cer_test=0.3000,
        wer_test=30.0,
        out_file="auto",
        basefolder=basefolder,
        n_aligned=50,
        seed=3,
    )
    # Later rounds append so we keep a full refinement history in one file.
    assert auto_file.read_text(encoding="utf-8").splitlines() == [
        "1\t7\t0.2000\t20.0000\t3",
        "2\t9\t0.3000\t30.0000\t3",
    ]


def test_assert_finite_passes_and_raises() -> None:
    """Finite checks should reject NaN and Inf values consistently.

    This protects the shared numeric-check helper so callers in trainer and
    eval continue getting the same failure condition after deduplication.
    """
    _assert_finite(torch.tensor([0.0, 1.0]), "tensor_ok")
    with pytest.raises(AssertionError, match="Non-finite values in tensor_bad"):
        _assert_finite(torch.tensor([0.0, float("nan")]), "tensor_bad")


def test_assert_grad_finite_supports_aggregate_and_detailed_modes() -> None:
    """Gradient checks should support aggregate and named-parameter failures.

    This preserves the shared helper contract across trainer and pretraining,
    where one caller wants a summary error and another wants parameter detail.
    """
    model = nn.Linear(2, 1)
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    y = model(x).sum()
    y.backward()

    _assert_grad_finite(model, "linear")
    _assert_grad_finite(model, "linear", include_param_names=True)

    model.weight.grad[0, 0] = float("nan")
    with pytest.raises(AssertionError, match="Gradient explosion in linear"):
        _assert_grad_finite(model, "linear")
    with pytest.raises(AssertionError, match="grad NaN/Inf in weight"):
        _assert_grad_finite(model, "linear", include_param_names=True)
