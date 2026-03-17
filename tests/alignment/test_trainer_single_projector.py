"""Smoke tests for trainer control flow after removing projector ensembles.

These tests protect the simplified trainer contract: refinement should accept
exactly one projector in both backbone-only and pseudo-labeling modes without
silently reintroducing list-based projector bookkeeping.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from alignment import trainer


class _DummyDataset:
    """Minimal dataset object for alternating_refinement tests."""

    def __init__(self, aligned: list[int]) -> None:
        """Initialize dataset metadata and alignment vector.

        Args:
            aligned (list[int]): Initial alignment ids.
        """
        self.aligned = torch.tensor(aligned, dtype=torch.int32)
        self.unique_words = ["a", "b"]
        self.transcriptions = ["a", "b"][: len(aligned)]
        self.basefolder = "dummy/basefolder"
        self.fixed_size = (64, 256)
        self.character_classes = ["a", "b"]
        self.config = SimpleNamespace(n_aligned=int((self.aligned != -1).sum().item()))
        self.n_aligned = int((self.aligned != -1).sum().item())

    def __len__(self) -> int:
        """Return dataset size."""
        return int(self.aligned.numel())

    def get_dataset_name(self) -> str:
        """Return a stable dataset token used in result paths."""
        return "GW"


class _DummyBackbone(nn.Module):
    """Backbone stub with the module interface used by trainer code."""

    def __init__(self) -> None:
        """Initialize a tiny module with one parameter."""
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0), requires_grad=True)


class _DummyProjector(nn.Module):
    """Projector stub for single-projector control-flow tests."""

    def __init__(self) -> None:
        """Initialize a tiny module with one parameter."""
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return features unchanged.

        Args:
            x (torch.Tensor): Input feature tensor.

        Returns:
            torch.Tensor: Unchanged input tensor.
        """
        return x


def test_alternating_refinement_runs_backbone_only_without_projector_loops(
    monkeypatch,
    tmp_path,
) -> None:
    """Backbone-only refinement should still accept one projector argument.

    This catches regressions where trainer control flow starts assuming a
    projector list again even when pseudo-labeling is disabled.
    """
    monkeypatch.chdir(tmp_path)
    dataset = _DummyDataset(aligned=[0, -1])
    backbone = _DummyBackbone()
    projector = _DummyProjector()
    test_dataset = _DummyDataset(aligned=[-1, -1])

    call_counter = {"metrics": 0}

    monkeypatch.setattr(trainer, "maybe_load_backbone", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(trainer, "HTRDataset", lambda **_kwargs: test_dataset)
    monkeypatch.setattr(trainer, "compute_cer", lambda *_args, **_kwargs: 0.1)
    monkeypatch.setattr(trainer, "compute_wer", lambda *_args, **_kwargs: 0.2)
    monkeypatch.setattr(
        trainer,
        "log_round_metrics",
        lambda *_args, **_kwargs: call_counter.__setitem__("metrics", call_counter["metrics"] + 1),
    )
    monkeypatch.setattr(trainer.cfg, "device", "cpu", raising=False)
    monkeypatch.setattr(trainer.cfg, "align_device", "cpu", raising=False)
    monkeypatch.setattr(trainer.cfg, "eval_batch_size", 2, raising=False)

    trainer.alternating_refinement(
        dataset,
        backbone,
        projector,
        rounds=1,
        backbone_epochs=0,
        projector_epochs=0,
        enable_pseudo_labeling=False,
    )

    assert call_counter["metrics"] == 1


def test_alternating_refinement_calls_align_once_with_single_projector(
    monkeypatch,
    tmp_path,
) -> None:
    """Pseudo-labeling mode should align exactly once with the single projector.

    This protects the post-ensemble control flow so alignment is driven by one
    projector path instead of reintroduced multi-projector loops.
    """
    monkeypatch.chdir(tmp_path)
    dataset = _DummyDataset(aligned=[0, -1])
    backbone = _DummyBackbone()
    projector = _DummyProjector()
    test_dataset = _DummyDataset(aligned=[-1, -1])

    state = {"align_calls": 0, "train_calls": 0}

    def _fake_align(ds, _backbone, proj, **_kwargs):
        state["align_calls"] += 1
        assert proj is projector
        # Simulate one new pseudo-label so the trainer path can verify state updates.
        ds.aligned[1] = 1
        return (
            torch.zeros((2, 2), dtype=torch.float32),
            torch.zeros((2, 2), dtype=torch.float32),
            torch.zeros((2,), dtype=torch.float32),
        )

    monkeypatch.setattr(trainer, "maybe_load_backbone", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(trainer, "HTRDataset", lambda **_kwargs: test_dataset)
    monkeypatch.setattr(trainer, "refine_visual_backbone", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        trainer,
        "train_projector",
        lambda *_args, **_kwargs: state.__setitem__("train_calls", state["train_calls"] + 1),
    )
    monkeypatch.setattr(trainer, "align_more_instances", _fake_align)
    monkeypatch.setattr(trainer, "compute_cer", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(trainer, "compute_wer", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(trainer, "log_round_metrics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(trainer.cfg, "device", "cpu", raising=False)
    monkeypatch.setattr(trainer.cfg, "align_device", "cpu", raising=False)
    monkeypatch.setattr(trainer.cfg, "eval_batch_size", 2, raising=False)

    trainer.alternating_refinement(
        dataset,
        backbone,
        projector,
        rounds=0,
        backbone_epochs=0,
        projector_epochs=0,
        enable_pseudo_labeling=True,
    )

    assert state["align_calls"] == 1
    assert state["train_calls"] == 0
    assert int((dataset.aligned != -1).sum().item()) == 2
