"""Regression tests for the single-projector OT alignment contract.

These tests pin down the behavior introduced when ensemble projector logic
was removed. They document that candidate selection is entropy-based,
projector outputs come from a single model path, and pseudo-label updates do
not overwrite already aligned samples.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from alignment import alignment_utilities as au


class _DummyDataset:
    """Minimal dataset stub for OTAligner tests."""

    def __init__(self, aligned: list[int]) -> None:
        """Store alignment state and word metadata.

        Args:
            aligned (list[int]): Initial alignment ids (-1 marks unaligned).
        """
        self.aligned = torch.tensor(aligned, dtype=torch.int32)
        self.unique_word_embeddings = torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
        self.unique_words = ["a", "b"]
        self.transcriptions = ["a", "b", "b"][: len(aligned)]
        self.transforms = None

    def __len__(self) -> int:
        """Return dataset size."""
        return int(self.aligned.numel())

    def get_dataset_name(self) -> str:
        """Return a supported dataset token for helper code."""
        return "GW"


class _DummyBackbone(nn.Module):
    """Backbone stub that exposes module interfaces used by OTAligner."""

    def __init__(self) -> None:
        """Initialize dummy module."""
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0), requires_grad=False)


class _IdentityProjector(nn.Module):
    """Projector stub that forwards descriptors unchanged."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return descriptors unchanged.

        Args:
            x (torch.Tensor): Input descriptors.

        Returns:
            torch.Tensor: Unmodified descriptors.
        """
        return x


def test_select_candidates_prefers_low_entropy_rows() -> None:
    """Pseudo-label selection should rank unlabeled rows by lowest entropy.

    This protects the entropy-only candidate selection contract so refactors
    do not drift back toward gap-based or ensemble-style selection logic.
    """
    ds = _DummyDataset(aligned=[-1, -1, -1])
    aligner = au.OTAligner(
        ds,
        _DummyBackbone(),
        _IdentityProjector(),
        device="cpu",
        k=2,
    )
    plan = torch.tensor(
        [
            [0.99, 0.01],  # sharply peaked row: most confident candidate
            [0.50, 0.50],  # diffuse row: least confident candidate
            [0.90, 0.10],  # intermediate confidence
        ],
        dtype=torch.float32,
    )
    chosen = aligner._select_candidates(plan, ds.aligned.clone())
    assert chosen.tolist() == [0, 2]


def test_get_projector_outputs_returns_single_projector_shapes(
    monkeypatch: object,
) -> None:
    """Projector output helper should expose the single-projector contract.

    This catches regressions where the helper quietly reintroduces ensemble
    aggregation semantics instead of returning one plan and one prediction path.
    """
    ds = _DummyDataset(aligned=[-1, -1])
    aligner = au.OTAligner(
        ds,
        _DummyBackbone(),
        _IdentityProjector(),
        device="cpu",
        k=1,
    )

    feats_all = torch.tensor([[0.1, 0.0], [1.9, 0.0]], dtype=torch.float32)
    aligned_all = torch.tensor([-1, -1], dtype=torch.int32)

    def _fake_harvest(*_args, **_kwargs):
        return feats_all, aligned_all

    def _fake_calc(_proj_feats: torch.Tensor):
        plan_np = np.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=np.float64)
        return feats_all.clone(), plan_np

    monkeypatch.setattr(au, "harvest_backbone_features", _fake_harvest)
    monkeypatch.setattr(aligner, "_calculate_ot", _fake_calc)

    out = aligner._get_projector_outputs()
    assert out["plan"].shape == (2, 2)
    assert out["proj_feats"].shape == (2, 2)
    assert out["dist_matrix"].shape == (2, 2)
    assert out["moved_dist"].shape == (2,)
    assert out["nearest_word"].tolist() == [0, 1]
    assert out["aligned_all"].tolist() == [-1, -1]


def test_align_more_instances_updates_only_unaligned_and_returns_tuple(
    monkeypatch: object,
) -> None:
    """Pseudo-label updates should affect only unaligned rows.

    This protects the alignment update contract so new pseudo-labels never
    overwrite seed labels while preserving the helper's tuple return shape.
    """
    ds = _DummyDataset(aligned=[0, -1, -1])
    backbone = _DummyBackbone()
    projector = _IdentityProjector()

    plan = torch.tensor(
        [
            [0.99, 0.01],
            [0.80, 0.20],
            [0.55, 0.45],
        ],
        dtype=torch.float32,
    )
    proj_feats = torch.tensor(
        [
            [0.0, 0.0],
            [1.5, 0.0],
            [1.2, 0.0],
        ],
        dtype=torch.float32,
    )
    moved = torch.tensor([0.0, 0.1, 0.2], dtype=torch.float32)
    nearest = torch.tensor([0, 1, 1], dtype=torch.long)
    aligned_all = ds.aligned.clone()

    def _fake_get_outputs(self):
        return {
            "plan": plan,
            "proj_feats": proj_feats,
            "dist_matrix": torch.zeros((3, 2), dtype=torch.float32),
            "moved_dist": moved,
            "nearest_word": nearest,
            "aligned_all": aligned_all,
        }

    monkeypatch.setattr(au.OTAligner, "_get_projector_outputs", _fake_get_outputs)

    plan_out, proj_out, moved_out = au.align_more_instances(
        ds,
        backbone,
        projector,
        device="cpu",
        k=1,
    )

    assert plan_out.shape == (3, 2)
    assert proj_out.shape == (3, 2)
    assert moved_out.shape == (3,)
    # Seed labels are authoritative and must never be overwritten.
    assert ds.aligned[0].item() == 0
    # Only the top-ranked unlabeled sample should be promoted when k=1.
    assert int((ds.aligned != -1).sum().item()) == 2
