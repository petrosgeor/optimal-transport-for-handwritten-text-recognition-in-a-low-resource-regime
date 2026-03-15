"""Unit tests for PHOC target construction."""

from __future__ import annotations

import torch

from htr_base.utils.phoc import build_phoc_description


def test_phoc_segment_blocks_are_level_and_segment_specific() -> None:
    """PHOC bins should be separated per level segment."""
    c2i = {"a": 1, "b": 2}
    words = ["ab"]
    levels = (1, 2)

    phoc = build_phoc_description(words, c2i, levels=levels)

    expected = torch.tensor([[1, 1, 1, 0, 0, 1]], dtype=torch.bool)
    assert phoc.shape == (1, 6)
    assert torch.equal(phoc, expected)


def test_phoc_output_dim_matches_vocab_without_blank() -> None:
    """PHOC second dimension should match A * sum(levels)."""
    c2i = {"blank_like": 0, "a": 1, "b": 2, "c": 3}
    words = ["abc", "cab"]
    levels = (1, 2, 4)

    phoc = build_phoc_description(words, c2i, levels=levels)

    alphabet_size = 3
    assert phoc.shape == (2, alphabet_size * sum(levels))


def test_phoc_preserves_case_in_character_bins() -> None:
    """Uppercase and lowercase characters should map to distinct bins."""
    c2i = {"a": 1, "A": 2}
    words = ["A", "a"]
    levels = (1,)

    phoc = build_phoc_description(words, c2i, levels=levels)

    expected = torch.tensor(
        [
            [0, 1],  # "A"
            [1, 0],  # "a"
        ],
        dtype=torch.bool,
    )
    assert torch.equal(phoc, expected)


def test_phoc_ignores_unknown_characters() -> None:
    """Characters missing from c2i should not activate PHOC bins."""
    c2i = {"a": 1}
    words = ["bbb"]
    levels = (1, 2)

    phoc = build_phoc_description(words, c2i, levels=levels)

    assert phoc.shape == (1, 3)
    assert not phoc.any()
