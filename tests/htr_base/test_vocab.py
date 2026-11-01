"""Unit tests for the dataset charsets and their union vocabulary."""

from __future__ import annotations

import pytest

from htr_base.utils.vocab import CHARSETS, load_vocab


def test_all_datasets_is_union_of_paper_charsets() -> None:
    """ALL_DATASETS should be the sorted union of the GW, IAM and CVL charsets."""
    c2i, i2c = load_vocab("ALL_DATASETS")
    expected = sorted(set(CHARSETS["GW"]) | set(CHARSETS["IAM"]) | set(CHARSETS["CVL"]))
    assert list(c2i) == expected
    assert len(c2i) == 83  # 82 from earlier runs plus the GW pound sign
    assert min(c2i.values()) == 1  # 0 stays the CTC blank
    assert i2c[c2i[expected[0]]] == expected[0]


def test_bessarion_is_not_a_known_dataset() -> None:
    """Bessarion is outside the paper scope and has no charset."""
    with pytest.raises(ValueError):
        load_vocab("BESSARION")
