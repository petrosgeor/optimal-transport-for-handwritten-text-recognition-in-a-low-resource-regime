"""Unit tests for HTRDataset configuration validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from htr_base.utils.htr_dataset import HTRDataset


def test_unknown_word_prob_strategy_raises(tmp_path) -> None:
    """An unsupported word prior must fail loudly instead of silently using another prior."""
    cfg_path = (
        Path(__file__).resolve().parents[2] / "alignment/alignment_configs/trainer_config.yaml"
    )
    ds_cfg = OmegaConf.load(cfg_path).dataset
    ds_cfg.word_prob_strategy = "wordfreq"
    with pytest.raises(ValueError, match="word_prob_strategy"):
        HTRDataset(basefolder=str(tmp_path), subset="all", fixed_size=(64, 256), config=ds_cfg)
