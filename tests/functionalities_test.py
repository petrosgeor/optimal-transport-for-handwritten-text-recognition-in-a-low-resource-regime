"""Minimal test suite for htr_new."""

from pathlib import Path
import sys
from omegaconf import OmegaConf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment.trainer import _shuffle_batch


def test_trainer_config_has_no_prior_weight():
    """Verify trainer configuration no longer includes ``prior_weight``."""
    cfg = OmegaConf.load("alignment/alignment_configs/trainer_config.yaml")
    assert "prior_weight" not in cfg


def test_shuffle_batch():
    """_shuffle_batch keeps pairs together and randomises order."""
    torch.manual_seed(0)
    imgs = torch.arange(6).view(3, 2)
    words = ["a", "b", "c"]
    shuffled_imgs, shuffled_words = _shuffle_batch(imgs.clone(), list(words))

    expected_pairs = [
        ((4, 5), "c"),
        ((0, 1), "a"),
        ((2, 3), "b"),
    ]
    observed_pairs = [
        (tuple(shuffled_imgs[i].tolist()), shuffled_words[i])
        for i in range(3)
    ]
    assert observed_pairs == expected_pairs
    assert not torch.equal(imgs, shuffled_imgs)

