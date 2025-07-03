"""Minimal test suite for htr_new."""

from pathlib import Path
import sys
from omegaconf import OmegaConf
import torch
import itertools

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment.ctc_utils import ctc_target_probability
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


def test_ctc_target_probability():
    """Probability of a short target via dynamic programming."""
    logits = torch.tensor([
        [0.1, 0.7, 0.2],
        [0.6, 0.2, 0.2],
        [0.1, 0.8, 0.1],
    ])
    c2i = {"a": 1, "b": 2}
    prob = ctc_target_probability(logits, "a", c2i)

    log_probs = logits.log_softmax(dim=1).exp()
    brute = 0.0
    for path in itertools.product(range(3), repeat=3):
        p = 1.0
        prev = None
        collapsed = []
        for t, idx in enumerate(path):
            p *= log_probs[t, idx].item()
            if idx != prev and idx != 0:
                collapsed.append("a" if idx == 1 else "b")
            prev = idx
        if "".join(collapsed) == "a":
            brute += p
    assert abs(prob - brute) < 1e-6


def test_ctc_target_probability_longer():
    """Dynamic vs brute force probability for a longer string."""
    logits = torch.tensor([
        [0.6, 0.2, 0.2],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.6, 0.2, 0.2],
        [0.1, 0.8, 0.1],
        [0.6, 0.2, 0.2],
    ])
    c2i = {"a": 1, "b": 2}
    target = "aaba"
    prob = ctc_target_probability(logits, target, c2i)

    log_probs = logits.log_softmax(dim=1).exp()
    brute = 0.0
    for path in itertools.product(range(3), repeat=6):
        p = 1.0
        prev = None
        collapsed = []
        for t, idx in enumerate(path):
            p *= log_probs[t, idx].item()
            if idx != prev and idx != 0:
                collapsed.append("a" if idx == 1 else "b")
            prev = idx
        if "".join(collapsed) == target:
            brute += p
    assert abs(prob - brute) < 1e-6

