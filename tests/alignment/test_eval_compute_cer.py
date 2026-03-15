"""Integration tests for alignment.eval.compute_cer."""

from __future__ import annotations

import importlib

import nltk
import pytest
import torch
from torch.utils.data import Dataset

pytest.importorskip("pyctcdecode")


def _import_eval_module():
    """Import alignment.eval while stubbing nltk download side effects.

    Returns:
    - (module): Imported alignment.eval module.
    """
    original_download = nltk.download
    nltk.download = lambda *args, **kwargs: True
    try:
        eval_mod = importlib.import_module("alignment.eval")
    finally:
        nltk.download = original_download
    return eval_mod


def _get_vocab_maps():
    """Return CTC vocabulary maps for the ALL_DATASETS charset.

    Returns:
    - (tuple[dict[str, int], dict[int, str]]): Character-index and inverse maps.
    """
    from htr_base.utils.vocab import load_vocab

    return load_vocab("ALL_DATASETS")


def _text_to_argmax_indices(text: str, c2i: dict[str, int], blank_id: int = 0) -> list[int]:
    """Create a blank-interleaved argmax path that decodes to text.

    Args:
    - text (str): Output text to represent in CTC argmax space.
    - c2i (dict[str, int]): Character-to-index map used by the model.
    - blank_id (int): Blank class index.

    Returns:
    - (list[int]): Time-step class ids whose greedy collapse yields text.
    """
    indices = [blank_id]
    for char in text:
        cls_idx = c2i[char]
        indices.extend([cls_idx, cls_idx, blank_id])
    return indices


class ToyDataset(Dataset):
    """Small deterministic dataset yielding index-coded tiny images.

    Args:
    - transcriptions (list[str]): Ground-truth strings for each sample.

    Returns:
    - None
    """

    def __init__(self, transcriptions: list[str]) -> None:
        """Store fixed transcriptions and an initial transforms sentinel.

        Args:
        - transcriptions (list[str]): Ground-truth strings for each sample.
        """
        self.transcriptions = transcriptions
        self.transforms = "active"

    def __len__(self) -> int:
        """Return dataset size.

        Returns:
        - (int): Number of samples.
        """
        return len(self.transcriptions)

    def __getitem__(self, index: int):
        """Return image, transcription, and sample id.

        Args:
        - index (int): Sample index.

        Returns:
        - (tuple[torch.Tensor, str, int]): Tiny image, transcription, and id.
        """
        image = torch.full((1, 1, 1), float(index), dtype=torch.float32)
        return image, self.transcriptions[index], index

    def get_dataset_name(self) -> str:
        """Return charset name expected by load_vocab in compute_cer.

        Returns:
        - (str): Dataset charset name.
        """
        return "ALL_DATASETS"


class ToyCTCModel(torch.nn.Module):
    """Toy model that emits deterministic CTC logits from sample ids.

    Args:
    - c2i (dict[str, int]): Character-to-index map.
    - predicted_texts (list[str]): Model outputs keyed by sample id.
    - blank_id (int): Blank class index.

    Returns:
    - None
    """

    def __init__(
        self,
        c2i: dict[str, int],
        predicted_texts: list[str],
        blank_id: int = 0,
    ) -> None:
        """Store vocabulary and per-sample output strings.

        Args:
        - c2i (dict[str, int]): Character-to-index map.
        - predicted_texts (list[str]): Model outputs keyed by sample id.
        - blank_id (int): Blank class index.
        """
        super().__init__()
        self.c2i = c2i
        self.predicted_texts = predicted_texts
        self.blank_id = blank_id
        self.num_classes = len(c2i) + 1
        # Keep at least one parameter so model.to(device) is always valid.
        self.anchor = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, imgs: torch.Tensor, return_feats: bool = False) -> torch.Tensor:
        """Emit (T, B, C) logits whose greedy decode matches predicted_texts.

        Args:
        - imgs (torch.Tensor): Mini-batch of tiny index-coded images.
        - return_feats (bool): Unused flag kept for API compatibility.

        Returns:
        - (torch.Tensor): Deterministic CTC logits shaped (T, B, C).
        """
        del return_feats

        sample_ids = imgs[:, 0, 0, 0].round().long().tolist()
        sequences = [
            _text_to_argmax_indices(self.predicted_texts[sample_id], self.c2i, self.blank_id)
            for sample_id in sample_ids
        ]
        max_time = max(len(sequence) for sequence in sequences)
        batch_size = len(sequences)
        logits = torch.full(
            (max_time, batch_size, self.num_classes),
            -20.0,
            dtype=torch.float32,
            device=imgs.device,
        )
        for batch_idx, sequence in enumerate(sequences):
            for time_idx, class_idx in enumerate(sequence):
                logits[time_idx, batch_idx, class_idx] = 20.0
            for time_idx in range(len(sequence), max_time):
                logits[time_idx, batch_idx, self.blank_id] = 20.0
        return logits


def _pick_two_non_space_chars(c2i: dict[str, int]) -> tuple[str, str]:
    """Select two deterministic non-space characters from the vocabulary.

    Args:
    - c2i (dict[str, int]): Character-to-index map.

    Returns:
    - (tuple[str, str]): Two valid non-space characters.
    """
    sorted_chars = [char for char, _ in sorted(c2i.items(), key=lambda item: item[1])]
    non_space_chars = [char for char in sorted_chars if char != " "]
    if len(non_space_chars) < 2:
        raise RuntimeError("ALL_DATASETS vocabulary must contain at least two non-space chars.")
    return non_space_chars[0], non_space_chars[1]


def test_compute_cer_returns_zero_for_perfect_logits() -> None:
    """compute_cer should be zero when model decodes exactly to targets."""
    eval_mod = _import_eval_module()
    compute_cer = eval_mod.compute_cer
    c2i, _ = _get_vocab_maps()
    char_a, char_b = _pick_two_non_space_chars(c2i)
    targets = [f"{char_a}{char_b}", f"{char_b}{char_a}{char_b}"]
    dataset = ToyDataset(targets)
    model = ToyCTCModel(c2i, predicted_texts=targets)

    cer = compute_cer(dataset, model, batch_size=2, device="cpu", decode="greedy")

    assert cer == pytest.approx(0.0, abs=1e-8)
    assert dataset.transforms == "active"


def test_compute_cer_is_positive_for_wrong_logits() -> None:
    """compute_cer should be > 0 when one decoded sample is intentionally wrong."""
    eval_mod = _import_eval_module()
    compute_cer = eval_mod.compute_cer
    c2i, _ = _get_vocab_maps()
    char_a, char_b = _pick_two_non_space_chars(c2i)
    targets = [f"{char_a}{char_b}", f"{char_b}{char_a}{char_b}"]
    wrong_predictions = [targets[0], f"{char_a}{char_a}{char_a}"]
    dataset = ToyDataset(targets)
    model = ToyCTCModel(c2i, predicted_texts=wrong_predictions)

    cer = compute_cer(dataset, model, batch_size=2, device="cpu", decode="greedy")

    assert cer > 0.0
