from __future__ import annotations
import warnings
import logging

warnings.filterwarnings(
    "ignore",
    message="kenlm python bindings are not installed.*"
)
# Suppress pyctcdecode warnings about missing kenlm bindings
logging.getLogger("pyctcdecode.decoder").setLevel(logging.ERROR)

import torch
from typing import List, Dict, Tuple, Callable
import math
from pyctcdecode import build_ctcdecoder


def encode_for_ctc(
    transcriptions: List[str],
    c2i: Dict[str, int],
    device: torch.device | str = None
) -> Tuple[torch.IntTensor, torch.IntTensor]:
    """Encode transcriptions into flattened CTC targets and per-sample lengths.

    Args:
    - transcriptions (list[str]): Input strings for a batch of size B.
      Boundary whitespace is stripped before encoding.
    - c2i (dict[str, int]): Character-to-index map where index 0 is reserved
      for the CTC blank symbol.
    - device (torch.device | str | None): Optional destination device.

    Returns:
    - (Tuple[torch.IntTensor, torch.IntTensor]): Encoded tensors.
      targets has shape (sum(L),), lengths has shape (B,).
    """
    normalized = [str(txt).strip() for txt in transcriptions]
    for idx, txt in enumerate(normalized):
        if txt == "":
            raise ValueError(
                "encode_for_ctc received an empty transcription after stripping "
                f"at index {idx}. Original value: {transcriptions[idx]!r}"
            )

    # --- build the flattened label vector ---
    flat_labels = [c2i[ch] for txt in normalized for ch in txt]
    targets = torch.tensor(flat_labels, dtype=torch.int32, device=device)

    # --- per-sample lengths (needed by CTCLoss) ---
    lengths = torch.tensor([len(txt) for txt in normalized],
                           dtype=torch.int32, device=device)

    return targets, lengths


def _unflatten_targets(targets: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
    """Convert flattened CTC targets back to per-sample index sequences.

    Args:
    - targets (torch.Tensor): Flattened target indices with shape (sum(L),).
    - lengths (torch.Tensor): Per-sample target lengths with shape (B,).

    Returns:
    - (list[list[int]]): Decoded index lists of length B where each element has
      length given by the corresponding entry in lengths.
    """
    out = []
    i = 0
    for length in lengths:
        out.append(targets[i : i + length].tolist())
        i += length
    return out




def greedy_ctc_decode(
    logits: torch.Tensor,
    i2c: Dict[int, str],
    blank_id: int = 0,
    time_first: bool = True,
) -> List[str]:
    """Greedy-decode a batch of CTC logits.

    Args:
    - logits (torch.Tensor): Network scores with shape (T, B, C) when
      time_first=True, otherwise shape (B, T, C).
    - i2c (dict[int, str]): Index-to-character map excluding the blank id.
    - blank_id (int): Index used for the CTC blank symbol.
    - time_first (bool): Layout flag indicating whether logits are time-major.

    Returns:
    - (List[str]): Decoded strings of length B after repeat-collapsing, blank
      removal, and boundary whitespace stripping.
    """
    # ---- bring tensor to (B, T) of arg-max indices ----
    if time_first:
        # logits: (T, B, C) → (T, B)
        argmax = logits.argmax(dim=2).transpose(0, 1)   # → (B, T)
    else:
        # logits: (B, T, C) → (B, T)
        argmax = logits.argmax(dim=2)

    decoded: List[str] = []
    for seq in argmax:                    # iterate over the batch
        prev = None
        chars: List[str] = []
        for idx in seq.tolist():
            # 1) squash repeats
            if idx == prev:
                continue
            prev = idx
            # 2) skip blanks
            if idx == blank_id:
                continue
            # 3) map index → character
            chars.append(i2c[idx])
        decoded.append("".join(chars).strip())

    return decoded




from pyctcdecode import build_ctcdecoder

# --------------------------------------------------------------------------- #
#                           Beam-search CTC decoding                          #
# --------------------------------------------------------------------------- #

def beam_search_ctc_decode(
    logits: torch.Tensor,
    i2c: Dict[int, str],
    *,
    beam_width: int = 10,
    blank_id: int = 0,
    time_first: bool = True,
) -> List[str]:
    """Beam-search decode a batch of CTC logits with pyctcdecode.

    Args:
    - logits (torch.Tensor): Network scores with shape (T, B, C) when
      time_first=True, otherwise shape (B, T, C).
    - i2c (dict[int, str]): Index-to-character map excluding the blank id.
    - beam_width (int): Number of kept prefixes per time step.
    - blank_id (int): Index used for the CTC blank symbol.
    - time_first (bool): Layout flag indicating whether logits are time-major.

    Returns:
    - (List[str]): Best-scoring decoded strings of length B with boundary
      whitespace stripped.
    """
    if not time_first:
        logits = logits.transpose(0, 1)

    # Determine the maximum index in i2c to know the total number of classes (C)
    # The number of classes should be max_idx + 1, or blank_id + 1 if blank_id is the highest.
    max_idx_in_i2c = max(i2c.keys()) if i2c else -1
    num_classes = max(max_idx_in_i2c + 1, blank_id + 1)

    # Create a list for the vocabulary, initialized with a placeholder for the blank token.
    # The actual string for the blank token doesn't matter for pyctcdecode, as long as its index is correct.
    # We'll use a special string '<blank>' for clarity, though an empty string also works.
    vocab_list = [""] * num_classes # Initialize with empty strings

    # Populate the vocabulary list with characters from i2c
    for idx, char in i2c.items():
        if idx < num_classes:
            vocab_list[idx] = char

    # Set the blank token at its designated ID.
    # pyctcdecode expects the blank token to be at the blank_id index in the labels list.
    vocab_list[blank_id] = "" # An empty string is a common convention for the blank token in pyctcdecode.

    decoder = build_ctcdecoder(labels=vocab_list)

    # Convert logits to numpy array for pyctcdecode
    log_probs = logits.log_softmax(dim=2).cpu().numpy()

    results: List[str] = []
    for b in range(log_probs.shape[1]): # Iterate over batch size
        # Check if the blank token is consistently the most probable
        if (log_probs[:, b, :].argmax(axis=1) == blank_id).all():
            decoded_text = ""
        else:
            # pyctcdecode expects a 2D array (time_steps, num_classes) for a single sample
            decoded_text = decoder.decode(log_probs[:, b, :], beam_width=beam_width)
        results.append(decoded_text.strip())

    return results


def ctc_target_probability(
    logits: torch.Tensor,
    target: str,
    c2i: Dict[str, int],
    *,
    blank_id: int = 0,
) -> float:
    """Compute the CTC path probability of one target string.

    Args:
    - logits (torch.Tensor): Single-sample logits with shape (T, C), where C
      includes the blank class.
    - target (str): Target transcription composed of characters from c2i.
    - c2i (dict[str, int]): Character-to-index map for non-blank symbols.
    - blank_id (int): Index used for the CTC blank symbol.

    Returns:
    - (float): Probability that logits collapse to target under the CTC
      forward algorithm.
    """

    log_probs = logits.log_softmax(dim=1)

    # Extended label sequence with blanks inserted between characters and
    # at the beginning and end as per the CTC forward algorithm.
    ext_labels = [blank_id]
    for ch in target:
        ext_labels.append(c2i[ch])
        ext_labels.append(blank_id)

    T = log_probs.size(0)
    S = len(ext_labels)

    # Forward variables initialised in log-space for numerical stability
    alpha = torch.full((T, S), float("-inf"), device=log_probs.device)
    alpha[0, 0] = log_probs[0, ext_labels[0]]
    if S > 1:
        alpha[0, 1] = log_probs[0, ext_labels[1]]

    for t in range(1, T):
        for s in range(S):
            prev = alpha[t - 1, s]
            if s > 0:
                prev = torch.logaddexp(prev, alpha[t - 1, s - 1])
            if (
                s > 1
                and ext_labels[s] != blank_id
                and ext_labels[s] != ext_labels[s - 2]
            ):
                prev = torch.logaddexp(prev, alpha[t - 1, s - 2])
            alpha[t, s] = prev + log_probs[t, ext_labels[s]]

    log_prob = torch.logaddexp(alpha[T - 1, S - 1], alpha[T - 1, S - 2])
    return float(log_prob.exp().item())
