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
    """
    Convert a batch of raw string transcriptions to the (targets, lengths)
    format expected by ``nn.CTCLoss``.

    Parameters
    ----------
    transcriptions : list[str]
        Each element is a single line/word **already wrapped with leading
        and trailing spaces** (the dataset does that for you).
    c2i : dict[str, int]
        Character-to-index mapping where **index 0 is reserved for CTC blank**
        and every real character starts at 1.
    device : torch.device, optional
        If given, the returned tensors are moved to this device.

    Returns
    -------
    targets : torch.IntTensor   # shape = (total_chars,)
        All label indices concatenated in batch order.
    lengths : torch.IntTensor   # shape = (batch,)
        The original length (in characters) of every element in *transcriptions*.
    """
    # --- build the flattened label vector ---
    flat_labels = [c2i[ch] for txt in transcriptions for ch in txt]
    targets = torch.tensor(flat_labels, dtype=torch.int32, device=device)

    # --- per-sample lengths (needed by CTCLoss) ---
    lengths = torch.tensor([len(txt) for txt in transcriptions],
                           dtype=torch.int32, device=device)

    return targets, lengths


def _unflatten_targets(targets: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
    """Convert flattened CTC targets to a list of lists."""
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
    """
    Greedy-decode a batch of CTC network outputs.

    Parameters
    ----------
    logits : torch.Tensor
        Tensor shaped either (T, B, C) if *time_first* is True
        **or** (B, T, C) if *time_first* is False, where  
        *T = time steps*, *B = batch size*, *C = n_classes*.
        It can contain raw scores, logits, or probabilities –
        only the arg-max along *C* is used.
    i2c : dict[int, str]
        Index-to-character mapping that complements the *c2i*
        used during encoding. It must **not** contain the blank id.
    blank_id : int, optional
        Integer assigned to the CTC blank (defaults to 0).
    time_first : bool, optional
        Set **True** if logits are (T, B, C); otherwise set False
        for (B, T, C).

    Returns
    -------
    List[str]
        One decoded string per element in the mini-batch.
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
        decoded.append("".join(chars))

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
    """
    Beam-search decoding for CTC outputs using pyctcdecode.

    Parameters
    ----------
    logits : torch.Tensor
        Network output – either ``(T, B, C)`` if *time_first* or ``(B, T, C)``.
    i2c : dict[int, str]
        Index-to-character map **excluding** the blank id.
    beam_width : int, optional
        Number of prefixes kept after every time-step.
    blank_id : int, optional
        Integer assigned to the CTC blank (defaults to ``0``).
    time_first : bool, optional
        ``True`` if *logits* are ``(T, B, C)``, else ``False`` for ``(B, T, C)``.

    Returns
    -------
    list[str]
        Best-scoring transcription for every element in the mini-batch.
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
        results.append(decoded_text)

    return results

