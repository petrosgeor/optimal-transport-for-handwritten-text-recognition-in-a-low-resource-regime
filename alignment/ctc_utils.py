import torch
from typing import List, Dict, Tuple


def encode_for_ctc(
    transcriptions: List[str],
    c2i: Dict[str, int],
    device: torch.device | None = None
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
