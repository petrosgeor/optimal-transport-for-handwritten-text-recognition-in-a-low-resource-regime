import torch
from typing import List, Dict, Tuple
import math


def encode_for_ctc(
    transcriptions: List[str],
    c2i: Dict[str, int],
    device: torch.device = None
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




def _logaddexp(a: float, b: float) -> float:
    """Stable log‑sum‑exp for two scalars (Python’s math.logaddexp is ≥3.11)."""
    if a == -float("inf"):   # handle −∞ quickly
        return b
    if b == -float("inf"):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))

def beam_search_ctc_decode(
    logits: "torch.Tensor",
    i2c: Dict[int, str],
    *,
    beam_width: int = 10,
    blank_id: int = 0,
    time_first: bool = True,
    lm: callable | None = None,
    lm_weight: float = 0.0,
    length_norm: bool = True,
) -> List[str]:
    """
    Beam‑search decoding for CTC outputs **with proper blank/non‑blank tracking**.
    The interface matches `greedy_ctc_decode` (plus beam/L‑M parameters).
    """
    import torch  # late import keeps signature clean

    if not time_first:                       # (B,T,C) → (T,B,C)
        logits = logits.transpose(0, 1)
    T, B, C = logits.shape
    logp = logits.log_softmax(dim=2).cpu()   # work on CPU → fewer GPU syncs
    results: List[str] = []

    for b in range(B):
        # Beam state: prefix → (P_blank, P_nonblank, last_idx)
        beams: Dict[Tuple[str, int | None], Tuple[float, float]] = {
            ("", None): (0.0, -float("inf"))      # log(1)=0
        }

        for t in range(T):
            step: Dict[Tuple[str, int | None], Tuple[float, float]] = {}
            for (pref, last), (p_b, p_nb) in beams.items():
                for c in range(C):
                    p = logp[t, b, c].item()

                    if c == blank_id:  # emit blank
                        key = (pref, None)             # reset *last*
                        pb_old, pnb_old = step.get(key, (-float("inf"), -float("inf")))
                        step[key] = (_logaddexp(pb_old, p_b + p),
                                     _logaddexp(pnb_old, p_nb + p))
                        continue

                    ch = i2c.get(c)
                    if ch is None:                      # unknown / padding
                        continue

                    # ---------- extend with a non‑blank label ----------
                    new_pref = pref
                    emit = True
                    if c == last:                       # duplicate char
                        if last is None:                # last time‑step was blank ⇒ OK to add
                            new_pref += ch
                        else:
                            emit = False                # repeated without blank
                    else:
                        new_pref += ch

                    key = (new_pref, c)
                    pb_old, pnb_old = step.get(key, (-float("inf"), -float("inf")))

                    if emit:
                        new_p_nb = p_b + p             # came from blank
                    else:
                        new_p_nb = -float("inf")       # illegal transition

                    # also possible to stay non‑blank and repeat the same char
                    cont_p_nb = p_nb + p if c == last else -float("inf")

                    step[key] = (pb_old,
                                 _logaddexp(pnb_old, _logaddexp(new_p_nb, cont_p_nb)))

            # ---------- pruning ----------
            scored = [ (pref, last, pb, pnb,
                        (pb if pb > pnb else pnb))           # log‑prob of prefix
                      for (pref, last), (pb, pnb) in step.items() ]

            # keep best *beam_width* prefixes (optionally length‑norm in view of next step)
            scored.sort(key=lambda x: x[4], reverse=True)
            beams = { (pref,last): (pb,pnb) for pref, last, pb, pnb, _ in scored[:beam_width] }

        # ---------- final selection ----------
        best_pref, best_score = "", -float("inf")
        for (pref, _), (pb, pnb) in beams.items():
            score = _logaddexp(pb, pnb)                # total log‑prob
            if lm is not None:
                score += lm_weight * lm(pref)
            if length_norm and len(pref) > 0:
                score /= len(pref)
            if score > best_score:
                best_pref, best_score = pref, score

        results.append(best_pref.strip())

    return results
