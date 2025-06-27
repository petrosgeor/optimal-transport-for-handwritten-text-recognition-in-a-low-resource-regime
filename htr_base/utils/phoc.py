import torch
from typing import List, Dict


def build_phoc_description(words: List[str],
                           c2i: Dict[str, int],
                           *,
                           levels: List[int] = (1, 2, 3, 4)) -> torch.Tensor:
    """Convert *words* into binary PHOC descriptors.

    Parameters
    ----------
    words : list[str]
        Words to encode.
    c2i : dict[str, int]
        Mapping from character to index; index 0 is the blank.
    levels : list[int], optional
        Pyramid levels. Default ``(1, 2, 3, 4)``.

    Returns
    -------
    torch.BoolTensor
        Tensor of shape ``(B, |c2i|-1 * sum(levels))`` with PHOC descriptors.
    """
    # alphabet excluding blank id 0, ordered by index
    alphabet = [c for c, i in sorted(c2i.items(), key=lambda kv: kv[1]) if i != 0]
    char_to_pos = {c: p for p, c in enumerate(alphabet)}
    dim_per_level = len(alphabet)
    total_dim = dim_per_level * sum(levels)

    phoc = torch.zeros(len(words), total_dim, dtype=torch.bool)

    for w_idx, word in enumerate(words):
        text = word.strip().lower()
        offset = 0
        for l in levels:
            seg_len = max(len(text), 1) / l
            for seg in range(l):
                start = int(round(seg * seg_len))
                end = int(round((seg + 1) * seg_len))
                segment_chars = set(text[start:end])
                for ch in segment_chars:
                    pos = char_to_pos.get(ch)
                    if pos is not None:
                        phoc[w_idx, offset + pos] = True
            offset += dim_per_level

    return phoc
