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
        Mapping from character to index. This dictionary **does not**
        contain the CTC blank which is implicitly reserved at index 0.
    levels : list[int], optional
        Pyramid levels. Default ``(1, 2, 3, 4)``.

    Returns
    -------
    torch.BoolTensor
        Tensor of shape ``(B, len(c2i) * sum(levels))`` with PHOC descriptors.
    """
    # alphabet ordered by index (index 0 reserved for the CTC blank)
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
