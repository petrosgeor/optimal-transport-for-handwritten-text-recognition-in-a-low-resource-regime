"""Utility helpers for dataset-specific character vocabularies.

Builds character→index and index→character mappings in-memory based on a
curated symbol set per dataset. No disk IO is performed.
"""

from typing import Dict, Tuple




CHARSETS = {
    "GW": list("0123456789abcdefghijklmnopqrstuvwxyz "),
    "IAM": [
        ' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
        '0','1','2','3','4','5','6','7','8','9', ':',';','?',
        'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R',
        'S','T','U','V','W','X','Y','Z',
        'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r',
        's','t','u','v','w','x','y','z'
    ],
    # CVL: character set extended to include common symbols.
    # Union of: space, apostrophe, digits 0–9, all uppercase A–Z,
    # all lowercase a–z, and dataset-specific umlauts.
    "CVL": [
        # Whitespace and punctuation
        " ", "'",
        # Digits
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        # Uppercase A–Z
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        # Lowercase a–z
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
        # Umlauts present in CVL
        "ä", "ö", "ü",
    ],
}

# Build a union charset that contains every symbol used across GW, IAM and CVL.
# The ordering is alphabetical/unicode code‑point order for determinism.
_UNION_KEYS = ("GW", "IAM", "CVL")
_union_set = set()
for _k in _UNION_KEYS:
    _union_set.update(CHARSETS[_k])
CHARSETS["ALL_DATASETS"] = sorted(_union_set)

def load_vocab(dataset_name: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Return character-index mappings for a named charset.

    Args:
        dataset_name (str): One of ``'GW'``, ``'IAM'``, ``'CVL'`` or
            ``'ALL_DATASETS'``. The special ``'ALL_DATASETS'`` entry is the
            union of all supported dataset character sets.

    Returns:
        tuple[dict, dict]: ``(c2i, i2c)`` where indices start at 1. Index 0 is
        implicitly reserved for the CTC blank in downstream code.

    Raises:
        ValueError: If ``dataset_name`` is not a key in ``CHARSETS``.
    """

    if dataset_name not in CHARSETS:
        raise ValueError(
            f"dataset_name must be one of {list(CHARSETS.keys())}, got {dataset_name!r}"
        )

    chars = list(CHARSETS[dataset_name])
    c2i = {c: i + 1 for i, c in enumerate(chars)}
    i2c = {i + 1: c for i, c in enumerate(chars)}
    return c2i, i2c
