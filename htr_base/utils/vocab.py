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
    ]
}

def load_vocab(dataset_name: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Return vocabulary mappings for the given dataset.

    Args:
        dataset_name (str): Must be either ``'GW'`` or ``'IAM'``.

    Returns:
        tuple[dict, dict]: ``(c2i, i2c)`` where indices start at 1. Index 0 is
        implicitly reserved for the CTC blank in downstream code.

    Raises:
        ValueError: If ``dataset_name`` is not one of the supported datasets.
    """

    if dataset_name not in CHARSETS:
        raise ValueError(
            f"dataset_name must be one of {list(CHARSETS.keys())}, got {dataset_name!r}"
        )

    chars = list(CHARSETS[dataset_name])
    c2i = {c: i + 1 for i, c in enumerate(chars)}
    i2c = {i + 1: c for i, c in enumerate(chars)}
    return c2i, i2c
