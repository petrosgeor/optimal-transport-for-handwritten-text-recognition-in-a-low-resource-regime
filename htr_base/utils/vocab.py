"""Utility helpers for the fixed character vocabulary."""

import pickle
from pathlib import Path
from typing import Dict, Tuple


def create_vocab() -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create the default vocabulary pickles and return the dictionaries.

    The mapping contains digits ``0``–``9``, lowercase ``a``–``z`` and a
    space character.  Index ``0`` is reserved for the CTC blank.
    """

    chars = list("0123456789abcdefghijklmnopqrstuvwxyz ")
    c2i = {c: i + 1 for i, c in enumerate(chars)}
    i2c = {i + 1: c for i, c in enumerate(chars)}

    base = Path(__file__).resolve().parents[2] / "htr_base" / "saved_models"
    base.mkdir(parents=True, exist_ok=True)
    with open(base / "c2i.pkl", "wb") as f:
        pickle.dump(c2i, f)
    with open(base / "i2c.pkl", "wb") as f:
        pickle.dump(i2c, f)

    return c2i, i2c


def load_vocab() -> Tuple[Dict[str, int], Dict[int, str]]:
    """Load ``c2i`` and ``i2c`` dictionaries from ``saved_models``.

    If the pickle files do not exist, :func:`create_vocab` is called to
    generate them first.
    """

    base = Path(__file__).resolve().parents[2] / "htr_base" / "saved_models"
    c2i_path = base / "c2i.pkl"
    i2c_path = base / "i2c.pkl"
    if not c2i_path.exists() or not i2c_path.exists():
        create_vocab()

    with open(c2i_path, "rb") as f:
        c2i = pickle.load(f)
    with open(i2c_path, "rb") as f:
        i2c = pickle.load(f)

    return c2i, i2c
