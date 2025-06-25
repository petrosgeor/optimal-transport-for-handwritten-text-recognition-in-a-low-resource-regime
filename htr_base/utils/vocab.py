"""
This file contains the functionality to load the vocabulary
for the project.
"""

import pickle
from pathlib import Path
from typing import Dict, Tuple


def load_vocab() -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    This function loads the vocabulary from the saved pickle files.
    """

    base = Path(__file__).resolve().parents[2] / 'htr_base' / 'saved_models'
    c2i_path = base / "c2i.pkl"
    i2c_path = base / "i2c.pkl"

    with open(c2i_path, "rb") as f:
        c2i = pickle.load(f)
    with open(i2c_path, "rb") as f:
        i2c = pickle.load(f)

    return c2i, i2c
