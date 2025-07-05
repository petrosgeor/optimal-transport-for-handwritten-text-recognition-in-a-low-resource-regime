"""
tests/test.py
This script demonstrates instantiating HTRDataset and HTRNet with the PHOC head enabled.
It fetches a batch of images, runs a forward pass, and prints the shapes of the
main logits, auxiliary logits, feature descriptors, and PHOC logits.
"""
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader
from types import SimpleNamespace

# ─── Add project root to import path ────────────────────────────────────────
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from htr_base.utils.vocab import load_vocab
from htr_base.utils.htr_dataset import HTRDataset
from alignment.ctc_utils import ctc_target_probability
from htr_base.models import HTRNet

# ─── 1. Load Vocabulary ─────────────────────────────────────────────────────
# The vocabulary is needed to determine the number of output classes for the model.
c2i, i2c = load_vocab()
print(f"Vocabulary loaded with {len(c2i)} characters.")


def count_transcriptions_by_length(dataset: HTRDataset, k: int) -> int:
    """Counts the number of transcriptions in the dataset with length <= k."""
    count = 0
    for transcription in dataset.transcriptions:
        if len(transcription.strip().replace(" ", "")) <= k:
            count += 1
    return count



# ─── 2. Instantiate Dataset and DataLoader ──────────────────────────────────
# We use the small George Washington sample dataset included in the repository.
dataset_path = str(root / "htr_base" / "data" / "GW" / "processed_words")
config = SimpleNamespace(k_external_words=200, n_aligned=5, two_views=False)
dataset = HTRDataset(
    basefolder=dataset_path,
    subset="train_val",
    fixed_size=(64, 256),
    transforms=None,
    config=config
)

words, probs = dataset.word_frequencies()





