"""
tests/evaluate_pretrained_cer.py

This script evaluates the Character Error Rate (CER) of the pretrained HTRNet backbone
on the test split of the HTRDataset.
"""
from pathlib import Path
import sys
import torch
from types import SimpleNamespace
from omegaconf import OmegaConf

# ─── Add project root to import path ────────────────────────────────────────
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from htr_base.utils.vocab import load_vocab
from htr_base.utils.htr_dataset import HTRDataset
from htr_base.models import HTRNet
from alignment.eval import compute_cer

# ─── 1. Load Configuration and Vocabulary ───────────────────────────────────
print("Loading configuration and vocabulary...")
c2i, i2c = load_vocab()
pretraining_config = OmegaConf.load(root / "alignment" / "alignment_configs" / "pretraining_config.yaml")
arch_cfg = SimpleNamespace(**pretraining_config.architecture)
device = pretraining_config.device

# ─── 2. Instantiate and Load Model ──────────────────────────────────────────
print("Instantiating and loading the pretrained model...")
model = HTRNet(arch_cfg, nclasses=len(c2i) + 1)
pretrained_path = root / "htr_base" / "saved_models" / "pretrained_backbone.pt"
model.load_state_dict(torch.load(pretrained_path, map_location=device))
model.to(device)

# ─── 3. Prepare Dataset ─────────────────────────────────────────────────────
print("Preparing the test dataset...")
dataset_path = str(root / "htr_base" / "data" / "GW" / "processed_words")
test_dataset = HTRDataset(
    basefolder=dataset_path,
    subset="test",
    fixed_size=(64, 256)
)

# ─── 4. Calculate and Report CER ────────────────────────────────────────────
print("Calculating Character Error Rate (CER)...")
cer_score = compute_cer(
    dataset=test_dataset,
    model=model,
    device=device,
    decode="beam",
    beam_width=10
)

print(f"\nFinal CER on the test set: {cer_score:.4f}")
