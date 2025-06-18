"""Test script that instantiates both the HTRDataset **and** an HTRNet
model so that downstream unit‑tests can exercise the full pipeline.

Run directly:
    python tests/test.py
"""

from pathlib import Path
import sys
from types import SimpleNamespace
import torch

# ──────────────────────────────────────────────────────────────────────────────
# Ensure project root (one level up) is importable
# ──────────────────────────────────────────────────────────────────────────────
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

# Local imports (resolved *after* sys.path fix)
from htr_base.utils.htr_dataset import HTRDataset
from htr_base.models import HTRNet


class SimpleCfg:
    """Lightweight namespace mirroring the main YAML defaults."""

    def __init__(self):
        self.k_external_words = 200      # size of external vocabulary
        self.n_aligned = 50             # pre‑aligned seed samples
        self.word_emb_dim = 512         # dimensionality of word embeddings


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 1. Create dataset instance
    # -------------------------------------------------------------------------
    proj_root = Path(__file__).resolve().parents[1]
    gw_folder = proj_root / "htr_base" / "data" / "GW" / "processed_words"
    if not gw_folder.exists():
        raise RuntimeError(
            "GW processed dataset not found – generate it first with "
            "htr_base/prepare_gw.py"
        )

    dataset = HTRDataset(
        basefolder=str(gw_folder),
        subset="all",              # use train+val+test for the smoke test
        fixed_size=(64, 256),
        transforms=None,
        config=SimpleCfg(),
    )

    # Optionally visualise distribution of external‑word hits
    dataset.external_word_histogram()

    print("Dataset created successfully!")
    print(f"Number of samples          : {len(dataset):5d}")
    print(f"Character classes (excluding CTC blank): {len(dataset.character_classes)}")
    print(f"External vocabulary size   : {len(dataset.external_words)}")

    # -------------------------------------------------------------------------
    # 2. Instantiate a minimal HTRNet backbone + CTC head
    # -------------------------------------------------------------------------
    arch_cfg = SimpleNamespace(
        cnn_cfg=[[2, 64], "M", [3, 128], "M", [2, 256]],
        head_type="both",          # main + auxiliary CTC heads
        rnn_type="gru",
        rnn_layers=3,
        rnn_hidden_size=256,
        flattening="maxpool",
        stn=False,
        feat_dim=512,              # per‑image descriptor (unused here)
    )

    n_classes = len(dataset.character_classes) + 1  # +1 for CTC blank
    model = HTRNet(arch_cfg, nclasses=n_classes)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"HTRNet instantiated – trainable parameters: {n_params:,}")

    # Forward a single mini‑batch to validate shapes (no training)
    print(dataset.letter_priors())
    sample_imgs, _, _ = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
    with torch.no_grad():
        out = model(sample_imgs)
        if arch_cfg.head_type == "both":
            main_logits, aux_logits = out[:2]
            print(main_logits.shape)
            print("Forward pass OK – main logits shape:", main_logits.shape,
                  "aux logits shape:", aux_logits.shape)
        else:
            print("Forward pass OK – logits shape:", out.shape)
