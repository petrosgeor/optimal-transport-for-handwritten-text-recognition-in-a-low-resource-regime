
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple, List

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from htr_base.utils.htr_dataset import HTRDataset
from htr_base.models import HTRNet
from alignment.ctc_utils import encode_for_ctc

# --------------------------------------------------------------------------- #
#                               Helper utilities                              #
# --------------------------------------------------------------------------- #

def _build_vocab_dicts(dataset: HTRDataset) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create <char→id> / <id→char> dicts leaving index 0 for the CTC blank."""
    chars: List[str] = list(dataset.character_classes)
    if " " not in chars:
        chars.append(" ")
    chars = sorted(set(chars))
    c2i = {c: i + 1 for i, c in enumerate(chars)}
    i2c = {i + 1: c for i, c in enumerate(chars)}
    return c2i, i2c


def _ctc_loss_fn(
    logits: torch.Tensor,
    targets: torch.IntTensor,
    inp_lens: torch.IntTensor,
    tgt_lens: torch.IntTensor,
) -> torch.Tensor:
    """A thin wrapper around `torch.nn.functional.ctc_loss` that takes *logits*."""
    log_probs = F.log_softmax(logits, dim=2)
    return F.ctc_loss(
        log_probs,
        targets,
        inp_lens,
        tgt_lens,
        reduction="mean",
        zero_infinity=True,
    )

# --------------------------------------------------------------------------- #
#                            Main refinement routine                          #
# --------------------------------------------------------------------------- #

def refine_visual_backbone(
    dataset: HTRDataset,
    backbone: HTRNet,
    num_epochs: int,
    *,
    batch_size: int = 128,
    lr: float = 1e-4,
    main_weight: float = 1.0,
    aux_weight: float = 0.1,
) -> None:
    """Fine‑tune *backbone* only on words already aligned to external words."""
    print(f"[Refine] epochs={num_epochs}  batch_size={batch_size}  lr={lr}")

    device = next(backbone.parameters()).device
    backbone.train().to(device)

    # Build CTC mapping once.
    c2i, _ = _build_vocab_dicts(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,            # keep CI simple
        pin_memory=(device.type == "cuda"),
    )

    optimizer = optim.AdamW(backbone.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        epoch_loss: float = 0.0
        effective_batches = 0

        for imgs, _, aligned in dataloader:  # we ignore the transcription string
            aligned_mask = aligned != -1
            if not aligned_mask.any():
                # nothing to learn from this mini‑batch
                continue

            # ── select only aligned items ────────────────────────────────
            sel_idx = aligned_mask.nonzero(as_tuple=True)[0]             # (K,)
            imgs_sel = imgs[sel_idx].to(device)                          # (K,1,H,W)
            aligned_ids = aligned[sel_idx].tolist()                     # List[int]
            ext_words = [dataset.external_words[i] for i in aligned_ids]

            # ── forward ─────────────────────────────────────────────────
            out = backbone(imgs_sel, return_feats=False)
            if not isinstance(out, (tuple, list)) or len(out) < 2:
                raise RuntimeError("Expected network.forward() → (main, aux, …)")
            main_logits, aux_logits = out[:2]           # (T,K,C)
            T, K, _ = main_logits.shape

            # ── encode labels ───────────────────────────────────────────
            targets, tgt_lens = encode_for_ctc(ext_words, c2i, device=device)
            inp_lens = torch.full((K,), T, dtype=torch.int32, device=device)

            # ── losses ─────────────────────────────────────────────────
            loss_main = _ctc_loss_fn(main_logits, targets, inp_lens, tgt_lens)
            loss_aux  = _ctc_loss_fn(aux_logits,  targets, inp_lens, tgt_lens)
            loss = main_weight * loss_main + aux_weight * loss_aux

            # ── optimisation step ──────────────────────────────────────
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            effective_batches += 1

        if effective_batches:
            avg_loss = epoch_loss / effective_batches
            print(f"Epoch {epoch:03d}/{num_epochs} – avg loss: {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch:03d}/{num_epochs} – no aligned batch encountered")

    print("[Refine] finished.")

# --------------------------------------------------------------------------- #
#                               Dummy quick test                              #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    """Run a *tiny* end‑to‑end refinement cycle to verify code execution."""
    from types import SimpleNamespace

    # ── 1. Dataset with 200 external words and a handful of alignments ─────
    proj_root = Path(__file__).resolve().parents[1]
    gw_folder = proj_root / "htr_base" / "data" / "GW" / "processed_words"
    if not gw_folder.exists():
        raise RuntimeError(
            "GW processed dataset not found – please generate it with "
            "`python htr_base/prepare_gw.py` before running this dummy test."
        )

    class DummyCfg:
        k_external_words = 200   # top‑200 most frequent English words
        n_aligned = 1000          # how many lines to mark as aligned (≈ training signal)

    dataset = HTRDataset(
        str(gw_folder),
        subset="train",
        fixed_size=(128, 256),
        transforms=None,
        config=DummyCfg(),
    )

    # ── 2. Minimal network able to run – small for CI speed ────────────────
    arch_cfg = SimpleNamespace(
        cnn_cfg=[[2, 64], 'M', [3, 128], 'M', [2, 256]],
        head_type="both",          # produces (main_logits, aux_logits)
        rnn_type="gru",
        rnn_layers=2,
        rnn_hidden_size=128,
        flattening="maxpool",
        stn=False,
        feat_dim=None,
    )


    nclasses = len(dataset.character_classes) + 1  # +1 for CTC blank
    net = HTRNet(arch_cfg, nclasses)

    # ── 3. One quick refinement epoch ──────────────────────────────────────
    refine_visual_backbone(
        dataset,
        net,
        num_epochs=100,      # just to smoke‑test the loop
        batch_size=128,
        lr=1e-4,
    )
