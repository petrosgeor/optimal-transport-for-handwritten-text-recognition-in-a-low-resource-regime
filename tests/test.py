# alignment/alignment_trainer.py
"""
Refinement‑stage trainer that
1. **trains only on images whose line is already aligned** to an entry of
   `dataset.external_words` (as before)
2. **every 5 epochs** evaluates the CTC *main head* on the **test split** and
   prints the Character Error Rate (CER).

A quick demo at the bottom still trains for a single epoch, but you can change
`num_epochs` to see the evaluation trigger.
"""
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
from htr_base.utils.metrics import CER
from alignment.ctc_utils import encode_for_ctc, greedy_ctc_decode

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
    """Balanced CTC loss helper that takes raw logits."""
    log_probs = F.log_softmax(logits, dim=2)
    return F.ctc_loss(
        log_probs,
        targets,
        inp_lens,
        tgt_lens,
        reduction="mean",
        zero_infinity=True,
    )


def _evaluate_cer(model: HTRNet, loader: DataLoader, i2c: Dict[int, str], device) -> float:
    """Compute CER of *model* on *loader* using greedy CTC decoding."""
    model.eval()
    cer = CER()
    with torch.no_grad():
        for imgs, transcrs, _ in loader:
            imgs = imgs.to(device)
            out = model(imgs, return_feats=False)
            if isinstance(out, (tuple, list)):
                out = out[0]  # keep main logits only
            preds = greedy_ctc_decode(out, i2c)
            for pred, tgt in zip(preds, transcrs):
                cer.update(pred.strip(), tgt.strip())
    model.train()
    return cer.score()

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
    """Fine‑tune *backbone* only on aligned samples & evaluate CER every 5 epochs."""
    print(f"[Refine] epochs={num_epochs}  batch_size={batch_size}  lr={lr}")

    device = next(backbone.parameters()).device
    backbone.train().to(device)

    # Build CTC mapping once (train + test share the same vocab).
    c2i, i2c = _build_vocab_dicts(dataset)

    # Training data loader (only `dataset` passed in).
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Prepare **test** dataset/loader for periodic evaluation.
    test_set = HTRDataset(
        dataset.basefolder,
        subset="test",
        fixed_size=dataset.fixed_size,
        transforms=None,
        config=dataset.config,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = optim.AdamW(backbone.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        epoch_loss: float = 0.0
        effective_batches = 0

        for imgs, _, aligned in train_loader:  # ignore transcription
            aligned_mask = aligned != -1
            if not aligned_mask.any():
                continue  # skip unaligned mini‑batch

            sel_idx = aligned_mask.nonzero(as_tuple=True)[0]
            imgs_sel = imgs[sel_idx].to(device)
            aligned_ids = aligned[sel_idx].tolist()
            ext_words = [f" {dataset.external_words[i]} " for i in aligned_ids]

            # forward
            main_logits, aux_logits = backbone(imgs_sel, return_feats=False)[:2]
            T, K, _ = main_logits.shape

            # encode labels
            targets, tgt_lens = encode_for_ctc(ext_words, c2i, device=device)
            inp_lens = torch.full((K,), T, dtype=torch.int32, device=device)

            # losses
            loss_main = _ctc_loss_fn(main_logits, targets, inp_lens, tgt_lens)
            loss_aux  = _ctc_loss_fn(aux_logits,  targets, inp_lens, tgt_lens)
            loss = main_weight * loss_main + aux_weight * loss_aux

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            effective_batches += 1

        if effective_batches:
            avg_loss = epoch_loss / effective_batches
            print(f"Epoch {epoch:03d}/{num_epochs} – avg train loss: {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch:03d}/{num_epochs} – no aligned batch encountered")

        # ── periodic evaluation every 5 epochs ────────────────────────────
        if epoch % 5 == 0 or epoch == num_epochs:
            cer_score = _evaluate_cer(backbone, test_loader, i2c, device)
            print(f"[Eval] CER on test set @ epoch {epoch}: {cer_score:.4f}")

    print("[Refine] finished.")

# --------------------------------------------------------------------------- #
#                               Dummy quick test                              #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    """Run a *tiny* end‑to‑end refinement cycle to verify code execution."""
    from types import SimpleNamespace

    proj_root = Path(__file__).resolve().parents[1]
    gw_folder = proj_root / "htr_base" / "data" / "GW" / "processed_words"
    if not gw_folder.exists():
        raise RuntimeError(
            "GW processed dataset not found – please generate it with "
            "`python htr_base/prepare_gw.py` before running this dummy test."
        )

    class DummyCfg:
        k_external_words = 200
        n_aligned = 500

    train_set = HTRDataset(
        str(gw_folder),
        subset="train",
        fixed_size=(128, 256),
        transforms=None,
        config=DummyCfg(),
    )

    arch_cfg = SimpleNamespace(
        cnn_cfg=[[2, 32], "M", [2, 64], "M", [2, 128]],
        head_type="both",
        rnn_type="gru",
        rnn_layers=2,
        rnn_hidden_size=128,
        flattening="maxpool",
        stn=False,
        feat_dim=None,
    )

    nclasses = len(train_set.character_classes) + 1
    net = HTRNet(arch_cfg, nclasses)

    refine_visual_backbone(
        train_set,
        net,
        num_epochs=100,  # triggers one evaluation at epoch 5 and final at 6
        batch_size=256,
        lr=1e-4,
    )
