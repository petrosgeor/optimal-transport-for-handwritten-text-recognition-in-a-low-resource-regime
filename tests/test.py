# alignment/alignment_trainer.py
"""
Refinement-stage trainer that
1. **trains only on images whose line is already aligned** to an entry of
   `dataset.external_words`.
2. **every 5 epochs** evaluates the CTC *main head* on the **test split** and
   prints the Character Error Rate (CER).

This version fixes three bugs that prevented CER from improving:
* network now has a logit column for every character **plus blank + space**
* labels fed to CTC no longer add superfluous leading/trailing blanks
* CER compares *stripped* predictions and targets symmetrically
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
from htr_base.utils.transforms import aug_transforms
from alignment.ctc_utils import encode_for_ctc, greedy_ctc_decode

# --------------------------------------------------------------------------- #
#                               Helper utilities                              #
# --------------------------------------------------------------------------- #

def _build_vocab_dicts(dataset: HTRDataset) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create <char→id> / <id→char> dicts leaving index 0 for the CTC blank."""
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
        reduction="sum",
        zero_infinity=True,
    )


def _evaluate_cer(
    model: HTRNet,
    loader: DataLoader,
    i2c: Dict[int, str],
    device,
    show_max: int = 5,                 # <-- how many examples to echo
) -> float:
    """Compute CER on *loader* with greedy CTC decoding
    and print a few (gt , pred) pairs for inspection."""
    model.eval()
    cer = CER()
    shown = 0                          # counter for printed samples

    with torch.no_grad():
        for imgs, transcrs, _ in loader:
            imgs = imgs.to(device)

            logits = model(imgs, return_feats=False)
            if isinstance(logits, (tuple, list)):     # keep main logits only
                logits = logits[0]
                
            preds = greedy_ctc_decode(logits, i2c)    # list[str]

            for pred, tgt in zip(preds, transcrs):
                # optional console preview
                if shown < show_max:
                    print(f"GT:   '{tgt.strip()}'\nPRD:  '{pred.strip()}'\n")
                    shown += 1

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
    """Fine-tune *backbone* only on aligned samples & evaluate CER every 5 epochs."""
    print(f"[Refine] epochs={num_epochs}  batch_size={batch_size}  lr={lr}")

    device = next(backbone.parameters()).device
    # backbone.train().to(device)

    # Build CTC mapping once (train + test share the same vocab).
    c2i, i2c = _build_vocab_dicts(dataset)

    # Training loader
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Test loader
    test_set = HTRDataset(
        dataset.basefolder, subset="test", fixed_size=dataset.fixed_size,
        transforms=None, config=dataset.config,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = optim.AdamW(backbone.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        epoch_loss, effective_batches = 0.0, 0

        for imgs, _, aligned in train_loader:
            mask = aligned != -1
            if not mask.any():
                continue

            sel = mask.nonzero(as_tuple=True)[0]
            imgs_sel = imgs[sel].to(device)
            ids = aligned[sel].tolist()
            ext_words = [f' {dataset.external_words[i]} ' for i in ids]  # no blanks!

            main_logits, aux_logits = backbone(imgs_sel, return_feats=False)[:2]
            # print(main_logits.shape)
            T, K, _ = main_logits.shape

            targets, tgt_lens = encode_for_ctc(ext_words, c2i, device='cpu')
            # print(targets)
            # print(tgt_lens)
            inp_lens = torch.full((K,), T, dtype=torch.int32, device='cpu')

            # print(main_logits.device, targets.device, inp_lens.device, tgt_lens.device)
            loss_main = _ctc_loss_fn(main_logits, targets, inp_lens, tgt_lens)
            loss_aux  = _ctc_loss_fn(aux_logits,  targets, inp_lens, tgt_lens)
            loss = main_weight * loss_main + aux_weight * loss_aux

            optimizer.zero_grad(set_to_none=True)
            loss.backward(); optimizer.step()

            epoch_loss += loss.item(); effective_batches += 1

        avg_loss = epoch_loss / max(1, effective_batches)
        print(f"Epoch {epoch:03d}/{num_epochs} – avg train loss: {avg_loss:.4f}")

        if epoch % 5 == 0 or epoch == num_epochs:
            cer = _evaluate_cer(backbone, test_loader, i2c, device)
            print(f"[Eval] CER on test set @ epoch {epoch}: {cer:.4f}")

    print("[Refine] finished.")

# --------------------------------------------------------------------------- #
#                               Dummy quick test                              #
# --------------------------------------------------------------------------- #

from types import SimpleNamespace

proj_root = Path(__file__).resolve().parents[1]
gw_folder = proj_root / "htr_base" / "data" / "GW" / "processed_words"
if not gw_folder.exists():
    raise RuntimeError("GW processed dataset not found – generate it first!")

class DummyCfg:
    k_external_words = 200
    n_aligned = 1000

train_set = HTRDataset(
    str(gw_folder), subset="train", fixed_size=(128, 256),
    transforms=aug_transforms, config=DummyCfg(),
)

# print('################, ', torch.where(train_set.is_in_dict == 1)[0].numel())

c2i, _ = _build_vocab_dicts(train_set)

# arch_cfg = SimpleNamespace(
#     cnn_cfg=[[2, 64], "M", [3, 128], "M", [2, 256]],
#     head_type="both", rnn_type="gru", rnn_layers=3,
#     rnn_hidden_size=256, flattening="maxpool", stn=False, feat_dim=None,
# )
# net = HTRNet(arch_cfg, nclasses=len(c2i) + 1)  # chars + blank
# net.to('cuda')


# refine_visual_backbone(
#     train_set, net, num_epochs=100, batch_size=256, lr=1e-3,
# )
