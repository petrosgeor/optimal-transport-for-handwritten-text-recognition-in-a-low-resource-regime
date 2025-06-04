from __future__ import annotations
import os, sys, argparse, random

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("-g", "--gpu", type=str,
                   help="Comma‑separated visible device IDs (e.g. '0,1')")
    known, rest = p.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = known.gpu or "0"
    # give the original argv back so downstream flags are untouched
    sys.argv = [sys.argv[0]] + rest

from pathlib import Path
from typing import Dict, Tuple, List

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, SubsetRandomSampler

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from htr_base.utils.htr_dataset import HTRDataset
from htr_base.models import HTRNet
from htr_base.utils.metrics import CER
from htr_base.utils.transforms import aug_transforms
from alignment.ctc_utils import encode_for_ctc, greedy_ctc_decode


def _build_vocab_dicts(ds: HTRDataset) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Return char→id / id→char dicts with index 0 reserved for the blank."""
    chars: List[str] = list(ds.character_classes)
    if " " not in chars:
        chars.append(" ")
    chars = sorted(set(chars))
    c2i = {c: i + 1 for i, c in enumerate(chars)}  # +1 → leave 0 for blank
    i2c = {i + 1: c for i, c in enumerate(chars)}
    return c2i, i2c


def _ctc_loss_fn(logits: torch.Tensor,
                 targets: torch.IntTensor,
                 inp_lens: torch.IntTensor,
                 tgt_lens: torch.IntTensor) -> torch.Tensor:
    """Length‑normalised CTC loss on raw logits."""
    return F.ctc_loss(F.log_softmax(logits, 2), targets, inp_lens, tgt_lens,
                      reduction="mean", zero_infinity=True)


def _evaluate_cer(model: HTRNet, loader: DataLoader, i2c: Dict[int, str],
                  device, show_max: int = 5) -> float:
    """Compute CER over *loader* and print a few (gt, pred) pairs."""
    model.eval(); cer = CER(); shown = 0
    with torch.no_grad():
        for imgs, transcrs, _ in loader:
            imgs = imgs.to(device)
            logits = model(imgs, return_feats=False)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            preds = greedy_ctc_decode(logits, i2c)
            for p, t in zip(preds, transcrs):
                if shown < show_max:
                    print(f"GT: '{t.strip()}'\nPR: '{p.strip()}'\n"); shown += 1
                cer.update(p.strip(), t.strip())
    model.train(); return cer.score()


def refine_visual_backbone(dataset: HTRDataset,
                           backbone: HTRNet,
                           num_epochs: int,
                           *,
                           batch_size: int = 128,
                           lr: float = 1e-4,
                           main_weight: float = 1.0,
                           aux_weight: float = 0.1,
                           mode: str = "aligned") -> None:
    """Fine‑tune *backbone* either on aligned words or on random GT lines."""
    if mode not in {"aligned", "ground_truth"}:
        raise ValueError("mode must be 'aligned' or 'ground_truth'")

    print(f"[Refine] mode={mode} epochs={num_epochs} batch={batch_size} lr={lr}")
    device = next(backbone.parameters()).device

    # Vocab
    c2i, i2c = _build_vocab_dicts(dataset)

    # Base loader (shuffled)
    base_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=(device.type == "cuda"))

    # Test loader
    test_set = HTRDataset(dataset.basefolder, subset="test",
                          fixed_size=dataset.fixed_size, transforms=None,
                          config=dataset.config)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=(device.type == "cuda"))

    # Optimiser & scheduler
    opt = optim.AdamW(backbone.parameters(), lr=lr, weight_decay=1e-4)
    sched = lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)

    n_aligned = getattr(dataset.config, "n_aligned", len(dataset))

    for epoch in range(1, num_epochs + 1):
        # Re‑sample subset each epoch if using GT mode
        if mode == "ground_truth":
            idx = random.sample(range(len(dataset)), k=min(n_aligned, len(dataset)))
            train_loader = DataLoader(dataset, batch_size=batch_size,
                                      sampler=SubsetRandomSampler(idx),
                                      num_workers=0,
                                      pin_memory=(device.type == "cuda"))
        else:
            train_loader = base_loader

        epoch_loss = 0.0; effective_batches = 0
        for imgs, trans, aligned in train_loader:
            if mode == "aligned":
                mask = aligned != -1
                if not mask.any():
                    continue
                sel = mask.nonzero(as_tuple=True)[0]
                imgs_sel = imgs[sel].to(device)
                ids = aligned[sel].tolist()
                targets_s = [f" {dataset.external_words[i]} " for i in ids]
            else:  # ground_truth
                imgs_sel = imgs.to(device)
                targets_s = [t if t.startswith(" ") else f" {t.strip()} " for t in trans]

            # Forward
            main_logits, aux_logits = backbone(imgs_sel, return_feats=False)[:2]
            T, B, _ = main_logits.shape

            # Labels → CPU
            targets, tgt_lens = encode_for_ctc(targets_s, c2i, device="cpu")
            inp_lens = torch.full((B,), T, dtype=torch.int32)

            # Loss
            loss_m = _ctc_loss_fn(main_logits, targets, inp_lens, tgt_lens)
            loss_a = _ctc_loss_fn(aux_logits,  targets, inp_lens, tgt_lens)
            loss = main_weight * loss_m + aux_weight * loss_a

            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            epoch_loss += loss.item(); effective_batches += 1

        sched.step()
        avg_loss = epoch_loss / max(1, effective_batches)
        print(f"Epoch {epoch:03}/{num_epochs}  loss={avg_loss:.4f}  lr={sched.get_last_lr()[0]:.2e}")

        if epoch % 5 == 0 or epoch == num_epochs:
            cer = _evaluate_cer(backbone, test_loader, i2c, device)
            print(f"[Eval] CER @ epoch {epoch}: {cer:.4f}")

    print("[Refine] finished.")


if __name__ == "__main__":
    from types import SimpleNamespace

    proj_root = Path(__file__).resolve().parents[1]
    gw_folder = proj_root / "htr_base" / "data" / "GW" / "processed_words"
    if not gw_folder.exists():
        raise RuntimeError("GW processed dataset not found – generate it first!")

    class DummyCfg:
        k_external_words = 200
        n_aligned = 500

    train_set = HTRDataset(str(gw_folder), subset="train", fixed_size=(64, 256),
                            transforms=aug_transforms, config=DummyCfg(),)

    c2i, _ = _build_vocab_dicts(train_set)
    arch_cfg = SimpleNamespace(
        cnn_cfg=[[2, 64], "M", [3, 128], "M", [2, 256]],
        head_type="both", rnn_type="gru", rnn_layers=3,
        rnn_hidden_size=256, flattening="maxpool", stn=False, feat_dim=None,
    )
    net = HTRNet(arch_cfg, nclasses=len(c2i) + 1)
    net.to("cuda")

    refine_visual_backbone(train_set, net, num_epochs=400, batch_size=128,
                           lr=1e-3, mode="ground_truth")
