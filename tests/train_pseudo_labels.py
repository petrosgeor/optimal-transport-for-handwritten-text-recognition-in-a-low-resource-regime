#!/usr/bin/env python3
"""
Fine‑tune an HTRNet **only on pseudo‑labelled instances**
logged by `alignment/trainer.py::log_pseudo_labels`.

The script expects one or more files named
    results/pseudo_labels_round_<n>.txt
where each line is
    <idx>\t<predicted_word>\t<ground_truth_word>

*Only* the predicted_word is used as training target.

Example
-------
python train_pseudo_labels.py \
       --results_dir results \
       --pretrained_backbone htr_base/saved_models/pretrained_backbone.pt \
       --batch_size 64 --epochs 200 --eval_every 20
"""
from __future__ import annotations
import argparse, glob, os, random
from pathlib import Path
from types import SimpleNamespace

# ------------------------------------------------------------------
# Hyperparameters used when running this file directly. They also
# serve as defaults for the command-line interface.
#
#   GPU_ID           – CUDA device identifier.
#   BATCH_SIZE       – mini-batch size during training.
#   RESULTS_DIR      – folder containing `pseudo_labels_round_*.txt`.
#   NUM_EPOCHS       – number of training epochs.
#   LR               – optimiser learning rate.
#   EVAL_EVERY       – evaluation frequency in epochs.
#   PRETRAINED_BACKBONE – path to pretrained weights.
#   MAIN_LOSS_WEIGHT – weight for the main CTC loss.
#   AUX_LOSS_WEIGHT – weight for the auxiliary CTC loss.
# ------------------------------------------------------------------
GPU_ID = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

BATCH_SIZE = 64
RESULTS_DIR = "results"
NUM_EPOCHS = 600
LR = 1e-3
EVAL_EVERY = 20
PRETRAINED_BACKBONE = "htr_base/saved_models/pretrained_backbone.pt"
MAIN_LOSS_WEIGHT = 1.0
AUX_LOSS_WEIGHT = 0.1

import torch, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from omegaconf import OmegaConf

import sys

# ─── Add project root to import path ────────────────────────────────────────
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from htr_base.utils.htr_dataset import (
    HTRDataset,
    PretrainingHTRDataset,
    FusedHTRDataset,
)
from htr_base.models import HTRNet
from htr_base.utils.transforms import aug_transforms
from htr_base.utils.vocab import load_vocab
from alignment.ctc_utils import encode_for_ctc, greedy_ctc_decode
from alignment.eval import compute_cer
from alignment.losses import _ctc_loss_fn          # balanced CTC wrapper

# ──────────────────── helper --------------------------------------------------
def _parse_pseudo_files(
    results_dir: str,
    rounds: list[int] | None = None,
    exclude_false: bool = False,
) -> tuple[dict[int, str], int]:
    """Collect pseudo-labels and count matches with ground truth.

    Scan ``results_dir`` for files called ``pseudo_labels_round_*.txt`` produced
    by ``log_pseudo_labels`` and return two values:

    1. ``dict`` mapping dataset indices to their latest predicted word.
    2. ``int`` count of indices whose predicted word equals the ground truth.

    When the same index appears in multiple rounds, the *latest* label is kept
    for training and for the correctness count.

    Args:
        results_dir: Folder containing pseudo-label text files.
        rounds: Specific pseudo-label rounds to include. ``None`` loads all
            available rounds.
        exclude_false: Drop rows whose prediction does not match the
            ground truth.

    Returns:
        Tuple[Dict[int, str], int]: Final mapping and number of correct labels.
    """
    mapping: dict[int, str] = {}
    ground_truth: dict[int, str] = {}
    if rounds is None:
        paths = sorted(Path(results_dir).glob("pseudo_labels_round_*.txt"))
    else:
        paths = [Path(results_dir) / f"pseudo_labels_round_{r}.txt" for r in rounds]
    if not paths:
        raise FileNotFoundError(
            f"No pseudo‑label files found in {results_dir!s}"
        )
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"{p!s} not found")
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                idx_s, pred, gt = line.rstrip("\n").split("\t")
                if exclude_false and pred != gt:
                    continue
                idx = int(idx_s)
                mapping[idx] = pred
                ground_truth[idx] = gt

    correct = sum(
        1 for idx, pred in mapping.items() if pred == ground_truth.get(idx)
    )
    return mapping, correct

def _maybe_load_pretrained(net: HTRNet, path: str) -> None:
    """Load weights into *net* if *path* exists (strict=False)."""
    if Path(path).is_file():
        state = torch.load(path, map_location="cpu")
        net.load_state_dict(state, strict=False)
        print(f"[Init] loaded pretrained backbone from {path}")
    else:
        raise FileNotFoundError(f"pretrained_backbone '{path}' not found")


# ──────────────────── main training routine ----------------------------------
def main(args) -> None:
    """Train an HTRNet using pseudo-labelled and synthetic words.

    Args:
        args: Command-line arguments controlling the training run.

    Returns:
        None
    """
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    # 1. ── configuration from trainer_config.yaml for consistency ──────────
    config_path = root / "alignment" / "alignment_configs" / "trainer_config.yaml"
    cfg = OmegaConf.load(config_path)
    arch_cfg = SimpleNamespace(**cfg["architecture"])
    ds_cfg = cfg.dataset
    syn_cfg = cfg.synthetic_dataset

    fixed_size = tuple(ds_cfg.fixed_size)

    # 2. ── build dataset & inject pseudo‑labels ─────────────────────────────
    basefolder = ds_cfg.basefolder
    real_ds = HTRDataset(
        basefolder=basefolder,
        subset="train_val",
        fixed_size=tuple(ds_cfg.fixed_size),
        transforms=aug_transforms,
        config=ds_cfg,
        two_views=ds_cfg.two_views,
        word_prob_mode=ds_cfg.word_prob_mode,
    )

    syn_ds = PretrainingHTRDataset(
        list_file=syn_cfg.list_file,
        base_path=syn_cfg.base_path,
        n_random=syn_cfg.n_random,
        fixed_size=tuple(syn_cfg.fixed_size),
        transforms=aug_transforms,
        preload_images=syn_cfg.preload_images,
        random_seed=syn_cfg.random_seed,
    )

    full_ds = FusedHTRDataset(real_ds, syn_ds, n_aligned=ds_cfg.n_aligned)

    pseudo_map, n_correct = _parse_pseudo_files(
        args.results_dir,
        args.include_rounds,
        exclude_false=args.exclude_false,
    )
    if not pseudo_map:
        raise RuntimeError("No pseudo‑labels left after filtering – aborting.")

    for idx, pred in pseudo_map.items():
        full_ds.transcriptions[idx] = pred
        word_id = full_ds.unique_words.index(pred)
        full_ds.aligned[idx] = word_id

    aligned_idx = torch.nonzero(full_ds.aligned != -1, as_tuple=True)[0]

    unique_words = {
        full_ds.transcriptions[i].strip().lower() for i in aligned_idx.tolist()
    }
    print(
        f"[Data] using {len(aligned_idx)} pseudo‑labelled samples from {len(unique_words)} unique words"
    )
    print(
        f"[Data] {n_correct} out of {len(aligned_idx)} pseudo-labels match the ground truth"
    )

    # 3. ── model, vocab, optimiser ─────────────────────────────────────────
    c2i, i2c = load_vocab()
    net = HTRNet(arch_cfg, nclasses=len(c2i) + 1)
    _maybe_load_pretrained(net, args.pretrained_backbone)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device).train()

    opt = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=200, gamma=0.1)

    # 4. ── loaders ─────────────────────────────────────────────────────────
    train_loader = DataLoader(
        torch.utils.data.Subset(full_ds, aligned_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    test_ds = HTRDataset(
        basefolder=basefolder,
        subset="test",
        fixed_size=fixed_size,
        transforms=aug_transforms,
        config=ds_cfg,
    )

    # 5. ── training loop with CER evaluation ──────────────────────────────
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        batches = 0
        for batch in train_loader:
            imgs, words, _ = batch
            imgs = imgs.to(device)
            words = [f" {w.strip()} " if not w.startswith(" ") else w for w in words]

            outputs = net(imgs, return_feats=False)
            if isinstance(outputs, (tuple, list)):
                main_logits, aux_logits = outputs[0], outputs[1]
            else:
                main_logits, aux_logits = outputs, None

            targets, t_lens = encode_for_ctc(words, c2i, device="cpu")
            inp_lens = torch.full((imgs.size(0),), main_logits.size(0), dtype=torch.int32)

            loss_main = _ctc_loss_fn(main_logits, targets, inp_lens, t_lens)
            if aux_logits is not None:
                loss_aux = _ctc_loss_fn(aux_logits, targets, inp_lens, t_lens)
            else:
                loss_aux = torch.tensor(0.0, device=device)

            loss = args.main_loss_weight * loss_main + args.aux_loss_weight * loss_aux
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running_loss += loss.item()
            batches += 1

        if batches:
            print(
                f"Epoch {epoch:03d}/{args.epochs} – "
                f"avg CTC loss {running_loss / batches:.4f}"
            )

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            compute_cer(
                dataset=test_ds,
                model=net,
                batch_size=args.batch_size,
                device=device,
                decode="beam",
                k=4,
            )

        scheduler.step()

    print("Training finished.")

# ──────────────────── CLI boilerplate ────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default=RESULTS_DIR,
                   help="folder containing pseudo_labels_round_*.txt")
    p.add_argument(
        "--include_rounds", nargs="+", type=int, metavar="N",
        help=(
            "Only load pseudo_labels_round_<N>.txt (space-separated list). "
            "If omitted, all rounds are loaded."
        ),
    )
    p.add_argument(
        "--exclude_false",
        action="store_true",
        help="Skip pseudo-label rows where prediction \u2260 ground-truth",
    )
    p.add_argument("--pretrained_backbone", default=PRETRAINED_BACKBONE,
                   help="path to pretrained backbone .pt file")
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--eval_every", type=int, default=EVAL_EVERY)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--gpu_id", type=int, help="GPU to use")
    p.add_argument("--main_loss_weight", type=float, default=MAIN_LOSS_WEIGHT)
    p.add_argument("--aux_loss_weight", type=float, default=AUX_LOSS_WEIGHT)
    args = p.parse_args()
    main(args)
