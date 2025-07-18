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
       --dataset_folder htr_base/data/GW/processed_words \
       --pretrained_backbone htr_base/saved_models/pretrained_backbone.pt \
       --batch_size 128 --epochs 200 --eval_every 20
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
#   SYN_BATCH_RATIO  – fraction of each batch drawn from the synthetic
#                       `PretrainingHTRDataset`.
#   RESULTS_DIR      – folder containing `pseudo_labels_round_*.txt`.
#   DATASET_FOLDER   – processed_words dataset folder.
#   SYN_LIST_FILE    – image list for the synthetic corpus.
#   SYN_BASE_PATH    – base path of the synthetic images.
#   NUM_EPOCHS       – number of training epochs.
#   LR               – optimiser learning rate.
#   EVAL_EVERY       – evaluation frequency in epochs.
#   PRETRAINED_BACKBONE – path to pretrained weights.
#   MAIN_LOSS_WEIGHT – weight for the main CTC loss.
#   AUX_LOSS_WEIGHT – weight for the auxiliary CTC loss.
# ------------------------------------------------------------------
GPU_ID = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

BATCH_SIZE = 128
SYN_BATCH_RATIO = 0.7
RESULTS_DIR = "results"
DATASET_FOLDER = "htr_base/data/GW/processed_words"
SYN_LIST_FILE = "/gpu-data3/pger/handwriting_rec/mnt/ramdisk/max/90kDICT32px/imlist.txt"
SYN_BASE_PATH = "/gpu-data3/pger/handwriting_rec/mnt/ramdisk/max/90kDICT32px"
NUM_EPOCHS = 600
LR = 1e-3
EVAL_EVERY = 20
PRETRAINED_BACKBONE = "htr_base/saved_models/pretrained_backbone.pt"
MAIN_LOSS_WEIGHT = 1.0
AUX_LOSS_WEIGHT = 0.1

import torch, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from omegaconf import OmegaConf

import sys

# ─── Add project root to import path ────────────────────────────────────────
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from htr_base.utils.htr_dataset import HTRDataset, PretrainingHTRDataset
from htr_base.models import HTRNet
from htr_base.utils.transforms import aug_transforms
from htr_base.utils.vocab import load_vocab
from alignment.ctc_utils import encode_for_ctc, greedy_ctc_decode
from alignment.eval import compute_cer
from alignment.losses import _ctc_loss_fn          # balanced CTC wrapper

# ──────────────────── helper --------------------------------------------------
def _parse_pseudo_files(results_dir: str) -> tuple[dict[int, str], int]:
    """Collect pseudo-labels and count matches with ground truth.

    Scan ``results_dir`` for files called ``pseudo_labels_round_*.txt`` produced
    by ``log_pseudo_labels`` and return two values:

    1. ``dict`` mapping dataset indices to their latest predicted word.
    2. ``int`` count of indices whose predicted word equals the ground truth.

    When the same index appears in multiple rounds, the *latest* label is kept
    for training and for the correctness count.

    Args:
        results_dir: Folder containing pseudo-label text files.

    Returns:
        Tuple[Dict[int, str], int]: Final mapping and number of correct labels.
    """
    mapping: dict[int, str] = {}
    ground_truth: dict[int, str] = {}
    paths = sorted(Path(results_dir).glob("pseudo_labels_round_*.txt"))
    if not paths:
        raise FileNotFoundError(
            f"No pseudo‑label files found in {results_dir!s}"
        )
    for p in paths:
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                idx_s, pred, gt = line.rstrip("\n").split("\t")
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

def _mix_batches(real_batch, pre_iter):
    """Concatenate pseudo-labelled and synthetic batches.

    Args:
        real_batch (Tuple[torch.Tensor, list, torch.Tensor] | Tuple[torch.Tensor, list]):
            Batch from the pseudo-labelled loader. The alignment tensor may be present.
        pre_iter (Iterator | None): Cycling iterator over the synthetic loader.

    Returns:
        Tuple[torch.Tensor, list]: Combined images and transcriptions.
    """
    if len(real_batch) == 3:
        imgs, words, _ = real_batch
    else:
        imgs, words = real_batch
    if pre_iter is not None:
        imgs_syn, words_syn = next(pre_iter)
        imgs = torch.cat([imgs, imgs_syn], dim=0)
        words = list(words) + list(words_syn)
    return imgs, list(words)

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
    fixed_size = tuple(cfg.dataset.fixed_size)

    # 2. ── build dataset & inject pseudo‑labels ─────────────────────────────
    basefolder = args.dataset_folder
    ds_cfg = SimpleNamespace(n_aligned=0, word_emb_dim=cfg.dataset.word_emb_dim)
    full_ds = HTRDataset(
        basefolder=basefolder,
        subset="train_val",
        fixed_size=fixed_size,
        transforms=aug_transforms,
        config=ds_cfg,
    )

    pseudo_map, n_correct = _parse_pseudo_files(args.results_dir)
    if not pseudo_map:
        raise RuntimeError("No pseudo‑labels collected → nothing to train on")

    # Overwrite the transcriptions *in‑place* with the predicted words
    for idx, pred in pseudo_map.items():
        full_ds.transcriptions[idx] = pred
        img_path, _old = full_ds.data[idx]
        full_ds.data[idx] = (img_path, pred)

    subset_idx = sorted(pseudo_map.keys())
    train_ds = Subset(full_ds, subset_idx)
    unique_words = {
        full_ds.transcriptions[i].strip().lower()
        for i in subset_idx
    }
    print(
        f"[Data] using {len(train_ds)} pseudo‑labelled samples "
        f"from {len(unique_words)} unique words"
    )
    print(
        f"[Data] {n_correct} out of {len(train_ds)} pseudo-labels match the ground truth"
    )


    pretrain_ds = PretrainingHTRDataset(
        list_file=args.syn_list_file,
        fixed_size=fixed_size,
        base_path=args.syn_base_path,
        transforms=aug_transforms,
        n_random=20000,
        preload_images=False,
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
    syn_bs = int(args.batch_size * args.syn_batch_ratio)
    gt_bs = args.batch_size - syn_bs

    train_loader = DataLoader(
        train_ds,
        batch_size=gt_bs if syn_bs > 0 else args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    if syn_bs > 0:
        pretrain_loader = DataLoader(
            pretrain_ds,
            batch_size=syn_bs,
            shuffle=True,
            num_workers=2,
            pin_memory=(device.type == "cuda"),
        )
        from itertools import cycle
        pre_iter = cycle(pretrain_loader)
    else:
        pre_iter = None
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
            imgs, words = _mix_batches(batch, pre_iter)
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
    p.add_argument("--dataset_folder", default=DATASET_FOLDER,
                   help="root folder with processed_words/{train,val,test}/")
    p.add_argument("--pretrained_backbone", default=PRETRAINED_BACKBONE,
                   help="path to pretrained backbone .pt file")
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--eval_every", type=int, default=EVAL_EVERY)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--gpu_id", type=int, help="GPU to use")
    p.add_argument("--syn_list_file", default=SYN_LIST_FILE,
                   help="text file listing synthetic image paths")
    p.add_argument("--syn_base_path", default=SYN_BASE_PATH,
                   help="root directory of the synthetic corpus")
    p.add_argument("--syn_batch_ratio", type=float, default=SYN_BATCH_RATIO,
                   help="fraction of each batch drawn from synthetic data")
    p.add_argument("--main_loss_weight", type=float, default=MAIN_LOSS_WEIGHT)
    p.add_argument("--aux_loss_weight", type=float, default=AUX_LOSS_WEIGHT)
    args = p.parse_args()
    main(args)
