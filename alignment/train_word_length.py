"""Train a ResNet-18 model to predict word lengths from images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import os

GPU_ID = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)


# Add project root to import path
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import torch
from torch import nn
from torch.utils.data import DataLoader
from types import SimpleNamespace

from htr_base.models import HTRNet

from htr_base.utils.htr_dataset import PretrainingHTRDataset, HTRDataset
from htr_base.utils.transforms import aug_transforms


def lengths_from_transcriptions(batch_txt: list[str]) -> torch.LongTensor:
    """Return class indices for word lengths between 1 and 20.

    Args:
        batch_txt (list[str]): Transcriptions from the dataset.

    Returns:
        torch.LongTensor: Tensor of shape ``(B,)`` with values ``0``–``19``.
    """
    lengths = [min(max(len(t.strip()), 1), 20) for t in batch_txt]
    return torch.tensor([l - 1 for l in lengths], dtype=torch.long)


def build_resnet18() -> nn.Module:
    """Return a tiny ``HTRNet`` with a length prediction head."""

    arch_cfg = SimpleNamespace(
        cnn_cfg=[[2, 64], "M", [3, 128], "M", [2, 256]],
        head_type="cnn",
        flattening="maxpool",
        feat_dim=512,
        feat_pool="attn",
        length_classes=15,
    )
    net = HTRNet(arch_cfg, nclasses=1)
    return net


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    p = argparse.ArgumentParser(description="Train word-length predictor")
    p.add_argument(
        "list_file",
        nargs="?",
        default="/gpu-data3/pger/handwriting_rec/mnt/ramdisk/max/90kDICT32px/imlist.txt",
        help="Path to 90k synthetic list file. Defaults to the common 90k-synth dataset path.",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-size", type=int, default=50000)
    p.add_argument("--val-size", type=int, default=10000)
    p.add_argument(
        "--save-path",
        type=str,
        default="htr_base/saved_models/length_resnet18.pt",
    )
    p.add_argument(
        "--save-model",
        action="store_true",
        help="If set, save the model with the best validation accuracy.",
    )
    return p.parse_args()


def main() -> None:
    """Entry point for the training script."""

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = PretrainingHTRDataset(
        args.list_file,
        transforms=aug_transforms,
        preload_images=True,
        n_random=args.train_size,
    )
    val_ds = PretrainingHTRDataset(
        args.list_file,
        transforms=None,
        preload_images=True,
        n_random=args.val_size,
        random_seed=1,
    )
    
    htr_val_ds = HTRDataset(
        basefolder=str(root / 'htr_base' / 'data' / 'GW' / 'processed_words'),
        subset='train_val',
        transforms=None,
        fixed_size=(64, 256),
    )


    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=3, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=3, pin_memory=True
    )
    
    htr_val_loader = DataLoader(
        htr_val_ds, batch_size=args.batch_size, shuffle=False, num_workers=3, pin_memory=True
    )


    net = build_resnet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    best_acc = 0.0
    save_path = Path(args.save_path)
    if args.save_model:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        net.train()
        running = 0.0
        for imgs, txts in train_loader:
            imgs = imgs.to(device)
            targets = lengths_from_transcriptions(list(txts)).to(device)
            logits = net(imgs, return_feats=True)[-1]
            loss = criterion(logits, targets)
            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
            running += loss.item()

        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, txts in val_loader:
                imgs = imgs.to(device)
                targets = lengths_from_transcriptions(list(txts)).to(device)
                logits = net(imgs, return_feats=True)[-1]
                preds = logits.argmax(1)
                correct += (preds == targets).sum().item()
                total += targets.numel()
        acc = correct / max(1, total)
        
        htr_correct = 0
        htr_total = 0
        with torch.no_grad():
            for imgs, txts, _ in htr_val_loader:
                imgs = imgs.to(device)
                targets = lengths_from_transcriptions(list(txts)).to(device)
                logits = net(imgs, return_feats=True)[-1]
                preds = logits.argmax(1)
                htr_correct += (preds == targets).sum().item()
                htr_total += targets.numel()
        htr_acc = htr_correct / max(1, htr_total)


        if acc > best_acc:
            best_acc = acc
            if args.save_model:
                torch.save(net.state_dict(), save_path)
        print(f"Epoch {epoch+1}/{args.epochs} - loss {running/len(train_loader):.4f} - val acc {acc:.4f} - htr val acc {htr_acc:.4f}")

    if args.save_model:
        json_path = save_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump({"val_accuracy": best_acc}, f)
    print(f"Best accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
