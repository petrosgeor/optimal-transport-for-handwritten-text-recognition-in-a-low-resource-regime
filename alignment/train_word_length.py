"""Train a ResNet-18 model to predict word lengths from images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from htr_base.utils.htr_dataset import PretrainingHTRDataset
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
    """Construct a ResNet-18 for single-channel images with 20 outputs."""

    net = models.resnet18(weights=None)
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    in_feats = net.fc.in_features
    net.fc = nn.Linear(in_feats, 20)

    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0)
    return net


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    p = argparse.ArgumentParser(description="Train word-length predictor")
    p.add_argument("list_file", help="Path to 90k synthetic list file")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-size", type=int, default=50000)
    p.add_argument("--val-size", type=int, default=10000)
    p.add_argument(
        "--save-path",
        type=str,
        default="htr_base/saved_models/length_resnet18.pt",
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

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=3, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=3, pin_memory=True
    )

    net = build_resnet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    best_acc = 0.0
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        net.train()
        running = 0.0
        for imgs, txts in train_loader:
            imgs = imgs.to(device)
            targets = lengths_from_transcriptions(list(txts)).to(device)
            logits = net(imgs)
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
                logits = net(imgs)
                preds = logits.argmax(1)
                correct += (preds == targets).sum().item()
                total += targets.numel()
        acc = correct / max(1, total)
        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), save_path)
        print(f"Epoch {epoch+1}/{args.epochs} - loss {running/len(train_loader):.4f} - val acc {acc:.4f}")

    json_path = save_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump({"val_accuracy": best_acc}, f)
    print(f"Best accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
