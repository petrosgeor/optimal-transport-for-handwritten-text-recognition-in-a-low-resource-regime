"""Dataset inspection and feature extraction utilities."""

from pathlib import Path
from typing import Tuple
import torch
from omegaconf import OmegaConf

from htr_base.utils.htr_dataset import HTRDataset
from htr_base.models import HTRNet

cfg = OmegaConf.load("alignment/config.yaml")


def harvest_backbone_features(
    dataset: HTRDataset,
    backbone: HTRNet,
    *,
    batch_size: int = 64,
    num_workers: int = 0,
    device: torch.device | str = torch.device(cfg.device),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return image descriptors and alignment info for the whole dataset."""
    device = torch.device(device)
    backbone = backbone.to(device).eval()

    orig_transforms = getattr(dataset, "transforms", None)
    dataset.transforms = None

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    feats_buf: list[torch.Tensor] = []
    align_buf: list[torch.Tensor] = []
    with torch.no_grad():
        for imgs, _txt, aligned in loader:
            imgs = imgs.to(device)
            feats = backbone(imgs, return_feats=True)[-1]
            if feats.dim() != 2:
                raise RuntimeError(
                    f"Expected (B, feat_dim) descriptors, got {feats.shape}"
                )
            feats_buf.append(feats.cpu())
            align_buf.append(aligned.cpu())

    dataset.transforms = orig_transforms

    feats_all = torch.cat(feats_buf, dim=0)
    aligned_all = torch.cat(align_buf, dim=0)
    return feats_all, aligned_all


def print_dataset_stats(dataset: HTRDataset) -> None:
    """Print basic statistics about ``dataset``."""
    n_total = len(dataset)
    n_aligned = int((dataset.aligned != -1).sum()) if hasattr(dataset, "aligned") else 0
    vocab_size = len(getattr(dataset, "external_words", []))
    if hasattr(dataset, "is_in_dict"):
        in_vocab = int(dataset.is_in_dict.sum())
    else:
        in_vocab = 0
    pct_in_vocab = (in_vocab / n_total) * 100 if n_total > 0 else 0.0
    all_lower = all(t == t.lower() for t in dataset.transcriptions)
    ext_lower = all(w == w.lower() for w in getattr(dataset, "external_words", []))
    avg_len = sum(len(t.strip()) for t in dataset.transcriptions) / n_total if n_total > 0 else 0.0
    print(f"[Stats] dataset size: {n_total} samples")
    print(f"[Stats] {n_aligned} already aligned")
    print(f"[Stats] external vocabulary: {vocab_size} words")
    print(f"[Stats] {in_vocab}/{n_total} ({pct_in_vocab:.1f}%) samples in vocabulary")
    print(f"[Stats] transcriptions lowercase: {all_lower}")
    print(f"[Stats] external words lowercase: {ext_lower}")
    print(f"[Stats] average transcription length: {avg_len:.2f}")

