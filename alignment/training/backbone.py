from __future__ import annotations
import os
from pathlib import Path
from itertools import cycle
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from htr_base.utils.htr_dataset import HTRDataset, PretrainingHTRDataset
from htr_base.models import HTRNet
from alignment.ctc_utils import encode_for_ctc
from alignment.losses import _ctc_loss_fn
from htr_base.utils.vocab import load_vocab
from htr_base.utils.metrics import word_silhouette_score
from alignment.features import harvest_backbone_features

cfg_file = Path(__file__).resolve().parents[1] / "config.yaml"
cfg = OmegaConf.load(cfg_file)

# Ensure CUDA_VISIBLE_DEVICES matches the configured GPU index
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
if str(cfg.device).startswith("cuda"):
    cfg.device = f"cuda:{cfg.gpu_id}"

def _assert_finite(t: torch.Tensor, where: str) -> None:
    assert torch.isfinite(t).all(), f"Non-finite values in {where}"


def _assert_grad_finite(model: nn.Module, name: str) -> None:
    assert all(
        p.grad is None or torch.isfinite(p.grad).all()
        for p in model.parameters()
    ), f"Gradient explosion in {name}"


def maybe_load_backbone(backbone: HTRNet, cfg=cfg) -> None:
    """Load pretrained backbone weights if ``cfg.load_pretrained_backbone``."""
    if getattr(cfg, "load_pretrained_backbone", False):
        path = cfg.pretrained_backbone_path
        state = torch.load(path, map_location=cfg.device)
        backbone.load_state_dict(state)
        print(f"[Init] loaded pretrained backbone from {path}")


def refine_visual_backbone(
    dataset: HTRDataset,
    backbone: HTRNet,
    num_epochs: int = cfg.refine_epochs,
    *,
    batch_size: int = cfg.refine_batch_size,
    lr: float = cfg.refine_lr,
    main_weight: float = cfg.refine_main_weight,
    aux_weight: float = cfg.refine_aux_weight,
    pretrain_ds: PretrainingHTRDataset | None = None,
    syn_batch_ratio: float = cfg.syn_batch_ratio,
) -> None:
    """Fine‑tune *backbone* only on words already aligned to external words."""
    print(f"[Refine] epochs={num_epochs}  batch_size={batch_size}  lr={lr}")
    device = next(backbone.parameters()).device
    backbone.train().to(device)
    c2i, _ = load_vocab()
    assert dataset.aligned.ndim == 1 and len(dataset) == len(dataset.aligned), (
        "Dataset alignment flags vector is malformed."
    )

    aligned_indices = (dataset.aligned != -1).nonzero(as_tuple=True)[0]
    subset = torch.utils.data.Subset(dataset, aligned_indices.tolist())

    if len(aligned_indices) == 0 and (pretrain_ds is None or syn_batch_ratio <= 0):
        print("[Refine] no pre-aligned samples found – aborting.")
        return

    syn_bs = int(batch_size * syn_batch_ratio) if pretrain_ds is not None else 0
    gt_bs = batch_size - syn_bs

    if pretrain_ds is not None and syn_bs > 0 and gt_bs > 0:
        gt_loader = DataLoader(
            subset,
            batch_size=gt_bs,
            shuffle=True,
            num_workers=2,
            pin_memory=(device.type == "cuda"),
        )
        pretrain_loader = DataLoader(
            pretrain_ds,
            batch_size=syn_bs,
            shuffle=True,
            num_workers=2,
            pin_memory=(device.type == "cuda"),
        )
        pre_iter = cycle(pretrain_loader)
        pretrain_only = False
    elif pretrain_ds is not None and syn_bs > 0 and gt_bs <= 0:
        gt_loader = DataLoader(
            pretrain_ds,
            batch_size=syn_bs,
            shuffle=True,
            num_workers=2,
            pin_memory=(device.type == "cuda"),
        )
        pre_iter = None
        pretrain_only = True
    else:
        gt_loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=(device.type == "cuda"),
        )
        pre_iter = None
        pretrain_only = False

    optimizer = optim.AdamW(backbone.parameters(), lr=lr)
    for epoch in range(1, num_epochs + 1):
        epoch_loss: float = 0.0
        effective_batches = 0
        for batch in gt_loader:
            if pretrain_only:
                imgs, trans = batch
                words = list(trans)
            else:
                imgs, _, aligned = batch
                words = [f" {dataset.external_words[i]} " for i in aligned.tolist()]

            imgs = imgs.to(device)

            if pre_iter is not None:
                imgs_syn, trans_syn = next(pre_iter)
                imgs = torch.cat([imgs, imgs_syn.to(device)], dim=0)
                words.extend(list(trans_syn))

            _assert_finite(imgs, "images")

            out = backbone(imgs, return_feats=False)
            if not isinstance(out, (tuple, list)) or len(out) < 2:
                raise RuntimeError("Expected network.forward() → (main, aux, …)")
            main_logits, aux_logits = out[:2]

            T, K, _ = main_logits.shape
            assert main_logits.shape[2] == len(c2i) + 1, "CTC class dimension mismatch"

            targets, tgt_lens = encode_for_ctc(words, c2i, device="cpu")

            inp_lens = torch.full((K,), T, dtype=torch.int32, device=device)
            loss_main = _ctc_loss_fn(main_logits, targets, inp_lens, tgt_lens)
            loss_aux = _ctc_loss_fn(aux_logits, targets, inp_lens, tgt_lens)
            loss = main_weight * loss_main + aux_weight * loss_aux
            _assert_finite(loss, "loss")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            _assert_grad_finite(backbone, "backbone")
            optimizer.step()
            epoch_loss += loss.item()
            effective_batches += 1
    backbone.eval()
    if len(aligned_indices) > 1:
        with torch.no_grad():
            feats, _ = harvest_backbone_features(subset, backbone, device=device)
        words = [dataset.external_words[dataset.aligned[i].item()] for i in aligned_indices.tolist()]
        score = word_silhouette_score(feats, words)
        print(f"[Refine] silhouette score: {score:.4f}")
    print("[Refine] finished.")

