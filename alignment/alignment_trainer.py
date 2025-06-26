from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from htr_base.utils.htr_dataset import HTRDataset
from htr_base.models import HTRNet
from alignment.losses import ProjectionLoss
from alignment.ctc_utils import encode_for_ctc
from alignment.losses import _ctc_loss_fn
from alignment.alignment_utilities import (
    align_more_instances,
    harvest_backbone_features,
    print_dataset_stats,
)
from htr_base.utils.metrics import word_silhouette_score
from htr_base.utils.transforms import aug_transforms
from htr_base.utils.vocab import load_vocab
from omegaconf import OmegaConf

from contextlib import contextmanager


class _Tee:
    """Write to multiple streams simultaneously."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


@contextmanager
def tee_output(path: str = "results.txt"):
    """Duplicate stdout to *path* while the context is active."""

    original = sys.stdout
    with open(path, "w") as f:
        sys.stdout = _Tee(original, f)
        try:
            yield
        finally:
            sys.stdout = original

# --------------------------------------------------------------------------- #
#                           Hyperparameter defaults                            #
# --------------------------------------------------------------------------- #
# Functions read defaults from the YAML configuration loaded at import time.
cfg_file = Path(__file__).with_name("config.yaml")
cfg = OmegaConf.load(cfg_file)

# Ensure CUDA_VISIBLE_DEVICES matches the configured GPU index
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
if str(cfg.device).startswith("cuda"):
    cfg.device = f"cuda:{cfg.gpu_id}"

# --------------------------------------------------------------------------- #
#                               Helper utilities                              #
# --------------------------------------------------------------------------- #
def _build_vocab_dicts(_: HTRDataset | None = None) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Return the project vocabulary loaded from disk."""
    return load_vocab()


def maybe_load_pretrained(backbone: HTRNet, device: torch.device) -> None:
    """Load weights into ``backbone`` if configured to do so."""
    if cfg.load_pretrained_backbone:
        state = torch.load(cfg.pretrained_path, map_location=device)
        backbone.load_state_dict(state, strict=False)


# --------------------------------------------------------------------------- #
#                            Main refinement routine                          #
# --------------------------------------------------------------------------- #
def optimise_backbone(
    dataset: HTRDataset,
    backbone: HTRNet,
    num_epochs: int = cfg.refine_epochs,
    *,
    batch_size: int = cfg.refine_batch_size,
    lr: float = cfg.refine_lr,
    main_weight: float = cfg.refine_main_weight,
    aux_weight: float = cfg.refine_aux_weight,
    proj_weight: float = cfg.proj_weight,
) -> None:
    """Optimise ``backbone`` using ProjectionLoss on all samples and CTC on aligned ones."""
    print(f"[Refine] epochs={num_epochs}  batch_size={batch_size}  lr={lr}")
    device = next(backbone.parameters()).device
    backbone.train().to(device)
    # Build CTC mapping once.
    c2i, _ = _build_vocab_dicts(dataset)

    # Mini-batches drawn randomly from the dataset. We keep the
    # number of workers low to make CI execution lighter.
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,            # keep CI simple
        pin_memory=(device.type == "cuda"),
    )
    optimizer = optim.AdamW(backbone.parameters(), lr=lr)

    criterion_proj = ProjectionLoss().to(device)
    word_embs = dataset.external_word_embeddings.to(device)
    probs = getattr(dataset, "external_word_probs", None)
    if probs is None or len(probs) == 0:
        word_probs = torch.full((word_embs.size(0),), 1.0 / word_embs.size(0), device=device)
    else:
        word_probs = torch.tensor(probs, dtype=torch.float32, device=device)
    # ------------------------------------------------------------------
    #                     Main training loop
    # ------------------------------------------------------------------
    for epoch in range(1, num_epochs + 1):
        epoch_loss: float = 0.0
        effective_batches = 0
        for imgs, _, aligned in dataloader:  # we ignore the transcription string
            imgs = imgs.to(device)
            # ---- forward pass -------------------------------------------------
            out = backbone(imgs, return_feats=True)
            if not isinstance(out, (tuple, list)):
                raise RuntimeError("Expected network.forward() → tuple")
            if len(out) == 3:
                main_logits, aux_logits, feats = out
            elif len(out) == 2:
                main_logits, feats = out
                aux_logits = None
            else:
                raise RuntimeError("Unexpected number of outputs from network")
            feats = torch.nan_to_num(feats)

            T, K, _ = main_logits.shape
            aligned_mask = aligned != -1
            if aligned_mask.any():
                # Convert the pseudo-labelled words to CTC targets. We wrap them
                # with spaces so no changes are persisted in ``dataset.external_words``.
                words = [f" {dataset.external_words[i]} " for i in aligned[aligned_mask].tolist()]
                targets, tgt_lens = encode_for_ctc(words, c2i, device="cpu")
                inp_lens = torch.full((aligned_mask.sum().item(),), T, dtype=torch.int32, device=device)
                loss_main = _ctc_loss_fn(main_logits[:, aligned_mask], targets, inp_lens, tgt_lens)
                if aux_logits is not None:
                    loss_aux = _ctc_loss_fn(aux_logits[:, aligned_mask], targets, inp_lens, tgt_lens)
                else:
                    loss_aux = torch.tensor(0.0, device=main_logits.device)
            else:
                loss_main = torch.tensor(0.0, device=main_logits.device)
                loss_aux = torch.tensor(0.0, device=main_logits.device)

            # Projection loss is computed on the full batch, while the CTC terms
            # use only the subset of aligned samples.
            loss_proj = criterion_proj(feats, word_embs, aligned.to(device), word_probs)
            loss = (
                main_weight * loss_main
                + aux_weight * loss_aux
                + proj_weight * loss_proj
            )
            # ── optimisation step ──────────────────────────────────────
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            effective_batches += 1
        # if effective_batches:
        #     avg_loss = epoch_loss / effective_batches
        #     print(f"Epoch {epoch:03d}/{num_epochs} – avg loss: {avg_loss:.4f}")
        # else:
        #     print(f"Epoch {epoch:03d}/{num_epochs} – no aligned batch encountered")
    backbone.eval()
    aligned_indices = (dataset.aligned != -1).nonzero(as_tuple=True)[0]
    if aligned_indices.numel() > 0:
        subset_feats = torch.utils.data.Subset(dataset, aligned_indices.tolist())
        with torch.no_grad():
            feats, _ = harvest_backbone_features(subset_feats, backbone, device=device)
        feats = torch.nan_to_num(feats)
        words = [dataset.external_words[dataset.aligned[i].item()] for i in aligned_indices.tolist()]
        score = word_silhouette_score(feats, words)
        print(f"[Refine] silhouette score: {score:.4f}")
    print("[Refine] finished.")

# File: alignment/alignment_trainer.py



def alternating_refinement(
    dataset: HTRDataset,
    backbone: HTRNet,
    *,
    rounds: int = cfg.alt_rounds,
    backbone_epochs: int = cfg.refine_epochs,
    refine_kwargs: dict | None = None,
    align_kwargs: dict | None = None,
) -> None:
    """Repeatedly optimise ``backbone`` and align more instances.

    The function alternates between fine‑tuning ``backbone`` on currently
    aligned samples and running :func:`align_more_instances` to pseudo‑label
    additional data.  The cycle continues until ``dataset.aligned`` no longer
    contains ``-1`` entries.
    """

    # Show basic statistics about the dataset before refinement starts.
    print_dataset_stats(dataset)
    # Optionally load pretrained weights if configured in ``cfg``.
    maybe_load_pretrained(backbone, torch.device(cfg.device))

    if refine_kwargs is None:
        refine_kwargs = {}
    if align_kwargs is None:
        align_kwargs = {}
    # Default parameters controlling the alignment step.
    align_kwargs.setdefault("batch_size", cfg.align_batch_size)
    align_kwargs.setdefault("device", cfg.align_device)
    align_kwargs.setdefault("reg", cfg.align_reg)
    align_kwargs.setdefault("unbalanced", cfg.align_unbalanced)
    align_kwargs.setdefault("reg_m", cfg.align_reg_m)
    align_kwargs.setdefault("k", cfg.align_k)
    align_kwargs.setdefault("agree_threshold", cfg.agree_threshold)

    # Repeat optimisation and alignment until every sample gets pseudo-labelled.
    while (dataset.aligned == -1).any():
        for r in range(rounds):
            print(f"[Round {r + 1}/{rounds}] Optimising backbone...")
            if backbone_epochs > 0:
                optimise_backbone(
                    dataset,
                    backbone,
                    num_epochs=backbone_epochs,
                    **refine_kwargs,
                )

        print("[Cycle] Aligning more instances...")
        # Use current backbone features to update pseudo-labels via OT.
        align_more_instances(dataset, backbone, [nn.Identity()], **align_kwargs)


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
        n_aligned = cfg.n_aligned   # how many images to mark as aligned (≈ training signal)

    dataset = HTRDataset(
        str(gw_folder),
        subset="all",
        fixed_size=(64, 256),
        transforms=aug_transforms,
        config=DummyCfg(),
    )

    arch = SimpleNamespace(
        cnn_cfg=[[2, 64], "M", [3, 128], "M", [2, 256]],
        head_type="both",
        rnn_type="gru",
        rnn_layers=3,
        rnn_hidden_size=256,
        flattening="maxpool",
        stn=False,
        feat_dim=512,
        feat_pool=cfg.feat_pool,
    )
    backbone = HTRNet(arch, nclasses=len(dataset.character_classes) + 1)

    with tee_output("results.txt"):
        alternating_refinement(dataset, backbone)

