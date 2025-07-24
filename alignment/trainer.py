from __future__ import annotations
import os
import sys
from pathlib import Path
from omegaconf import OmegaConf

# --------------------------------------------------------------------------- #
#                           Hyperparameter defaults                            #
# --------------------------------------------------------------------------- #
# Functions read defaults from the YAML configuration loaded at import time.

cfg_file = Path(__file__).parent / "alignment_configs" / "trainer_config.yaml"
cfg = OmegaConf.load(cfg_file)
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)

from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import cycle
from torch.utils.data import DataLoader, TensorDataset
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from htr_base.utils.htr_dataset import (
    HTRDataset, PretrainingHTRDataset, FusedHTRDataset
)
from htr_base.models import HTRNet, Projector
from alignment.losses import ProjectionLoss, SoftContrastiveLoss
from alignment.ctc_utils import encode_for_ctc
from alignment.losses import _ctc_loss_fn
from alignment.alignment_utilities import (
    align_more_instances,
    harvest_backbone_features,
)
from alignment.eval import compute_cer
from alignment.plot import (
    plot_tsne_embeddings,
    plot_projector_tsne,
    plot_pretrained_backbone_tsne
)
from htr_base.utils.transforms import aug_transforms
from htr_base.utils.vocab import load_vocab
from htr_base.utils import build_phoc_description
from omegaconf import OmegaConf

def _assert_finite(t: torch.Tensor, where: str) -> None:
    """Assert that ``t`` contains no ``NaN`` or ``Inf`` values.

    Args:
        t (torch.Tensor): Tensor to validate.
        where (str): Human readable location used in the assertion message.

    Returns:
        None
    """
    assert torch.isfinite(t).all(), f"Non-finite values in {where}"

def _assert_grad_finite(model: nn.Module, name: str) -> None:
    """Ensure all gradients in ``model`` are finite.

    Args:
        model (nn.Module): Model whose parameters were backpropagated.
        name (str): Identifier used in the error message.

    Returns:
        None
    """
    assert all(
        p.grad is None or torch.isfinite(p.grad).all()
        for p in model.parameters()
    ), f"Gradient explosion in {name}"

def _shuffle_batch(images: torch.Tensor, words: List[str]) -> Tuple[torch.Tensor, List[str]]:
    """Randomly shuffle a mini-batch of images and transcriptions together.

    Args:
        images (torch.Tensor): Tensor of images in the batch.
        words (List[str]): Transcriptions corresponding to ``images``.

    Returns:
        Tuple[torch.Tensor, List[str]]: Shuffled images and transcriptions.
    """
    assert len(images) == len(words)
    perm = torch.randperm(len(words), device=images.device)
    return images[perm], [words[i] for i in perm.tolist()]


def log_pseudo_labels(
    new_indices: torch.Tensor,
    dataset: HTRDataset,
    round_idx: int,
    out_dir: str = "results",
) -> None:
    """
    Save pseudo‑labeled examples for the current refinement round.

    The output file has **three tab‑separated columns**:
        1) Dataset index of the sample (integer)
        2) Predicted transcription (pseudo‑label)
        3) Ground‑truth transcription (original label)

    Parameters
    ----------
    new_indices : torch.Tensor
        1‑D tensor with the positions in `dataset` that were newly pseudo‑labeled.
    dataset : HTRDataset
        Must expose `unique_words`, `transcriptions` and `aligned` as in this repo.
    round_idx : int
        Index of the refinement cycle (used to name the log file).
    out_dir : str, default "results"
        Directory where the file `pseudo_labels_round_<round_idx>.txt` is saved.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    for idx in new_indices.tolist():
        word_id = int(dataset.aligned[idx])
        if word_id == -1:        # skip if still unaligned (safety check)
            continue
        pred_word = dataset.unique_words[word_id]
        true_word = dataset.transcriptions[idx]
        rows.append(f"{idx}\t{pred_word}\t{true_word}")

    file_path = out_path / f"pseudo_labels_round_{round_idx}.txt"
    with open(file_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(rows))




# PHOC configuration defaults
PHOC_WEIGHT = float(cfg.get("phoc_loss_weight", 0.1))
ENABLE_PHOC = bool(cfg.get("enable_phoc", False))
PHOC_LEVELS = tuple(cfg.get("phoc_levels", (1, 2, 3, 4)))
SUPERVISED_WEIGHT = float(cfg.get("supervised_weight", 1.0))

# Soft-contrastive configuration defaults
CONTRASTIVE_ENABLE = bool(cfg.get("contrastive_enable", False))
CONTRASTIVE_WEIGHT = float(cfg.get("contrastive_weight", 0.0))
CONTRASTIVE_TAU = float(cfg.get("contrastive_tau", 0.07))
CONTRASTIVE_TEXT_T = float(cfg.get("contrastive_text_T", 1.0))

# Ensure CUDA_VISIBLE_DEVICES matches the configured GPU index
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
if str(cfg.device).startswith("cuda"):
    cfg.device = "cuda:0"


def maybe_load_backbone(backbone: HTRNet, cfg) -> None:
    """Load pretrained backbone weights if ``cfg.load_pretrained_backbone``."""
    if getattr(cfg, "load_pretrained_backbone", False):
        path = cfg.pretrained_backbone_path
        # Load to CPU first, then the model will be moved to the correct
        # device later in the training pipeline.
        state = torch.load(path, map_location='cpu')
        backbone.load_state_dict(state)
        print(f"[Init] loaded pretrained backbone from {path}")

# --------------------------------------------------------------------------- #
#                            Main refinement routine                          #
# --------------------------------------------------------------------------- #
def refine_visual_backbone(
    dataset: FusedHTRDataset,
    backbone: nn.Module,
    *,
    batch_size: int,
    epochs: int,
    device: torch.device,
) -> None:
    """Refine ``backbone`` using only the aligned portion of ``dataset``.

    Only instances where ``dataset.aligned`` is not ``-1`` are used for
    training. Each epoch iterates over a ``DataLoader`` built from these
    aligned samples.

    Args:
        dataset (FusedHTRDataset): Combined dataset with alignment flags.
        backbone (nn.Module): Model to refine.
        batch_size (int): Mini-batch size.
        epochs (int): Number of training epochs.
        device (torch.device): Target device for training.
        logger (Logger | None): Optional progress logger.

    Returns:
        None
    """
    device = torch.device(device)
    assert device.type == "cuda", "Backbone is not on a CUDA device"
    backbone.train().to(device)
    c2i, _ = load_vocab()
    assert dataset.aligned.ndim == 1 and len(dataset) == len(dataset.aligned), "Dataset alignment flags vector is malformed."

    keep_mask = dataset.aligned != -1
    sub_ds = torch.utils.data.Subset(dataset, torch.nonzero(keep_mask, as_tuple=True)[0])
    if len(sub_ds) == 0:
        return

    loader = DataLoader(
        sub_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.projector_workers if hasattr(cfg, "projector_workers") else 0,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = optim.AdamW(backbone.parameters(), lr=cfg.refine_lr)
    for _ in range(epochs):
        for imgs, words, _ in loader:
            imgs = imgs.to(device)

            _assert_finite(imgs, "images")

            out = backbone(imgs)
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                main_logits, aux_logits = out[:2]
            else:
                main_logits = aux_logits = out

            T, B, _ = main_logits.shape
            assert main_logits.shape[2] == len(c2i) + 1, "CTC class dimension mismatch"

            targets, tgt_lens = encode_for_ctc(list(words), c2i, device="cpu")
            inp_lens = torch.full((B,), T, dtype=torch.int32, device=device)

            loss_main = _ctc_loss_fn(main_logits, targets, inp_lens, tgt_lens)
            loss_aux = _ctc_loss_fn(aux_logits, targets, inp_lens, tgt_lens)

            if ENABLE_PHOC and backbone.phoc_head is not None:
                phoc_logits = out[-1]
                phoc_targets = build_phoc_description(list(words), c2i, levels=PHOC_LEVELS).float().to(device)
                loss_phoc = F.binary_cross_entropy_with_logits(phoc_logits, phoc_targets)
            else:
                loss_phoc = torch.tensor(0.0, device=device)

            loss = (
                cfg.refine_main_weight * loss_main
                + cfg.refine_aux_weight * loss_aux
                + PHOC_WEIGHT * loss_phoc
            )
            _assert_finite(loss, "loss")

            # Optimization step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            _assert_grad_finite(backbone, "backbone")
            optimizer.step()
    
    plot_tsne_embeddings(dataset=dataset, backbone=backbone, save_path='results/figures/tsne_backbone_contrastive.png', device=device)
    # print('the backbone TSNE plot is saved')
    backbone.eval()

# File: alignment/trainer.py


def train_projector(  # pylint: disable=too-many-arguments
    dataset: "FusedHTRDataset | HTRDataset",
    backbone: "HTRNet",
    projector: nn.Module | List[nn.Module],
    num_epochs: int = cfg.projector_epochs,
    batch_size: int = cfg.projector_batch_size,
    lr: float = cfg.projector_lr,
    num_workers: int = cfg.projector_workers,
    weight_decay: float = cfg.projector_weight_decay,
    device: torch.device | str = cfg.device,
    plot_tsne: bool = cfg.plot_tsne,
) -> None:
    """
    Trains a projector network to map backbone features to an embedding space.

    This function freezes the ``backbone``, harvests image descriptors for the
    entire dataset, and then trains the ``projector`` using a combination of an
    unsupervised Optimal Transport (OT) loss and a supervised MSE loss on
    pre-aligned samples. The ``dataset`` is expected to be a
    ``FusedHTRDataset`` combining real and synthetic words, but a plain
    ``HTRDataset`` is still accepted for backward compatibility. The projector
    can be a single module or a list of modules for ensemble training.

    Args:
        dataset: A ``FusedHTRDataset`` combining real & synthetic words.
            (Falls back to a plain ``HTRDataset`` for legacy calls.)
        backbone: The HTRNet model (frozen) to extract features from.
        projector: The projector network(s) to be trained.
        num_epochs: The number of training epochs.
        batch_size: The batch size for training the projector.
        lr: The learning rate for the AdamW optimizer.
        num_workers: The number of workers for the DataLoader.
        weight_decay: The weight decay for the optimizer.
        device: The device to run the training on.
        plot_tsne: A flag to enable or disable t-SNE plotting of embeddings.
    """
    # ---------------------------------------------------------------- setup
    device = torch.device(device)
    assert device.type == "cuda", "Projector training must be on a CUDA device"
    backbone = backbone.to(device).eval()          # freeze visual encoder
    projs = projector if isinstance(projector, (list, tuple)) else [projector]
    projs = [p.to(device).train() for p in projs]
    
    word_embs_cpu = dataset.unique_word_embeddings
    if word_embs_cpu is None:
        raise RuntimeError(
            "FATAL: dataset.unique_word_embeddings is required but was not found."
        )
        
    # Target probability for each unique word – use uniform if absent
    # --- THIS BLOCK IS NOW FIXED ---
    probs_attr = getattr(dataset, "unique_word_probs", None)
    if probs_attr is not None and len(probs_attr) > 0:
        if isinstance(probs_attr, list):
            word_probs_cpu = torch.tensor(probs_attr, dtype=torch.float)
        else: # It's already a tensor
            word_probs_cpu = probs_attr.float()
    else:
        v = word_embs_cpu.size(0)
        word_probs_cpu = torch.full((v,), 1.0 / v)
        
    word_embs = word_embs_cpu.to(device)
    word_probs = word_probs_cpu.to(device)
    if word_probs.numel() != word_embs.size(0):
        pad = word_embs.size(0) - word_probs.numel()
        word_probs = torch.cat([word_probs, torch.zeros(pad, device=device)])

    if plot_tsne:
        plot_tsne_embeddings(
            dataset,
            backbone=backbone,
            save_path='tests/figures/tsne_backbone.png',
            device=device,
        )

    # ---------------------------------------------------------------- 1. Harvest descriptors for the whole dataset
    # Augmentations are temporarily disabled inside ``harvest_backbone_features`` to get stable descriptors.
    feats_all, aligned_all = harvest_backbone_features(
        dataset,
        backbone,
        batch_size=64,
        num_workers=num_workers,
        device=device,
    )
    if hasattr(dataset, "_is_real"):
        is_real_all = dataset._is_real.clone()
    else:
        is_real_all = torch.ones_like(aligned_all, dtype=torch.bool)
    assert feats_all.shape[1] == backbone.feat_dim, \
        "Descriptor dimension mismatch after harvesting features."
    
    # ---------------------------------------------------------------- 2. Create a new DataLoader for projector training
    # This loader will shuffle the collected features for effective training.
    proj_loader = DataLoader(
        TensorDataset(feats_all, aligned_all, is_real_all),
        batch_size=batch_size,
        shuffle=True, # Shuffle is True here for training
        pin_memory=(device.type == "cuda"),
    )

    # ---------------------------------------------------------------- 3. Optimiser + loss
    criterion = ProjectionLoss(supervised_weight=SUPERVISED_WEIGHT).to(device)
    # Iterate over each projector in the ensemble
    for idx, proj in enumerate(projs):
        optimiser = optim.AdamW(proj.parameters(), lr=lr, weight_decay=weight_decay)

        # Training loop for the current projector
        for epoch in range(1, num_epochs + 1):
            running_loss = 0.0
            num_batches = 0
            for feats_cpu, align_cpu, real_mask_cpu in proj_loader:
                feats = feats_cpu.to(device)
                align = align_cpu.to(device)
                real_m = real_mask_cpu.to(device)
                syn_m = ~real_m

                pred = proj(feats)

                loss_real = torch.tensor(0.0, device=device)
                if real_m.any():
                    loss_real = criterion(
                        pred[real_m],
                        word_embs,
                        align[real_m],
                        word_probs,
                    )

                loss_syn = torch.tensor(0.0, device=device)
                if syn_m.any():
                    target_syn = word_embs[align[syn_m]]
                    loss_syn = F.mse_loss(pred[syn_m], target_syn)

                loss = loss_real + loss_syn

                # Backpropagation and optimization
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(proj.parameters(), max_norm=1.0)

                optimiser.step()
                optimiser.zero_grad(set_to_none=True)
                running_loss += loss.item()
                num_batches += 1
            
            avg_loss = running_loss / num_batches if num_batches > 0 else 0
            # if epoch % 20 == 0:
            #     print(f"Projector {idx} - Epoch {epoch:03d}/{num_epochs} – avg loss: {avg_loss:.4f}")
                
        # Set projector to evaluation mode and generate t-SNE plot if enabled

        if plot_tsne and idx == 0:
            proj.eval()
            with torch.no_grad():
                proj_vecs = proj(feats_all.to(device)).cpu()
            plot_projector_tsne(
                proj_vecs,
                dataset,
                save_path=f'tests/figures/tsne_projections_{idx}.png'
            )



def alternating_refinement(
    dataset: FusedHTRDataset,
    backbone: HTRNet,
    projectors: List[nn.Module],
    *,
    rounds: int = cfg.alt_rounds,
    backbone_epochs: int = cfg.refine_epochs,
    projector_epochs: int = cfg.projector_epochs,
    refine_kwargs: dict | None = None,
    projector_kwargs: dict | None = None,
    align_kwargs: dict | None = None,
) -> None:
    """
    Performs an alternating training cycle between the backbone and projectors.

    This function expects a ``FusedHTRDataset`` (real + synthetic) where the
    synthetic portion is treated as pre-aligned. It implements a semi-supervised
    learning strategy where the ``backbone`` and ``projectors`` are trained in
    alternation. In each round,
    the backbone is first refined, then the projectors are trained. After a set
    number of rounds, more instances from the dataset are pseudo-labeled using
    Optimal Transport (OT) alignment. Initial seed alignments present in
    ``dataset.aligned`` are logged to ``pseudo_labels_round_0.txt`` before the
    refinement loop starts. The cycle continues as long as there are unaligned
    instances in the dataset. After each alignment step the Character Error Rate
    (CER) is printed for both the training and test sets.

    Args:
        dataset: The FusedHTRDataset to be used for training.
        backbone: The HTRNet model.
        projectors: A list of projector models.
        rounds: The number of backbone/projector training cycles per alignment pass.
        backbone_epochs: The number of epochs for each backbone refinement round.
        projector_epochs: The number of epochs for each projector training round.
        refine_kwargs: Additional keyword arguments for `refine_visual_backbone`.
        projector_kwargs: Additional keyword arguments for `train_projector`.
        align_kwargs: Additional keyword arguments for `align_more_instances`.
    """

    out_dir = "results"
    out_path = Path(out_dir)
    if out_path.is_dir():
        for f in out_path.glob("pseudo_labels_round_*.txt"):
            f.unlink()

    maybe_load_backbone(backbone, cfg)

    device = torch.device(cfg.device)
    backbone.to(device)
    projectors = [p.to(device) for p in projectors]
    
    assert isinstance(projectors, (list, tuple)) and len(projectors) > 0, \
        "Projectors must be a non-empty list or tuple."

    if refine_kwargs is None:
        refine_kwargs = {}
    if projector_kwargs is None:
        projector_kwargs = {}
    if align_kwargs is None:
        align_kwargs = {}
    # Set default alignment arguments from config
    align_kwargs.setdefault("batch_size", cfg.align_batch_size)
    align_kwargs.setdefault("device", cfg.align_device)
    align_kwargs.setdefault("k", cfg.align_k)
    align_kwargs.setdefault("agree_threshold", cfg.agree_threshold)
    align_kwargs.setdefault("metric", cfg.metric)

    real = dataset.real_ds
    test_dataset = HTRDataset(
        basefolder=real.basefolder,
        subset='test',
        fixed_size=real.fixed_size,
        character_classes=real.character_classes,
        config=real.config,
        two_views=False,
        word_prob_mode=real.word_prob_mode,
    )

    cycle_idx = 1
    initial_aligned = torch.nonzero(dataset.aligned != -1, as_tuple=True)[0]
    log_pseudo_labels(initial_aligned, dataset, 0, out_dir="results")
    # Main loop: continue as long as there are unaligned instances
    while (dataset.aligned == -1).any():
        for r in range(rounds):
            print(f"[Round {r + 1}/{rounds}] Refining backbone...")
            if backbone_epochs > 0:
                # Refine the visual backbone
                refine_visual_backbone(
                    dataset,
                    backbone,
                    batch_size=cfg.refine_batch_size,
                    epochs=backbone_epochs,
                    device=device,
                    **refine_kwargs,
                )

            # Freeze backbone and unfreeze projectors for training
            for param in backbone.parameters():
                param.requires_grad_(False)
            for proj in projectors:
                for param in proj.parameters():
                    param.requires_grad_(True)
            
            # Verify that exactly one module family has requires_grad=True
            backbone_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
            projectors_trainable = sum(p.numel() for proj in projectors for p in proj.parameters() if p.requires_grad)
            assert (backbone_trainable == 0 and projectors_trainable > 0) or \
                   (backbone_trainable > 0 and projectors_trainable == 0), \
                   "Exactly one module family (backbone or projectors) should be trainable."

            print(f"[Round {r + 1}/{rounds}] Training projector...")
            if projector_epochs > 0:
                # Temporarily handle unique_word_probs format for projector training
                _probs_backup = None
                if isinstance(getattr(dataset, "unique_word_probs", None), list):
                    _probs_backup = dataset.unique_word_probs
                    dataset.unique_word_probs = torch.tensor(
                        _probs_backup, dtype=torch.float
                    )

                # Train the projector(s)
                train_projector(
                    dataset,
                    backbone,
                    projectors,
                    num_epochs=projector_epochs,
                    **projector_kwargs,
                )

                # Restore original unique_word_probs format if it was changed
                if _probs_backup is not None:
                    dataset.unique_word_probs = _probs_backup

            # Unfreeze backbone and freeze projectors for next round or alignment
            for param in backbone.parameters():
                param.requires_grad_(True)
            for proj in projectors:
                for param in proj.parameters():
                    param.requires_grad_(False)

            # Verify that exactly one module family has requires_grad=True
            backbone_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
            projectors_trainable = sum(p.numel() for proj in projectors for p in proj.parameters() if p.requires_grad)
            assert (backbone_trainable > 0 and projectors_trainable == 0), \
                   "Exactly one module family (backbone or projectors) should be trainable."

        print("[Cycle] Aligning more instances...")
        assert (dataset.aligned != -1).sum() > 0, \
            "Cannot align more instances with zero seeds."
        # Save current alignment state before pseudo-labelling
        prev_aligned = dataset.aligned.clone()
        # Perform Optimal Transport alignment to pseudo-label more instances
        align_more_instances(dataset, backbone, projectors, **align_kwargs)
        changed = torch.nonzero(
            (prev_aligned == -1) & (dataset.aligned != -1), as_tuple=True
        )[0]
        log_pseudo_labels(changed, dataset, cycle_idx, out_dir="results")
        compute_cer(
            test_dataset,
            backbone,
            batch_size=cfg.eval_batch_size,
            device=cfg.device,
            k=4
        )

        # --- restrict CER to already-aligned samples -------------------
        aligned_idx = torch.nonzero(dataset.aligned != -1, as_tuple=True)[0]
        aligned_subset = torch.utils.data.Subset(dataset, aligned_idx.tolist())
        print('CER on the aligned subset: ')
        compute_cer(
            aligned_subset,
            backbone,
            batch_size=cfg.eval_batch_size,
            device=cfg.device,
            k=4
        )
        cycle_idx += 1



if __name__ == "__main__":
    """Run a *tiny* end‑to‑end refinement cycle to verify code execution."""
    from types import SimpleNamespace

    # ── 1. Dataset with 200 unique words and a handful of alignments ─────
    proj_root = Path(__file__).resolve().parents[1]
    
    ds_cfg = cfg.dataset
    basefolder = proj_root / ds_cfg.basefolder
    if not basefolder.exists():
        raise RuntimeError(
            f"Dataset folder {basefolder} not found – run the dataset preparation step "
            "before executing this dummy test."
        )

    real_ds = HTRDataset(
        basefolder=str(basefolder),
        subset=ds_cfg.subset,
        fixed_size=tuple(ds_cfg.fixed_size),
        transforms=aug_transforms,
        config=ds_cfg,
        two_views=ds_cfg.two_views,
        word_prob_mode=ds_cfg.word_prob_mode,
    )

    arch = SimpleNamespace(**cfg["architecture"])
    backbone = HTRNet(arch, nclasses=len(real_ds.character_classes) + 1)
    # maybe_load_backbone(backbone, cfg)
    projectors = [
        Projector(arch.feat_dim, real_ds.word_emb_dim, dropout=0.2)
        for _ in range(cfg.ensemble_size)
    ]

    syn_cfg = cfg.get("synthetic_dataset", None)
    pre_syn_ds = None
    if syn_cfg is not None:
        pre_syn_ds = PretrainingHTRDataset(
            list_file=syn_cfg.list_file,
            base_path=syn_cfg.base_path,
            n_random=syn_cfg.n_random,
            fixed_size=tuple(syn_cfg.fixed_size),
            preload_images=syn_cfg.get("preload_images", False),
            random_seed=syn_cfg.get("random_seed", 0),
        )

    dataset = FusedHTRDataset(real_ds, pre_syn_ds, n_aligned=ds_cfg.n_aligned)

    alternating_refinement(
        dataset=dataset,
        backbone=backbone,
        projectors=projectors,
    )
