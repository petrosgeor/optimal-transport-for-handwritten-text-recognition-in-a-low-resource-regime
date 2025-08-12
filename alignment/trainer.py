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
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from htr_base.utils.htr_dataset import HTRDataset
from htr_base.models import HTRNet, Projector
from alignment.losses import ProjectionLoss, SoftContrastiveLoss
from alignment.ctc_utils import encode_for_ctc, ctc_target_probability
from alignment.losses import _ctc_loss_fn, _em_word_loss_for_batch, expected_phoc_from_responsibilities
from alignment.alignment_utilities import (
    harvest_backbone_features,
    compute_ot_responsibilities,
    IndexedSubset,
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

import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="sklearn.manifold._t_sne"
)
try:
    from requests.exceptions import RequestsDependencyWarning
    warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
except Exception:  # pragma: no cover
    pass


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


def _harden_aligned_rows(R: torch.Tensor, aligned: torch.Tensor) -> torch.Tensor:
    """Make responsibility rows 1‑hot for aligned real samples.

    Purpose:
        Enforce that for each index ``i`` with a known alignment ``aligned[i]=k``,
        the corresponding responsibility row becomes one‑hot: ``R[i,k]=1`` and
        zeros elsewhere. This ensures the unified EM loss reduces to the
        supervised NLL for aligned instances.

    Args:
        R (torch.Tensor): Responsibility matrix of shape ``(N, W)`` on CPU.
        aligned (torch.Tensor): Alignment indices of shape ``(N,)`` with ``-1``
            for unaligned and ``k`` in ``[0, W)`` for aligned.

    Returns:
        torch.Tensor: The same tensor instance ``R`` with aligned rows hardened
        in place. Returned for convenience/chaining.
    """
    if R is None:
        return R
    if R.ndim != 2:
        raise ValueError("R must be 2-D (N, W)")
    if aligned.ndim != 1 or aligned.size(0) != R.size(0):
        raise ValueError("aligned must be shape (N,) and match R rows")
    # clone to ensure it's a normal tensor even if created under inference_mode
    R = R.clone()
    with torch.no_grad():
        N, W = R.shape
        aligned = aligned.to(torch.long)
        for i in range(N):
            k = int(aligned[i].item())
            if k >= 0 and k < W:
                R[i].zero_()
                R[i, k] = 1.0
    return R


_TRUNCATED_METRIC_FILES: set[str] = set()




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
    dataset: HTRDataset,
    backbone: HTRNet,
    num_epochs: int = cfg.refine_epochs,
    *,
    batch_size: int = cfg.refine_batch_size,
    lr: float = cfg.refine_lr,
    main_weight: float = cfg.refine_main_weight,
    aux_weight: float = cfg.refine_aux_weight,
    phoc_weight: float = cfg.phoc_loss_weight,
    enable_phoc: bool = cfg.enable_phoc,
    phoc_levels: Tuple[int, ...] = tuple(cfg.phoc_levels),
    enable_contrastive: bool = CONTRASTIVE_ENABLE,
    contrastive_weight: float = CONTRASTIVE_WEIGHT,
    contrastive_tau: float = CONTRASTIVE_TAU,
    contrastive_text_T: float = CONTRASTIVE_TEXT_T,
    # --- NEW: optional responsibilities‑based EM loss over unaligned items ---
    projectors: List[nn.Module] | None = None,
    use_responsibilities: bool = True,
    em_weight: float = 0.2,
    em_topk: int = 5,
    resp_topk: int = 10,
    em_phoc_weight: float = float(cfg.get("em_phoc_weight", 0.25)),
) -> None:
    """Fine‑tune ``backbone`` on a real dataset using aligned and unaligned words.

    Purpose:
        Refine the visual backbone using only real data. Aligned items receive
        supervised CTC (and optional PHOC/contrastive) losses, while unaligned
        items contribute non‑differentiable EM word loss and optional EM‑PHOC.

    Args:
        dataset (HTRDataset): Training dataset with alignment information.
        backbone (HTRNet): Model to be refined.
        num_epochs (int): Number of optimisation epochs.
        batch_size (int): Mini-batch size.
        lr (float): Learning rate for the optimiser.
        main_weight (float): Scale for the main CTC loss.
        aux_weight (float): Scale for the auxiliary CTC loss.
        phoc_weight (float): Scale for the PHOC loss.
        enable_phoc (bool): Whether to include the PHOC loss.
        phoc_levels (Tuple[int, ...]): Levels for PHOC descriptors.
        enable_contrastive (bool): Use the SoftContrastiveLoss.
        contrastive_weight (float): Weight of the contrastive term.
        contrastive_tau (float): Temperature for descriptor similarities.
        contrastive_text_T (float): Temperature in edit-distance space.
        projectors (List[nn.Module] | None): Optional projector ensemble used to
            compute OT responsibilities.
        use_responsibilities (bool): Compute responsibilities for EM losses.
        em_weight (float): Weight of the EM word loss on unaligned items.
        em_topk (int): Number of top words considered in EM loss.
        resp_topk (int): Sparsity of responsibility matrix rows.
        em_phoc_weight (float): Weight of the EM‑PHOC loss when PHOC is enabled.

    Returns:
        None

    Additional behaviour:
        When ``use_responsibilities=True``, the function computes a matrix of
        per‑image responsibilities over the vocabulary once per call. If
        ``projectors`` are provided, it uses ``compute_ot_responsibilities`` with
        mean fusion across projectors and optional top‑K sparsification. If not,
        it warm‑starts using one‑hot rows for seeds and a unigram (or uniform)
        distribution for unaligned rows. The EM word loss evaluates
        ``ctc_target_probability`` for unaligned items, and EM‑PHOC trains a
        PHOC head against expected PHOC targets.
    """
    # Resolve runtime device from backbone parameters (tests monkeypatch .to())
    device = next(backbone.parameters()).device
    assert device.type == "cuda", "Backbone is not on a CUDA device"
    backbone.train().to(device)
    # Build CTC mapping once using the fixed vocabulary.
    c2i, _ = load_vocab()
    assert dataset.aligned.ndim == 1 and len(dataset) == len(dataset.aligned), "Dataset alignment flags vector is malformed."

    # --- E-step: build responsibilities once per call (optional) ---------
    R: torch.Tensor | None = None
    if use_responsibilities:
        with torch.inference_mode():
            try:
                if projectors and len(projectors) > 0 and hasattr(dataset, "unique_word_embeddings"):
                    R = compute_ot_responsibilities(
                        dataset,
                        backbone,
                        projectors,
                        batch_size=cfg.align_batch_size,
                        device=cfg.align_device,
                        reg=cfg.align_reg,
                        unbalanced=cfg.align_unbalanced,
                        reg_m=cfg.align_reg_m,
                        ensemble="mean",
                        topk=resp_topk,
                    )
                else:
                    # Warm‑start: one‑hot rows for seeds; unigram/uniform for others
                    V = len(getattr(dataset, "unique_words", []))
                    if V > 0:
                        R_np = np.zeros((len(dataset), V), dtype=np.float32)
                        probs = getattr(dataset, "unique_word_probs", None)
                        if probs is None or len(probs) != V:
                            base = np.full((V,), 1.0 / V, dtype=np.float32)
                        else:
                            p = np.asarray(probs, dtype=np.float64)
                            p = p / max(p.sum(), 1e-12)
                            base = p.astype(np.float32)
                        for i in range(len(dataset)):
                            k = int(dataset.aligned[i].item()) if hasattr(dataset.aligned, "item") else int(dataset.aligned[i])
                            if k != -1 and 0 <= k < V:
                                R_np[i, k] = 1.0
                            else:
                                row = base.copy()
                                if resp_topk is not None and resp_topk < V:
                                    idx = np.argpartition(row, -resp_topk)[-resp_topk:]
                                    mask = np.zeros(V, dtype=bool)
                                    mask[idx] = True
                                    row = np.where(mask, row, 0.0)
                                    s = row.sum()
                                    if s > 0:
                                        row /= s
                                R_np[i] = row
                        R = torch.from_numpy(R_np)
            except Exception:
                R = None
        if R is not None:
            # Harden rows for aligned items so EM reduces to supervised on those
            R = _harden_aligned_rows(R.detach().contiguous(), dataset.aligned.cpu())

    # Precompute PHOC for the vocabulary once (used by EM‑PHOC for real items)
    phoc_vocab = None
    if enable_phoc and use_responsibilities and getattr(dataset, "unique_words", None):
        try:
            vocab_words = [f" {w} " for w in dataset.unique_words]
            phoc_vocab = build_phoc_description(vocab_words, c2i, levels=phoc_levels)
            phoc_vocab = phoc_vocab.float().to(device)
        except Exception:
            phoc_vocab = None

    # Build real DataLoader over the full dataset, preserving original indices
    real_loader = DataLoader(
        IndexedSubset(dataset, list(range(len(dataset)))),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = optim.AdamW(backbone.parameters(), lr=lr)
    contr_fn = None
    if enable_contrastive:
        contr_fn = SoftContrastiveLoss(contrastive_tau, contrastive_text_T).to(device)

    # Training loop for backbone refinement
    for epoch in range(1, num_epochs + 1):
        epoch_loss: float = 0.0
        effective_batches = 0
        for imgs, txts, aligned_r, idx_r in real_loader:
            imgs = imgs.to(device)
            idx_r_list = [int(i) for i in idx_r]
            _assert_finite(imgs, "images")

            # 3) Forward pass through the backbone once
            need_feats = (
                (enable_phoc and backbone.phoc_head is not None)
                or enable_contrastive
            )
            out = backbone(imgs, return_feats=need_feats)
            if not isinstance(out, (tuple, list)) or len(out) < 2:
                raise RuntimeError("Expected network.forward() → (main, aux, …)")
            if need_feats:
                main_logits, aux_logits, feats = out[:3]
                phoc_logits = out[3] if enable_phoc and backbone.phoc_head is not None else None
            else:
                main_logits, aux_logits = out[:2]
                feats = None
                phoc_logits = None

            T, _, _ = main_logits.shape
            assert main_logits.shape[2] == len(c2i) + 1, "CTC class dimension mismatch"

            # 4) Supervised losses on aligned real items
            aligned_pos = (aligned_r != -1).nonzero(as_tuple=True)[0]
            loss_main = torch.tensor(0.0, device=device)
            loss_aux = torch.tensor(0.0, device=device)
            loss_phoc = torch.tensor(0.0, device=device)
            if aligned_pos.numel() > 0:
                words_aligned = [txts[i] for i in aligned_pos.tolist()]
                targets, tgt_lens = encode_for_ctc(words_aligned, c2i, device=device)
                inp_lens = torch.full((aligned_pos.numel(),), T, dtype=torch.int32, device=device)
                loss_main = _ctc_loss_fn(main_logits[:, aligned_pos, :], targets, inp_lens, tgt_lens)
                loss_aux = _ctc_loss_fn(aux_logits[:, aligned_pos, :], targets, inp_lens, tgt_lens)
                if enable_phoc and phoc_logits is not None:
                    phoc_t = build_phoc_description(words_aligned, c2i, levels=phoc_levels)
                    phoc_t = phoc_t.float().to(device)
                    loss_phoc = F.binary_cross_entropy_with_logits(phoc_logits[aligned_pos], phoc_t)

            loss_contr = torch.tensor(0.0, device=device)
            if enable_contrastive and feats is not None and aligned_pos.numel() > 1:
                words_aligned = [txts[i] for i in aligned_pos.tolist()]
                targets_aligned, tgt_lens_aligned = encode_for_ctc(words_aligned, c2i, device=device)
                loss_contr = contr_fn(feats[aligned_pos], targets_aligned, tgt_lens_aligned)
                _assert_finite(loss_contr, "contrastive loss")

            # 5) EM losses on unaligned items
            em_loss = torch.tensor(0.0, device=device)
            em_phoc_loss = torch.tensor(0.0, device=device)
            if use_responsibilities and em_weight > 0 and R is not None:
                unaligned_pos = (aligned_r == -1).nonzero(as_tuple=True)[0]
                if unaligned_pos.numel() > 0:
                    idx_u = [idx_r_list[j] for j in unaligned_pos.tolist()]
                    em_loss = _em_word_loss_for_batch(
                        main_logits[:, unaligned_pos, :],
                        batch_indices=idx_u,
                        R_full=R,
                        unique_words=dataset.unique_words,
                        c2i_map=c2i,
                        k_top=em_topk,
                    )
                    if enable_phoc and phoc_logits is not None and phoc_vocab is not None:
                        with torch.no_grad():
                            R_sel = R[torch.tensor(idx_u)].to(device).float()
                            exp_phoc = expected_phoc_from_responsibilities(R_sel, phoc_vocab)
                        em_phoc_loss = F.binary_cross_entropy_with_logits(phoc_logits[unaligned_pos], exp_phoc)

            loss = (
                main_weight * loss_main
                + aux_weight * loss_aux
                + phoc_weight * loss_phoc
                + contrastive_weight * loss_contr
                + em_weight * em_loss
                + em_phoc_weight * em_phoc_loss
            )
            _assert_finite(loss, "loss")

            # 6) Optimisation step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            _assert_grad_finite(backbone, "backbone")
            optimizer.step()
            epoch_loss += loss.item()
            effective_batches += 1
        # if effective_batches:
        #     avg_loss = epoch_loss / effective_batches
        #     print(f"Epoch {epoch:03d}/{num_epochs} – avg loss: {avg_loss:.4f}")
        # else:
        #     print(f"Epoch {epoch:03d}/{num_epochs} – no aligned batch encountered")
    
    plot_tsne_embeddings(dataset=dataset, backbone=backbone, save_path='results/figures/tsne_backbone_contrastive.png', device=device)
    # print('the backbone TSNE plot is saved')
    backbone.eval()

# File: alignment/trainer.py


def train_projector(  # pylint: disable=too-many-arguments
    dataset: "HTRDataset",
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

    This function freezes the `backbone`, harvests image descriptors for the entire
    dataset, and then trains the `projector` using a combination of an unsupervised
    Optimal Transport (OT) loss and a supervised MSE loss on pre-aligned samples.
    The projector can be a single module or a list of modules for ensemble training.

    Args:
        dataset: The HTRDataset containing the data.
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
    assert feats_all.shape[1] == backbone.feat_dim, \
        "Descriptor dimension mismatch after harvesting features."
    
    # ---------------------------------------------------------------- 2. Create a new DataLoader for projector training
    # This loader will shuffle the collected features for effective training.
    # Use multiple workers and pinned memory to better overlap host↔device
    # transfers with GPU compute. Keep workers persistent to avoid respawn cost.
    _pin = (device.type == "cuda")
    _workers = int(num_workers) if num_workers is not None else 0
    proj_loader = DataLoader(
        TensorDataset(feats_all, aligned_all),
        batch_size=batch_size,
        shuffle=True,  # Shuffle is True here for training
        pin_memory=_pin,
        num_workers=_workers,
        persistent_workers=(_workers > 0),
    )

    # ---------------------------------------------------------------- 3. Optimiser + loss
    criterion = ProjectionLoss(supervised_weight=SUPERVISED_WEIGHT).to(device)
    # Iterate over each projector in the ensemble
    for idx, proj in enumerate(projs):
        optimiser = optim.AdamW(proj.parameters(), lr=lr, weight_decay=weight_decay)

        # Training loop for the current projector
        for epoch in range(1, num_epochs + 1):
            # print('the epoch is: ', epoch)
            running_loss = 0.0
            num_batches = 0
            for feats_cpu, align_cpu in proj_loader:
                # Non-blocking copies overlap H2D transfers with compute
                feats = feats_cpu.to(device, non_blocking=_pin)
                align = align_cpu.to(device, non_blocking=_pin)
                
                # Forward pass through the projector
                pred = proj(feats)
                # Compute the projection loss
                loss = criterion.forward(pred, word_embs, align, word_probs)

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
    dataset: HTRDataset,
    backbone: HTRNet,
    projectors: List[nn.Module],
    *,
    rounds: int = cfg.alt_rounds,
    backbone_epochs: int = cfg.refine_epochs,
    projector_epochs: int = cfg.projector_epochs,
    refine_kwargs: dict | None = None,
    projector_kwargs: dict | None = None,
) -> None:
    """Alternating EM-only refinement: backbone then projectors, fixed rounds.

    Responsibilities for EM are computed inside ``refine_visual_backbone`` via
    ``compute_ot_responsibilities`` when ``projectors`` are provided. No
    pseudo‑labelling is performed here.
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
    assert isinstance(projectors, (list, tuple)) and len(projectors) > 0

    refine_kwargs = {} if refine_kwargs is None else dict(refine_kwargs)
    projector_kwargs = {} if projector_kwargs is None else dict(projector_kwargs)

    test_dataset = HTRDataset(
        basefolder=dataset.basefolder,
        subset='test',
        fixed_size=dataset.fixed_size,
        character_classes=dataset.character_classes,
        config=dataset.config,
        two_views=False,
    )

    for r in range(1, rounds + 1):
        print(f"[Round {r}/{rounds}] Refining backbone (EM only)…")
        if backbone_epochs > 0:
            refine_visual_backbone(
                dataset,
                backbone,
                num_epochs=backbone_epochs,
                projectors=projectors,
                **refine_kwargs,
            )

        for p in backbone.parameters():
            p.requires_grad_(False)
        for proj in projectors:
            for p in proj.parameters():
                p.requires_grad_(True)
        print(f"[Round {r}/{rounds}] Training projector…")
        if projector_epochs > 0:
            _probs_backup = None
            if isinstance(getattr(dataset, "unique_word_probs", None), list):
                _probs_backup = dataset.unique_word_probs
                dataset.unique_word_probs = torch.tensor(_probs_backup, dtype=torch.float)
            train_projector(
                dataset,
                backbone,
                projectors,
                num_epochs=projector_epochs,
                **projector_kwargs,
            )
            if _probs_backup is not None:
                dataset.unique_word_probs = _probs_backup

        for p in backbone.parameters():
            p.requires_grad_(True)
        for proj in projectors:
            for p in proj.parameters():
                p.requires_grad_(False)

        try:
            cer = compute_cer(
                test_dataset,
                backbone,
                batch_size=cfg.eval_batch_size,
                device=cfg.device,
                k=4,
            )
            print(f"[Round {r}/{rounds}] Test CER: {cer:.4f}")
            # Also compute CER on the training/val dataset for parity
            compute_cer(
                dataset,
                backbone,
                batch_size=cfg.eval_batch_size,
                device=cfg.device,
                k=4,
            )
        except Exception:
            pass

    # Optional final refinement when all samples are already aligned or rounds=0
    if rounds == 0 or not (dataset.aligned == -1).any():
        refine_visual_backbone(
            dataset,
            backbone,
            num_epochs=backbone_epochs,
            projectors=projectors,
            **refine_kwargs,
        )

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

    dataset = HTRDataset(
        basefolder=str(basefolder),
        subset=ds_cfg.subset,
        fixed_size=tuple(ds_cfg.fixed_size),
        transforms=aug_transforms,
        config=ds_cfg,
        two_views=ds_cfg.two_views,
    )

    arch = SimpleNamespace(**cfg["architecture"])
    backbone = HTRNet(arch, nclasses=len(dataset.character_classes) + 1)
    projectors = [
        Projector(arch.feat_dim, dataset.word_emb_dim, dropout=0.2)
        for _ in range(cfg.ensemble_size)
    ]

    alternating_refinement(
        dataset,
        backbone,
        projectors,
    )
