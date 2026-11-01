from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from omegaconf import OmegaConf
# --------------------------------------------------------------------------- #
#                           Hyperparameter defaults                            #
# --------------------------------------------------------------------------- #
# Functions read defaults from the YAML configuration loaded at import time.


def _resolve_trainer_config_path() -> Path:
    """Resolve the trainer config path from CLI or default.

    Args:
        None

    Returns:
        Path: Absolute path to the selected trainer YAML configuration.
    """
    repo_root = Path(__file__).resolve().parents[1]
    default_cfg = repo_root / "alignment" / "alignment_configs" / "trainer_config.yaml"

    cfg_candidate = default_cfg
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(add_help=True)
        parser.add_argument(
            "--config",
            type=str,
            default=None,
            help="Path to trainer YAML. Relative paths resolve from repo root.",
        )
        args, _ = parser.parse_known_args()
        if args.config:
            user_cfg = Path(args.config)
            cfg_candidate = user_cfg if user_cfg.is_absolute() else (repo_root / user_cfg)

    return cfg_candidate.resolve()


cfg_file = _resolve_trainer_config_path()
cfg = OmegaConf.load(cfg_file)
os.environ["ALIGNMENT_TRAINER_CONFIG"] = str(cfg_file)
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
if __name__ == "__main__":
    print(f"[trainer] Using config: {cfg_file}")

from typing import Iterator, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from htr_base.utils.htr_dataset import HTRDataset, PretrainingHTRDataset
from htr_base.models import HTRNet, Projector
from alignment.losses import ProjectionLoss
from alignment.utils.ctc import encode_for_ctc
from alignment.utils.checkpoints import maybe_load_backbone
from alignment.utils.reporting import log_pseudo_labels, log_round_metrics
from alignment.utils.checks import _assert_finite, _assert_grad_finite
from alignment.losses import _ctc_loss_fn
from alignment.alignment_utilities import (
    align_more_instances,
    harvest_backbone_features,
)
from alignment.eval import compute_cer, compute_wer
from htr_base.utils.transforms import aug_transforms
from htr_base.utils.vocab import load_vocab
from htr_base.utils import build_phoc_description
import copy
import random
import numpy as np

import warnings

from requests.exceptions import RequestsDependencyWarning

warnings.filterwarnings("ignore", category=RequestsDependencyWarning)


def _shuffle_batch(images: torch.Tensor, words: List[str]) -> Tuple[torch.Tensor, List[str]]:
    """Randomly shuffle a mini-batch of images and transcriptions together.

    Args:
        images (torch.Tensor): Tensor of images with shape (B, C, H, W) or
            any batched shape where the first axis is B.
        words (List[str]): Transcriptions corresponding to images.

    Returns:
        Tuple[torch.Tensor, List[str]]: Shuffled image tensor with same shape
            as input and shuffled transcription list of length B.
    """
    assert len(images) == len(words)
    perm = torch.randperm(len(words), device=images.device)
    return images[perm], [words[i] for i in perm.tolist()]


def _endless(loader: DataLoader) -> Iterator:
    """Yield batches from ``loader`` forever, starting a new pass when it runs out.

    Unlike ``itertools.cycle``, which replays the stored batches of its first
    pass, each new pass re-runs the dataset's ``__getitem__``, so shuffling and
    augmentation are redrawn.

    Args:
        loader (DataLoader): Loader to iterate repeatedly.

    Returns:
        Iterator: Infinite iterator over the loader's batches.
    """
    while True:
        yield from loader


# Projector loss weight
SUPERVISED_WEIGHT = float(cfg.get("supervised_weight", 1.0))

# Seed for the initially aligned samples and the global random generators
SEED = int(cfg.get("seed", 0))

# Enable cuDNN autotuner for potentially faster kernels on fixed-size inputs
try:
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = True
except Exception:
    pass


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
    pretrain_ds: PretrainingHTRDataset | None = None,
    syn_batch_ratio: float = cfg.syn_batch_ratio,
    phoc_weight: float = cfg.phoc_loss_weight,
    enable_phoc: bool = cfg.enable_phoc,
    phoc_levels: Tuple[int, ...] = tuple(cfg.phoc_levels),
) -> None:
    """Refine the visual backbone on aligned words, with optional synthetic data.

    Training uses only real samples whose ``dataset.aligned`` value is not ``-1``.
    When ``pretrain_ds`` is provided, batches can mix synthetic and real samples
    according to ``syn_batch_ratio``. The objective combines CTC supervision and,
    when enabled, a PHOC term.

    Args:
        dataset (HTRDataset): Dataset that provides images, transcriptions, and
            alignment flags.
        backbone (HTRNet): Backbone network to optimize.
        num_epochs (int): Number of optimization epochs. With synthetic data,
            one epoch spans the synthetic loader.
        batch_size (int): Total mini-batch size used for refinement.
        lr (float): Learning rate for the Adam optimizer.
        main_weight (float): Weight applied to the main CTC loss.
        aux_weight (float): Weight applied to the auxiliary CTC loss branch.
        pretrain_ds (PretrainingHTRDataset | None): Optional synthetic dataset
            mixed into refinement batches.
        syn_batch_ratio (float): Fraction of each mini-batch drawn from
            ``pretrain_ds``. ``1.0`` means synthetic-only, ``0.0`` disables
            synthetic sampling.
        phoc_weight (float): Weight applied to the PHOC loss term.
        enable_phoc (bool): Whether PHOC loss is included.
        phoc_levels (Tuple[int, ...]): Pyramid levels used to build PHOC targets.
    """
    # The refinement path in this codebase is designed for CUDA execution.
    # Keep the explicit assertion so failures happen early and readably.
    device = next(backbone.parameters()).device
    assert device.type == "cuda", "Backbone is not on a CUDA device"

    # Train mode enables dropout/batchnorm updates during refinement.
    backbone.train().to(device)

    # Build CTC mapping once from the dataset-specific charset. This mapping is
    # reused for every batch in this function.
    c2i, _ = load_vocab(dataset_name=dataset.get_dataset_name())

    # Sanity check: one alignment flag per dataset sample.
    assert dataset.aligned.ndim == 1 and len(dataset) == len(dataset.aligned), (
        "Dataset alignment flags vector is malformed."
    )

    # Real-data supervision is available only for samples already aligned to a
    # word id (aligned != -1).
    aligned_indices = (dataset.aligned != -1).nonzero(as_tuple=True)[0]
    subset = torch.utils.data.Subset(dataset, aligned_indices.tolist())

    # Nothing to optimize if we have neither aligned real samples nor synthetic
    # batches enabled.
    if len(aligned_indices) == 0 and (pretrain_ds is None or syn_batch_ratio <= 0):
        return

    # Split each mini-batch into synthetic and real portions. If synthetic data
    # is absent, syn_bs is forced to 0 and the full batch is real.
    syn_bs = int(batch_size * syn_batch_ratio) if pretrain_ds is not None else 0
    gt_bs = batch_size - syn_bs

    # Loader for aligned real samples (ground truth supervision).
    gt_loader = None
    if gt_bs > 0 and len(aligned_indices) > 0:
        gt_loader = DataLoader(
            subset,
            batch_size=gt_bs,
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

    # Loader for optional synthetic pretraining samples.
    pretrain_loader = None
    if pretrain_ds is not None and syn_bs > 0:
        pretrain_loader = DataLoader(
            pretrain_ds,
            batch_size=syn_bs,
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

    # Epoch length convention:
    # - if synthetic is enabled, one epoch follows pretrain_loader length;
    # - real batches are drawn from an endless re-iteration of gt_loader to
    #   match that length, so every pass gets fresh shuffling and augmentation.
    if pretrain_loader is not None:
        epoch_loader = pretrain_loader
        gt_iter = _endless(gt_loader) if gt_loader is not None else None
    else:
        epoch_loader = gt_loader
        gt_iter = None

    # Optimizer updates only backbone parameters in this stage.
    optimizer = optim.AdamW(backbone.parameters(), lr=lr)

    # Main refinement loop.
    for epoch in range(1, num_epochs + 1):
        epoch_loss: float = 0.0
        effective_batches = 0
        for batch in epoch_loader:
            # Batch assembly has two modes:
            # 1) synthetic-driven epoch (optionally mixed with real aligned data)
            # 2) real-only aligned batches.
            if pretrain_loader is not None:
                # Synthetic dataset already returns text strings for CTC.
                imgs, trans = batch
                words = list(trans)
                if gt_iter is not None:
                    # Append real aligned samples, convert alignment ids to words,
                    # then reshuffle so model does not see fixed ordering
                    # (all-real first, all-synthetic second).
                    imgs_gt, _, aligned = next(gt_iter)
                    imgs = torch.cat([imgs_gt.to(device), imgs.to(device)], dim=0)
                    words = [str(dataset.unique_words[i]).strip() for i in aligned.tolist()] + words
                    imgs, words = _shuffle_batch(imgs, words)
                else:
                    imgs = imgs.to(device)
            else:
                # Real-only path: convert aligned ids to their string labels.
                imgs, _, aligned = batch
                words = [str(dataset.unique_words[i]).strip() for i in aligned.tolist()]
                imgs = imgs.to(device)

            _assert_finite(imgs, "images")

            # Forward pass through the backbone. Request features only when
            # they are used (PHOC), which keeps compatibility with lightweight
            # dummy backbones used in tests.
            need_feats = enable_phoc and backbone.phoc_head is not None
            out = backbone(imgs, return_feats=need_feats)

            # This routine expects the dual-CTC-head contract:
            # (main_logits, aux_logits, ...).
            if not isinstance(out, (tuple, list)) or len(out) < 2:
                raise RuntimeError("Expected network.forward() → (main, aux, …)")
            main_logits, aux_logits = out[:2]
            # With features requested, out is (main, aux, descriptors, phoc_logits).
            phoc_logits = out[3] if need_feats else None

            # CTC dimensions:
            # T = input sequence length, K = batch size.
            T, K, _ = main_logits.shape
            assert main_logits.shape[2] == len(c2i) + 1, "CTC class dimension mismatch"

            # Encode text targets into the flattened CTC representation.
            targets, tgt_lens = encode_for_ctc(words, c2i, device=device)

            # CTC input lengths are constant across the batch because all samples
            # share the same time dimension T after the forward pass.
            inp_lens = torch.full((K,), T, dtype=torch.int32, device=device)

            # Main objective: CTC on the recurrent/main branch.
            loss_main = _ctc_loss_fn(main_logits, targets, inp_lens, tgt_lens)
            # Auxiliary objective: CTC on the secondary head.
            loss_aux = _ctc_loss_fn(aux_logits, targets, inp_lens, tgt_lens)

            # Optional PHOC regularization branch (off by default).
            loss_phoc = torch.tensor(0.0, device=device)
            if enable_phoc and phoc_logits is not None:
                phoc_targets = build_phoc_description(words, c2i, levels=phoc_levels)
                phoc_targets = phoc_targets.float().to(device)
                loss_phoc = F.binary_cross_entropy_with_logits(phoc_logits, phoc_targets)

            # Weighted multi-objective loss.
            loss = (
                main_weight * loss_main
                + aux_weight * loss_aux
                + phoc_weight * loss_phoc
            )
            _assert_finite(loss, "loss")

            # Standard optimizer step with explicit finite-gradient check.
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            _assert_grad_finite(backbone, "backbone")
            optimizer.step()
            epoch_loss += loss.item()
            effective_batches += 1
        # Debug logging kept disabled to reduce console noise on long runs.
        # if effective_batches:
        #     avg_loss = epoch_loss / effective_batches
        #     print(f"Epoch {epoch:03d}/{num_epochs} – avg loss: {avg_loss:.4f}")
        # else:
        #     print(f"Epoch {epoch:03d}/{num_epochs} – no aligned batch encountered")

    # Leave backbone in eval mode for downstream phases (projector/alignment).
    backbone.eval()


# File: alignment/trainer.py


def train_projector(  # pylint: disable=too-many-arguments
    dataset: "HTRDataset",
    backbone: "HTRNet",
    projector: nn.Module,
    num_epochs: int = cfg.projector_epochs,
    batch_size: int = cfg.projector_batch_size,
    lr: float = cfg.projector_lr,
    num_workers: int = cfg.projector_workers,
    weight_decay: float = cfg.projector_weight_decay,
    device: torch.device | str = cfg.device,
) -> None:
    """
    Train a single projector network to map backbone features to embeddings.

    This function freezes the backbone, harvests image descriptors for the entire
    dataset, and then trains the projector using a combination of an unsupervised
    Optimal Transport (OT) loss and a supervised MSE loss on pre-aligned samples.
    Args:
        dataset: The HTRDataset containing the data.
        backbone: The HTRNet model (frozen) to extract features from.
        projector: The projector network to be trained.
        num_epochs: The number of training epochs.
        batch_size: The batch size for training the projector.
        lr: The learning rate for the AdamW optimizer.
        num_workers: The number of workers for the DataLoader.
        weight_decay: The weight decay for the optimizer.
        device: The device to run the training on.
    """
    # ---------------------------------------------------------------- setup
    device = torch.device(device)
    assert device.type == "cuda", "Projector training must be on a CUDA device"
    backbone = backbone.to(device).eval()  # freeze visual encoder
    projector = projector.to(device).train()

    word_embs_cpu = dataset.unique_word_embeddings
    if word_embs_cpu is None:
        raise RuntimeError("FATAL: dataset.unique_word_embeddings is required but was not found.")

    # Target probability for each unique word – use uniform if absent
    # --- THIS BLOCK IS NOW FIXED ---
    probs_attr = getattr(dataset, "unique_word_probs", None)
    if probs_attr is not None and len(probs_attr) > 0:
        if isinstance(probs_attr, list):
            word_probs_cpu = torch.tensor(probs_attr, dtype=torch.float)
        else:  # It's already a tensor
            word_probs_cpu = probs_attr.float()
    else:
        v = word_embs_cpu.size(0)
        word_probs_cpu = torch.full((v,), 1.0 / v)

    word_embs = word_embs_cpu.to(device)
    word_probs = word_probs_cpu.to(device)

    # ---------------------------------------------------------------- 1. Harvest descriptors for the whole dataset
    # Augmentations are temporarily disabled inside ``harvest_backbone_features`` to get stable descriptors.
    feats_all, aligned_all = harvest_backbone_features(
        dataset,
        backbone,
        batch_size=64,
        num_workers=num_workers,
        device=device,
    )
    assert feats_all.shape[1] == backbone.feat_dim, (
        "Descriptor dimension mismatch after harvesting features."
    )

    # ---------------------------------------------------------------- 2. Create a new DataLoader for projector training
    # This loader will shuffle the collected features for effective training.
    proj_loader = DataLoader(
        TensorDataset(feats_all, aligned_all),
        batch_size=batch_size,
        shuffle=True,  # Shuffle is True here for training
        pin_memory=(device.type == "cuda"),
        num_workers=num_workers,
    )

    # ---------------------------------------------------------------- 3. Optimiser + loss
    criterion = ProjectionLoss(supervised_weight=SUPERVISED_WEIGHT).to(device)
    optimiser = optim.AdamW(projector.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler("cuda")

    for _epoch in range(1, num_epochs + 1):
        for feats_cpu, align_cpu in proj_loader:
            feats = feats_cpu.to(device)
            align = align_cpu.to(device)

            optimiser.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.float16):
                pred = projector(feats)

            # Keep OT loss in fp32 for solver stability.
            with autocast("cuda", enabled=False):
                loss = criterion.forward(
                    pred.float(),
                    word_embs.float(),
                    align,
                    word_probs.float(),
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm=1.0)
            scaler.step(optimiser)
            scaler.update()


def alternating_refinement(
    dataset: HTRDataset,
    backbone: HTRNet,
    projector: nn.Module,
    *,
    rounds: int = cfg.alt_rounds,
    backbone_epochs: int = cfg.refine_epochs,
    projector_epochs: int = cfg.projector_epochs,
    refine_kwargs: dict | None = None,
    projector_kwargs: dict | None = None,
    align_kwargs: dict | None = None,
) -> None:
    """
    Perform an alternating training cycle between the backbone and projector.

    This function implements a semi-supervised learning strategy where the backbone
    and projector are trained in alternation. In each round, the backbone is first
    refined, then the projector is trained. After a set number of rounds, more
    instances from the dataset are pseudo-labeled using Optimal Transport (OT) alignment.
    This cycle continues as long as there are unaligned instances in the dataset.
    After each alignment step the Character Error Rate (CER) is printed for both
    the training and test sets.

    Args:
        dataset: The HTRDataset to be used for training.
        backbone: The HTRNet model.
        projector: Projector model.
        rounds: The number of backbone/projector training cycles per alignment pass.
        backbone_epochs: The number of epochs for each backbone refinement round.
        projector_epochs: The number of epochs for each projector training round.
        refine_kwargs: Additional keyword arguments for refine_visual_backbone.
        projector_kwargs: Additional keyword arguments for train_projector.
        align_kwargs: Additional keyword arguments for align_more_instances.
    """

    out_dir = "results"
    out_path = Path(out_dir)
    if out_path.is_dir():
        for f in out_path.glob("pseudo_labels_round_*.txt"):
            f.unlink()

    # Dataset-specific output folder for pseudo-label TSVs
    save_dir = (
        Path("results")
        / dataset.get_dataset_name()
        / "pseudo_labels"
        / f"{dataset.n_aligned}_initial"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    for f in save_dir.glob("pseudo_labels_round_*.txt"):
        f.unlink()

    # Save initial aligned examples (seeds) as round 0
    init_idx = torch.nonzero(dataset.aligned != -1, as_tuple=True)[0]
    log_pseudo_labels(init_idx, dataset, round_idx=0, out_dir=str(save_dir))

    maybe_load_backbone(backbone, cfg)

    device = torch.device(cfg.device)
    backbone.to(device)
    projector = projector.to(device)

    if refine_kwargs is None:
        refine_kwargs = {}
    if projector_kwargs is None:
        projector_kwargs = {}
    if align_kwargs is None:
        align_kwargs = {}
    # Set default alignment arguments from config
    align_kwargs.setdefault("batch_size", cfg.align_batch_size)
    align_kwargs.setdefault("device", cfg.align_device)
    align_kwargs.setdefault("reg", cfg.align_reg)
    align_kwargs.setdefault("k", cfg.align_k)

    # Build test dataset with n_aligned=0 to avoid seeding on the test split.

    test_cfg = copy.deepcopy(getattr(dataset, "config", None))
    if test_cfg is not None:
        test_cfg.n_aligned = 0
    test_dataset = HTRDataset(
        basefolder=dataset.basefolder,
        subset="test",
        fixed_size=dataset.fixed_size,
        character_classes=dataset.character_classes,
        config=test_cfg,
    )

    def _log_test_metrics(
        round_idx: int, correct_pseudo: int, rules_disagree: int, correct_argmax: int
    ) -> None:
        """Evaluate test CER/WER for the current backbone and append a TSV row.

        Args:
            round_idx (int): One-based row index in the round metrics file.
            correct_pseudo (int): Correct pseudo-labels added in this round.
            rules_disagree (int): New pseudo-labels where the argmax rule picks
                a different word than the barycentre rule.
            correct_argmax (int): New pseudo-labels the argmax rule gets right.
        """
        cer_test = compute_cer(
            test_dataset, backbone, batch_size=cfg.eval_batch_size, device=cfg.device, k=4
        )
        wer_test = compute_wer(
            test_dataset,
            backbone,
            batch_size=cfg.eval_batch_size,
            device=cfg.device,
        )
        log_round_metrics(
            round_idx,
            correct_pseudo,
            cer_test,
            float(wer_test),
            cfg.round_metrics_file,
            basefolder=str(getattr(cfg.dataset, "basefolder", "")),
            n_aligned=int(getattr(cfg.dataset, "n_aligned", 0)),
            seed=SEED,
            rules_disagree=rules_disagree,
            correct_argmax=correct_argmax,
        )

    cycle_idx = 1
    while (dataset.aligned == -1).any():
        for r in range(rounds):
            print(f"[Round {r + 1}/{rounds}] Refining backbone...")
            if backbone_epochs > 0:
                refine_visual_backbone(
                    dataset,
                    backbone,
                    num_epochs=backbone_epochs,
                    **refine_kwargs,
                )

            for param in backbone.parameters():
                param.requires_grad_(False)
            for param in projector.parameters():
                param.requires_grad_(True)

            print(f"[Round {r + 1}/{rounds}] Training projector...")
            if projector_epochs > 0:
                train_projector(
                    dataset,
                    backbone,
                    projector,
                    num_epochs=projector_epochs,
                    **projector_kwargs,
                )

            for param in backbone.parameters():
                param.requires_grad_(True)
            for param in projector.parameters():
                param.requires_grad_(False)

            backbone_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
            projector_trainable = sum(p.numel() for p in projector.parameters() if p.requires_grad)
            assert backbone_trainable > 0 and projector_trainable == 0, (
                "Exactly one module family (backbone or projector) should be trainable."
            )

        print("[Cycle] Aligning more instances...")
        assert (dataset.aligned != -1).sum() > 0, "Cannot align more instances with zero seeds."
        # Save current alignment state before pseudo-labelling
        prev_aligned = dataset.aligned.clone()
        # Perform Optimal Transport alignment to pseudo-label more instances
        plan, _, _ = align_more_instances(dataset, backbone, projector, **align_kwargs)
        changed = torch.nonzero((prev_aligned == -1) & (dataset.aligned != -1), as_tuple=True)[0]
        # Log newly pseudo-labeled items for this cycle
        log_pseudo_labels(changed, dataset, round_idx=cycle_idx, out_dir=str(save_dir))
        correct_round = sum(
            dataset.unique_words[dataset.aligned[i]] == dataset.transcriptions[i].strip()
            for i in changed.tolist()
        )
        argmax_word = plan[changed].argmax(dim=1)
        rules_disagree = int((argmax_word != dataset.aligned[changed].long()).sum())
        correct_argmax = sum(
            dataset.unique_words[w] == dataset.transcriptions[i].strip()
            for i, w in zip(changed.tolist(), argmax_word.tolist())
        )
        n_new = len(changed)
        print(
            f"[Align] rules disagree on {rules_disagree}/{n_new}; "
            f"argmax correct {correct_argmax}/{n_new} (barycentre {correct_round}/{n_new})"
        )
        _log_test_metrics(cycle_idx, correct_round, rules_disagree, correct_argmax)

        # --- restrict CER to already-aligned samples -------------------
        # aligned_idx = torch.nonzero(dataset.aligned != -1, as_tuple=True)[0]
        # aligned_subset = torch.utils.data.Subset(dataset, aligned_idx.tolist())
        # print('CER on the aligned subset: ')
        # compute_cer(
        #     aligned_subset,
        #     backbone,
        #     batch_size=cfg.eval_batch_size,
        #     device=cfg.device,
        #     k=4
        # )
        cycle_idx += 1

    # ────────────────────────────────────────────────────────────────────────────────────────
    #   FINISH: every sample has an alignment – keep refining
    # ────────────────────────────────────────────────────────────────────────────────────────
    if not (dataset.aligned == -1).any():
        print("[Info] All samples aligned – continuing backbone training …")
        refine_visual_backbone(dataset, backbone, num_epochs=backbone_epochs, **refine_kwargs)
        # Final row: the backbone after training on every label, with no new pseudo-labels.
        _log_test_metrics(cycle_idx, 0, 0, 0)


if __name__ == "__main__":
    from types import SimpleNamespace

    # Best-effort reproducibility: weights, batch order and augmentations.
    # GPU kernels (CUDA CTC, cuDNN autotuning) can still differ between runs.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

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
        seed=SEED,
    )

    arch = SimpleNamespace(**cfg["architecture"])
    backbone = HTRNet(arch, nclasses=len(dataset.character_classes) + 1)
    # maybe_load_backbone(backbone, cfg)
    projector = Projector(arch.feat_dim, dataset.word_emb_dim, dropout=0.2)

    # Synthetic dataset for pretraining samples
    syn_cfg = cfg.get("synthetic_dataset", None)
    pre_syn_ds = None
    if syn_cfg is not None:
        pre_syn_ds = PretrainingHTRDataset(
            list_file=syn_cfg.list_file,
            base_path=syn_cfg.base_path,
            n_random=syn_cfg.n_random,
            fixed_size=tuple(syn_cfg.fixed_size),
            transforms=aug_transforms,
            preload_images=syn_cfg.get("preload_images", False),
            random_seed=syn_cfg.get("random_seed", 0),
        )

    alternating_refinement(
        dataset,
        backbone,
        projector,
        refine_kwargs={"pretrain_ds": pre_syn_ds},
    )
