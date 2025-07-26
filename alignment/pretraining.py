# Copyright
from __future__ import annotations
import os, sys, random, pickle
from pathlib import Path
from types import SimpleNamespace
from omegaconf import OmegaConf

# Set CUDA devices before importing torch
_cfg_file = Path(__file__).parent / "alignment_configs" / "pretraining_config.yaml"
_cfg = OmegaConf.load(_cfg_file)
os.environ["CUDA_VISIBLE_DEVICES"] = str(_cfg.gpu_id)

import torch

def _assert_finite(t: torch.Tensor, name: str) -> None:
    """Check ``t`` for ``NaN`` or ``Inf`` values.

    Args:
        t (torch.Tensor): Tensor produced during training.
        name (str): Label used in the raised assertion message.

    Returns:
        None
    """
    assert torch.isfinite(t).all(), f"{name} contains NaN/Inf"

def _check_grad_finite(model: torch.nn.Module) -> None:
    """Ensure all gradients in ``model`` are finite.

    Args:
        model (torch.nn.Module): Model to validate after ``backward``.

    Returns:
        None
    """
    for n, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"grad NaN/Inf in {n}"
# ------------------------------------------------------------------
# Configuration parameters for pretraining
# ------------------------------------------------------------------
# Add project root to path for imports
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from htr_base.utils.htr_dataset import PretrainingHTRDataset
from htr_base.models import HTRNet
from htr_base.utils.transforms import aug_transforms
from htr_base.utils.metrics import CER
from htr_base.utils.vocab import load_vocab
from alignment.ctc_utils import (
    encode_for_ctc,
    greedy_ctc_decode,
    beam_search_ctc_decode,
)
from alignment.losses import _ctc_loss_fn, SoftContrastiveLoss   # new class
from htr_base.utils.phoc import build_phoc_description
from omegaconf import OmegaConf
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

yaml_cfg = OmegaConf.load(Path(__file__).parent / "alignment_configs" / "pretraining_config.yaml")
GPU_ID = int(yaml_cfg.get("gpu_id", 0))
DEVICE = "cuda"
ENABLE_PHOC = bool(yaml_cfg.get("enable_phoc", False))
PHOC_LEVELS = tuple(yaml_cfg.get("phoc_levels", (1, 2, 3, 4)))
PHOC_W = float(yaml_cfg.get("phoc_loss_weight", 0.1))
ENABLE_CONTR = bool(yaml_cfg.get("contrastive_enable", False))
CONTR_W      = float(yaml_cfg.get("contrastive_weight", 0.0))
CONTR_TAU    = float(yaml_cfg.get("contrastive_tau", 0.07))
CONTR_TTXT   = float(yaml_cfg.get("contrastive_text_T", 1.0))
# Default pretraining configuration
PRETRAINING_CONFIG = {
    "list_file": yaml_cfg.get("list_file", "/gpu-data3/pger/handwriting_rec/mnt/ramdisk/max/90kDICT32px/imlist.txt"),
    "train_set_size": int(yaml_cfg.get("train_set_size", 30000)),
    "test_set_size": int(yaml_cfg.get("test_set_size", 10000)),
    "num_epochs": int(yaml_cfg.get("num_epochs", 10000)),
    "batch_size": int(yaml_cfg.get("batch_size", 128)),
    "learning_rate": float(yaml_cfg.get("learning_rate", 1e-3)),
    "base_path": yaml_cfg.get("base_path", None),
    "fixed_size": tuple(yaml_cfg.get("fixed_size", (64, 256))),
    "device": DEVICE,
    "use_augmentations": bool(yaml_cfg.get("use_augmentations", True)),
    "main_loss_weight": float(yaml_cfg.get("main_loss_weight", 1.0)),
    "aux_loss_weight": float(yaml_cfg.get("aux_loss_weight", 0.1)),
    "save_path": yaml_cfg.get("save_path", "htr_base/saved_models/pretrained_backbone.pt"),
    "save_backbone": bool(yaml_cfg.get("save_backbone", True)),
}
# Architecture configuration for the pretraining backbone
# Loaded from alignment/alignment_configs/pretraining_config.yaml to stay consistent with other scripts
ARCHITECTURE_CONFIG = yaml_cfg["architecture"]

def main(config: dict | None = None) -> Path:
    """Train a small HTRNet on the given image list using dictionary configuration."""
    if config is None:
        config = {}
    config = {**PRETRAINING_CONFIG, **config}
    # Extract parameters from config
    list_file = config["list_file"]
    assert Path(list_file).is_file(), "list_file not found"
    n_random = config.get("train_set_size", config.get("n_random", None))
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    assert batch_size > 0 and batch_size <= 1024, "batch_size must be between 1 and 1024"
    lr = config["learning_rate"]
    assert lr > 0, "learning_rate must be positive"
    base_path = config.get("base_path", None)
    fixed_size = config["fixed_size"]
    assert fixed_size[0] > 0 and fixed_size[1] > 0, "fixed_size dimensions must be positive"
    device = config["device"]
    gpu_id = config.pop("gpu_id", None)
    if gpu_id is not None:
        print("[Pretraining] 'gpu_id' in config is ignored; set it in alignment/alignment_configs/trainer_config.yaml")
    use_augmentations = config.get("use_augmentations", True)
    main_weight = config.get("main_loss_weight", 1.0)
    aux_weight = config.get("aux_loss_weight", 0.1)
    save_path = config.get("save_path", "htr_base/saved_models/pretrained_backbone.pt")
    save_backbone = config.get("save_backbone", False)
    print(f"[Pretraining] Starting with config:")
    print(f"  list_file: {list_file}")
    print(f"  train_set_size: {n_random}")
    print(f"  epochs: {num_epochs}")
    print(f"  batch_size: {batch_size}")
    print(f"  learning_rate: {lr}")
    print(f"  device: {device}")
    print(f"  augmentations: {use_augmentations}")
    print(f"  save_backbone: {save_backbone}")
    print(f"  gpu_id: {GPU_ID}")
    if base_path is None:
        base_path = str(Path(list_file).parent)
    # Create training dataset with optional augmentations
    transforms = aug_transforms if use_augmentations else None
    train_set = PretrainingHTRDataset(
        list_file,
        fixed_size=fixed_size,
        base_path=base_path,
        transforms=transforms,
        n_random=n_random,
        preload_images=True,
        random_seed=0,
    )
    assert len(train_set) > 0, "training dataset is empty"
    assert train_set.fixed_size == fixed_size, "train_set fixed_size mismatch"
    # Separate test set without augmentations
    test_set = PretrainingHTRDataset(
        list_file,
        fixed_size=fixed_size,
        base_path=base_path,
        transforms=None,
        n_random=config.get("test_set_size", 10000),
        preload_images=True,
        random_seed=1,
    )
    assert test_set.fixed_size == fixed_size, "test_set fixed_size mismatch"
    print(f"[Pretraining] Dataset size: {len(train_set)}")
    print(f"[Pretraining] Test set size: {len(test_set)}")
    save_dir = Path(save_path).parent
    c2i_path = save_dir / "c2i.pkl"
    i2c_path = save_dir / "i2c.pkl"

    # Use the fixed vocabulary for training
    c2i, i2c = load_vocab()

    assert 0 not in c2i.values(), "blank index 0 found in c2i values"

    # Optionally (re-)save the dictionaries next to the backbone weights
    if save_backbone:
        save_dir.mkdir(parents=True, exist_ok=True)


    nclasses = len(c2i) + 1
    assert (len(c2i) + 1) == nclasses, "nclasses mismatch with vocab size"
    print(f"[Pretraining] Vocabulary size: {nclasses} (including blank)")
    arch = SimpleNamespace(**ARCHITECTURE_CONFIG)
    net = HTRNet(arch, nclasses=nclasses).to(device).train()
    # if Path(save_path).exists():
    #     print(f"[Pretraining] Loading checkpoint from {save_path}")
    #     state = torch.load(save_path, map_location=device)
    #     net.load_state_dict(state)
    # Print number of parameters
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"[Pretraining] Network parameters: {n_params:,}")
    # Create data loader and optimizer
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=3)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)
    opt = optim.Adam(net.parameters(), lr=lr)
    sched = lr_scheduler.StepLR(opt, step_size=1500, gamma=0.5)
    contr_loss_fn = SoftContrastiveLoss(CONTR_TAU, CONTR_TTXT).to(device)
    print(f"[Pretraining] Starting training...")
    print(f"[Pretraining] PHOC loss enabled: {ENABLE_PHOC}")
    print(f"[Pretraining] Contrastive loss enabled: {ENABLE_CONTR}")
    def _decode_random_samples(ds):
        """Print predictions for up to ten random samples from *ds*."""
        net.eval()
        with torch.no_grad():
            assert len(ds) > 0, "empty dataset for random samples"
            # 1) pick up to 10 random indices
            indices = random.sample(range(len(ds)), min(10, len(ds)))
            # 2) load all (image, gt) pairs and stack into a batch
            imgs, gts = zip(*(ds[i] for i in indices))
            batch = torch.stack(imgs, dim=0).to(device)   # shape: (B, C, H, W)
            # 3) forward-pass the whole batch at once
            logits = net(batch, return_feats=False)[0]   # shape: (T, B, C)
            # 4) decode the entire batch with greedy and beam search
            greedy_preds = greedy_ctc_decode(logits, i2c)               # List[str], len=B
            beam_preds   = beam_search_ctc_decode(logits, i2c, beam_width=5)
            # 5) print GT vs. predictions
            for gt, gr, bm in zip(gts, greedy_preds, beam_preds):
                print(f"GT: '{gt.strip()}' | greedy: '{gr}' | beam5: '{bm}'")
        net.train()
    def _evaluate_cer(loader):
        """Return CER on *loader* and print the value."""
        if len(loader.dataset) == 0:
            print("[Eval] Test set is empty, skipping CER evaluation.")
            return float('nan')
        net.eval()
        metric = CER()
        with torch.no_grad():
            for imgs, trans in loader:
                imgs = imgs.to(device)
                logits = net(imgs, return_feats=False)[0]
                preds = greedy_ctc_decode(logits, i2c)
                for p, t in zip(preds, trans):
                    metric.update(p.strip(), t.strip())
        net.train()
        score = metric.score()
        print(f"[Eval] CER: {score:.4f}")
        return score
        # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        phoc_epoch_loss = 0.0
        running_contr = 0.0
        for imgs, txts in train_loader:
            imgs = imgs.to(device)
            out = net(imgs)
            main_logits, aux_logits, features = out[:3]
            assert features.dim() == 2 and features.size(1) == arch.feat_dim, "features tensor has wrong dimensions or feature size"
            for name, tens in {"main_logits":main_logits, "aux_logits":aux_logits, "features":features}.items(): _assert_finite(tens, name)

            # Prepare CTC targets
            targets, lengths = encode_for_ctc(list(txts), c2i, device=device)
            assert targets.max() < nclasses, "target contains index out of nclasses range"
            inp_lens = torch.full((imgs.size(0),), main_logits.size(0), dtype=torch.int32, device=device)

            # Compute losses
            loss_main = _ctc_loss_fn(main_logits, targets, inp_lens, lengths)
            loss_aux = _ctc_loss_fn(aux_logits, targets, inp_lens, lengths)

            loss_contr = torch.tensor(0.0, device=device)
            if ENABLE_CONTR:
                loss_contr = contr_loss_fn(features, targets, lengths)
                assert loss_contr >= 0 and torch.isfinite(loss_contr), "Contrastive loss is negative or non-finite"

            if ENABLE_PHOC:
                phoc_logits = out[-1]
                phoc_targets = build_phoc_description(list(txts), c2i, levels=PHOC_LEVELS).float().to(device)
                assert phoc_logits.shape == phoc_targets.shape, "shape mismatch"
                loss_phoc = torch.nn.functional.binary_cross_entropy_with_logits(phoc_logits, phoc_targets)
                assert not torch.isnan(loss_phoc), "PHOC loss is NaN"
            else:
                loss_phoc = torch.tensor(0.0, device=device)

            total_loss = (main_weight * loss_main +
                            aux_weight  * loss_aux  +
                            PHOC_W      * loss_phoc +
                            CONTR_W     * loss_contr)
            _assert_finite(total_loss, "total_loss")
            # Optimization step
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); _check_grad_finite(net)
            opt.step()
            opt.zero_grad(set_to_none=True)
            epoch_loss += total_loss.item()
            phoc_epoch_loss += loss_phoc.item()
            running_contr += loss_contr.item()
            num_batches += 1
        avg_loss = epoch_loss / max(1, num_batches)
        avg_phoc_loss = phoc_epoch_loss / max(1, num_batches)
        avg_contr_loss = running_contr / max(1, num_batches)
        sched.step()
        # Print progress every 20 epochs or on last epoch
        if (epoch + 1) % 30 == 0 or epoch == num_epochs - 1:
            lr_val = sched.get_last_lr()[0]
            import math
            assert lr_val > 0 and math.isfinite(lr_val), "learning rate underflow or became non-finite"
            msg = f"[Pretraining] Epoch {epoch+1:03d}/{num_epochs} - Loss: {avg_loss:.4f}"
            if ENABLE_PHOC:
                msg += f" - PHOC: {avg_phoc_loss:.4f}"
            if ENABLE_CONTR:
                msg += f" - Contr: {avg_contr_loss:.4f}"
            msg += f" - lr: {lr_val:.2e}"
            print(msg)
            _evaluate_cer(test_loader)
            _decode_random_samples(test_set)
            if save_backbone:
                # Save the trained model and vocabulary
                save_dir = Path(save_path).parent
                save_dir.mkdir(parents=True, exist_ok=True); assert os.access(save_dir, os.W_OK), "save_dir is not writable"
                torch.save(net.state_dict(), save_path)
                print(f"[Pretraining] Model saved to: {save_path}")
    return Path(save_path)
if __name__ == '__main__':
    main()
