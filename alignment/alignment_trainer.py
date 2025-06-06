from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Tuple, List
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
from alignment.losses import ProjectionLoss
from alignment.ctc_utils import encode_for_ctc
from alignment.losses import _ctc_loss_fn
from alignment.alignment_utilities import align_more_instances
from htr_base.utils.transforms import aug_transforms

# --------------------------------------------------------------------------- #
#                           Hyperparameter defaults                            #
# --------------------------------------------------------------------------- #
# These constants centralise the most common tunable values so they can be
# tweaked from a single place. Functions in this file use them as defaults but
# also accept explicit keyword arguments.
HP = {
    "refine_batch_size": 128,     # mini-batch size for backbone refinement
    "refine_lr": 1e-4,            # learning rate for backbone refinement
    "refine_main_weight": 1.0,    # weight for the main CTC loss branch
    "refine_aux_weight": 0.1,     # weight for the auxiliary CTC loss branch
    "projector_epochs": 150,      # training epochs for the projector network
    "projector_batch_size": 4000,  # mini-batch size for projector training
    "projector_lr": 1e-4,         # learning rate for projector optimisation
    "projector_workers": 1,       # dataloader workers when collecting features
    "projector_weight_decay": 1e-4,  # weight decay for projector optimiser
    "device": "cuda",             # default compute device
    "alt_rounds": 4,              # number of backbone/projector cycles
    "alt_backbone_epochs": 20,     # epochs for each backbone refinement phase
    "alt_projector_epochs": 100,    # epochs for each projector training phase
}

# --------------------------------------------------------------------------- #
#                               Helper utilities                              #
# --------------------------------------------------------------------------- #
def _build_vocab_dicts(dataset: HTRDataset) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create <char→id> / <id→char> dicts leaving index 0 for the CTC blank."""
    chars: List[str] = list(dataset.character_classes)
    if " " not in chars:
        chars.append(" ")
    chars = sorted(set(chars))
    c2i = {c: i + 1 for i, c in enumerate(chars)}
    i2c = {i + 1: c for i, c in enumerate(chars)}
    return c2i, i2c


# --------------------------------------------------------------------------- #
#                            Main refinement routine                          #
# --------------------------------------------------------------------------- #
def refine_visual_backbone(
    dataset: HTRDataset,
    backbone: HTRNet,
    num_epochs: int,
    *,
    batch_size: int = HP["refine_batch_size"],
    lr: float = HP["refine_lr"],
    main_weight: float = HP["refine_main_weight"],
    aux_weight: float = HP["refine_aux_weight"],
) -> None:
    """Fine‑tune *backbone* only on words already aligned to external words."""
    print(f"[Refine] epochs={num_epochs}  batch_size={batch_size}  lr={lr}")
    device = next(backbone.parameters()).device
    backbone.train().to(device)
    # Build CTC mapping once.
    c2i, _ = _build_vocab_dicts(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,            # keep CI simple
        pin_memory=(device.type == "cuda"),
    )
    optimizer = optim.AdamW(backbone.parameters(), lr=lr)
    for epoch in range(1, num_epochs + 1):
        epoch_loss: float = 0.0
        effective_batches = 0
        for imgs, _, aligned in dataloader:  # we ignore the transcription string
            aligned_mask = aligned != -1
            if not aligned_mask.any():
                # nothing to learn from this mini‑batch
                continue
            # ── select only aligned items ────────────────────────────────
            sel_idx = aligned_mask.nonzero(as_tuple=True)[0]             # (K,)
            imgs_sel = imgs[sel_idx].to(device)                          # (K,1,H,W)
            aligned_ids = aligned[sel_idx].tolist()                     # List[int]
            ext_words = [dataset.external_words[i] for i in aligned_ids]
            # ── forward ─────────────────────────────────────────────────
            out = backbone(imgs_sel, return_feats=False)
            if not isinstance(out, (tuple, list)) or len(out) < 2:
                raise RuntimeError("Expected network.forward() → (main, aux, …)")
            main_logits, aux_logits = out[:2]           # (T,K,C)

            T, K, _ = main_logits.shape
            # ── encode labels ───────────────────────────────────────────
            targets, tgt_lens = encode_for_ctc(ext_words, c2i, device="cpu")

            inp_lens = torch.full((K,), T, dtype=torch.int32, device=device)
            # ── losses ─────────────────────────────────────────────────
            loss_main = _ctc_loss_fn(main_logits, targets, inp_lens, tgt_lens)
            loss_aux  = _ctc_loss_fn(aux_logits,  targets, inp_lens, tgt_lens)
            loss = main_weight * loss_main + aux_weight * loss_aux
            # ── optimisation step ──────────────────────────────────────
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            effective_batches += 1
        if effective_batches:
            avg_loss = epoch_loss / effective_batches
            print(f"Epoch {epoch:03d}/{num_epochs} – avg loss: {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch:03d}/{num_epochs} – no aligned batch encountered")
    print("[Refine] finished.")

# File: alignment/alignment_trainer.py


def train_projector(  # pylint: disable=too-many-arguments
    dataset: "HTRDataset",
    backbone: "HTRNet",
    projector: nn.Module,
    num_epochs: int = HP["projector_epochs"],
    batch_size: int = HP["projector_batch_size"],
    lr: float = HP["projector_lr"],
    num_workers: int = HP["projector_workers"],
    weight_decay: float = HP["projector_weight_decay"],
    device: torch.device | str = HP["device"],
) -> None:
    """
    Freeze `backbone`, collect all image descriptors, and then train the `projector`
    using a combination of an unsupervised Optimal Transport loss on all samples
    and a supervised MSE loss on the subset of pre-aligned samples.
    
    All images are first forwarded through the frozen `backbone` **without
    augmentation** to obtain a stable descriptor for every sample. These descriptors,
    along with their alignment information, are cached in a temporary TensorDataset.
    The `projector` is then optimised using this dataset.
    """
    # ---------------------------------------------------------------- setup
    device = torch.device(device)
    backbone = backbone.to(device).eval()          # freeze visual encoder
    projector = projector.to(device).train()       # learnable mapping
    
    word_embs_cpu = dataset.external_word_embeddings
    if word_embs_cpu is None:
        raise RuntimeError("FATAL: dataset.external_word_embeddings is required but was not found.")
        
    # Target probability for each external word – use uniform if absent
    # --- THIS BLOCK IS NOW FIXED ---
    probs_attr = getattr(dataset, "external_word_probs", None)
    if probs_attr is not None and len(probs_attr) > 0:
        if isinstance(probs_attr, list):
            word_probs_cpu = torch.tensor(probs_attr, dtype=torch.float)
        else: # It's already a tensor
            word_probs_cpu = probs_attr.float()
    else:
        v = word_embs_cpu.size(0)
        print("Warning: `dataset.external_word_probs` not found or is empty. Using uniform distribution.")
        word_probs_cpu = torch.full((v,), 1.0 / v)
        
    word_embs = word_embs_cpu.to(device)
    word_probs = word_probs_cpu.to(device)

    # ---------------------------------------------------------------- 1. Harvest descriptors for the whole dataset
    # It's crucial to disable augmentations during this phase to get consistent
    # features for each image.
    feats_buf, align_buf = [], []
    
    # Backup and temporarily modify dataset state for harvesting
    transforms_bak = dataset.transforms
    dataset.transforms = None
    
    harvest_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False, # Shuffle must be False to maintain order
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    
    print("Harvesting image descriptors from the backbone...")
    with torch.no_grad():
        for imgs, _txt, aligned in harvest_loader:
            imgs = imgs.to(device)
            # The last element of the backbone's output is the feature descriptor
            feats = backbone(imgs, return_feats=True)[-1]
            if feats.dim() != 2:
                raise RuntimeError(f"Expected (B, feat_dim) descriptors, but got shape {feats.shape}")
                
            feats_buf.append(feats.cpu())
            align_buf.append(aligned.cpu())

    # Restore dataset's original state
    dataset.transforms = transforms_bak
    

    # Consolidate all harvested data into single tensors
    feats_all = torch.cat(feats_buf, dim=0)
    aligned_all = torch.cat(align_buf, dim=0)

    # ---------------------------------------------------------------- 2. Create a new DataLoader for projector training
    # This loader will shuffle the collected features for effective training.
    proj_loader = DataLoader(
        TensorDataset(feats_all, aligned_all),
        batch_size=batch_size,
        shuffle=True, # Shuffle is True here for training
        pin_memory=(device.type == "cuda"),
    )

    # ---------------------------------------------------------------- 3. Optimiser + loss
    criterion = ProjectionLoss().to(device)
    optimiser = optim.AdamW(projector.parameters(), lr=lr, weight_decay=weight_decay)

    # ---------------------------------------------------------------- 4. Training loop
    print("Starting projector training...")
    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        # The dataloader now yields batches of (features, alignment_info)
        for feats_cpu, align_cpu in proj_loader:
            
            # Move the entire batch to the training device. No filtering!
            feats = feats_cpu.to(device)
            align = align_cpu.to(device)

            # --- Pre-computation Assertions ---
            assert torch.isfinite(feats).all(), \
                "FATAL: Non-finite values (NaN or Inf) detected in features fed to the projector."

            # --- Forward pass ---
            pred = projector(feats)

            # --- Loss Calculation ---
            # The criterion internally handles which samples are used for the
            # supervised loss based on the `align` tensor (where -1 indicates unsupervised).
            loss = criterion.forward(pred, word_embs, align, word_probs)
            
            assert torch.isfinite(loss), f"FATAL: Loss is not finite ({loss.item()}). Aborting."

            # --- Optimization Step ---
            
            loss.backward()
            grad_ok = all(
                torch.isfinite(p.grad).all()
                for p in projector.parameters()
                if p.grad is not None
            )
            assert grad_ok, 'gradient explosion in projector - contains NaN/Inf'

            # (CRITICAL) Gradient Clipping: Prevents exploding gradients from corrupting model weights.
            torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm=1.0)
            
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(proj_loader))
        if epoch == 1 or epoch % 10 == 0 or epoch == num_epochs:
            print(f"[Projector] epoch {epoch:03d}/{num_epochs} – avg loss: {avg_loss:.4f}")
            
    print("[Projector] training complete ✔")



def alternating_refinement(
    dataset: HTRDataset,
    backbone: HTRNet,
    projector: nn.Module,
    *,
    rounds: int = HP["alt_rounds"],
    backbone_epochs: int = HP["alt_backbone_epochs"],
    projector_epochs: int = HP["alt_projector_epochs"],
    refine_kwargs: dict | None = None,
    projector_kwargs: dict | None = None,
    align_kwargs: dict | None = None,
) -> None:
    """Alternately train ``backbone`` and ``projector`` with OT alignment."""

    if refine_kwargs is None:
        refine_kwargs = {}
    if projector_kwargs is None:
        projector_kwargs = {}
    if align_kwargs is None:
        align_kwargs = {}

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

            print(f"[Round {r + 1}/{rounds}] Training projector...")
            if projector_epochs > 0:
                _probs_backup = None
                if isinstance(getattr(dataset, "external_word_probs", None), list):
                    _probs_backup = dataset.external_word_probs
                    dataset.external_word_probs = torch.tensor(
                        _probs_backup, dtype=torch.float
                    )

                train_projector(
                    dataset,
                    backbone,
                    projector,
                    num_epochs=projector_epochs,
                    **projector_kwargs,
                )

                if _probs_backup is not None:
                    dataset.external_word_probs = _probs_backup

            for param in backbone.parameters():
                param.requires_grad_(True)

        print("[Cycle] Aligning more instances...")
        align_more_instances(dataset, backbone, projector, **align_kwargs)


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
        n_aligned = 1000          # how many images to mark as aligned (≈ training signal)

    dataset = HTRDataset(
        str(gw_folder),
        subset="train",
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
    )
    backbone = HTRNet(arch, nclasses=len(dataset.character_classes) + 1)
    projector = Projector(arch.feat_dim, dataset.word_emb_dim)

    alternating_refinement(dataset, backbone, projector)

