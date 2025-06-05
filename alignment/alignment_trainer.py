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
    "alt_backbone_epochs": 10,     # epochs for each backbone refinement phase
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
            targets, tgt_lens = encode_for_ctc(ext_words, c2i, device=device)
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
    """Freeze *backbone*, collect image descriptors -> train *projector* with OT loss.

    All images are first forwarded through `backbone` **without augmentation** so
    that a descriptor is obtained for every sample. The descriptors together with
    their `aligned` indices and `is_in_dict` flags are cached inside a temporary
    :class:`torch.utils.data.TensorDataset`. The `projector` is then optimised
    using :class:`ProjectionLoss` on this dataset.
    """
    # ---------------------------------------------------------------- setup
    device = torch.device(device)
    backbone = backbone.to(device).eval()          # freeze visual encoder
    projector = projector.to(device).train()       # learnable mapping

    word_embs_cpu = dataset.external_word_embeddings  # (V, E)
    if word_embs_cpu is None:
        raise RuntimeError("dataset.external_word_embeddings is required")

    # target probability for each external word – use uniform if absent
    if hasattr(dataset, "external_word_probs") and dataset.external_word_probs is not None:
        word_probs_cpu = dataset.external_word_probs.float()
    else:
        v = word_embs_cpu.size(0)
        word_probs_cpu = torch.full((v,), 1.0 / v)

    word_embs = word_embs_cpu.to(device)
    word_probs = word_probs_cpu.to(device)
    # ---------------------------------------------------------------- 1. Harvest descriptors for the whole dataset
    feats_buf, align_buf, dict_buf = [], [], []
    subset_bak = dataset.subset
    transforms_bak = dataset.transforms
    if dataset.subset == "train":
        dataset.subset = "eval"
    dataset.transforms = None
    harvest_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    start = 0
    with torch.no_grad():
        for imgs, _txt, aligned in harvest_loader:
            imgs = imgs.to(device)
            out = backbone(imgs)
            feats = out[-1]
            if feats.dim() != 2:
                raise RuntimeError("Expected (B, feat_dim) descriptors")
            end = start + imgs.size(0)
            feats_buf.append(feats.cpu())
            align_buf.append(aligned.cpu())
            dict_buf.append(dataset.is_in_dict[start:end].clone())
            start = end
    dataset.subset = subset_bak
    dataset.transforms = transforms_bak
    if not feats_buf:
        raise RuntimeError("Empty dataset – cannot train projector")
    feats_all = torch.cat(feats_buf, dim=0)
    aligned_all = torch.cat(align_buf, dim=0)
    dict_all = torch.cat(dict_buf, dim=0)
    # ---------------------------------------------------------------- 2. Loader for projector training only
    proj_loader = DataLoader(
        TensorDataset(feats_all, aligned_all, dict_all),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )
    # ---------------------------------------------------------------- 3. Optimiser + loss
    criterion = ProjectionLoss().to(device)
    optimiser = optim.AdamW(projector.parameters(), lr=lr, weight_decay=weight_decay)
    # ---------------------------------------------------------------- 4. Training loop
    for epoch in range(1, num_epochs + 1):
        running = 0.0
        for feats_cpu, align_cpu, dict_cpu in proj_loader:
            mask = dict_cpu.bool()
            if not mask.any():
                continue
            feats = feats_cpu[mask].to(device)
            align = align_cpu[mask].to(device)
            pred = projector(feats)
            loss = criterion(pred, word_embs, align, word_probs)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            running += loss.item()
        avg = running / max(1, len(proj_loader))
        if epoch == 1 or epoch % 10 == 0 or epoch == num_epochs:
            print(f"[Projector] epoch {epoch:03}/{num_epochs}  loss={avg:.4f}")

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

    for r in range(rounds):
        print(f"[Cycle {r + 1}/{rounds}] Refining backbone...")
        if backbone_epochs > 0:
            refine_visual_backbone(
                dataset,
                backbone,
                num_epochs=backbone_epochs,
                **refine_kwargs,
            )

        for param in backbone.parameters():
            param.requires_grad_(False)

        print(f"[Cycle {r + 1}/{rounds}] Training projector...")
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

        print(f"[Cycle {r + 1}/{rounds}] Aligning more instances...")
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
        fixed_size=(128, 256),
        transforms=None,
        config=DummyCfg(),
    )

