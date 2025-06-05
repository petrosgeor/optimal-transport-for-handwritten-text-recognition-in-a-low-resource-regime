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

def _ctc_loss_fn(
    logits: torch.Tensor,
    targets: torch.IntTensor,
    inp_lens: torch.IntTensor,
    tgt_lens: torch.IntTensor,
) -> torch.Tensor:
    """A thin wrapper around `torch.nn.functional.ctc_loss` that takes *logits*."""
    log_probs = F.log_softmax(logits, dim=2)
    return F.ctc_loss(
        log_probs,
        targets,
        inp_lens,
        tgt_lens,
        reduction="mean",
        zero_infinity=True,
    )

# --------------------------------------------------------------------------- #
#                            Main refinement routine                          #
# --------------------------------------------------------------------------- #
def refine_visual_backbone(
    dataset: HTRDataset,
    backbone: HTRNet,
    num_epochs: int,
    *,
    batch_size: int = 128,
    lr: float = 1e-4,
    main_weight: float = 1.0,
    aux_weight: float = 0.1,
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
    num_epochs: int = 150,
    batch_size: int = 512,
    lr: float = 1e-4,
    num_workers: int = 0,
    weight_decay: float = 1e-4,
    device: torch.device | str = "cuda",
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

    arch_cfg = SimpleNamespace(
        cnn_cfg=[[2, 64], 'M', [3, 128], 'M', [2, 256]],
        head_type="both",          # produces (main_logits, aux_logits)
        rnn_type="gru",
        rnn_layers=2,
        rnn_hidden_size=128,
        flattening="maxpool",
        stn=False,
        feat_dim=None,
    )

    nclasses = len(dataset.character_classes) + 1  # +1 for CTC blank
    net = HTRNet(arch_cfg, nclasses)

    # ── 3. One quick refinement epoch ──────────────────────────────────────
    refine_visual_backbone(
        dataset,
        net,
        num_epochs=100,
        batch_size=128,
        lr=1e-4,
    )
