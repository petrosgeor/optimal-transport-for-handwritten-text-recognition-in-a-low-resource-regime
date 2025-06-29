from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from omegaconf import OmegaConf

from htr_base.utils.htr_dataset import HTRDataset
from htr_base.models import HTRNet, Projector
from alignment.losses import ProjectionLoss
from alignment.features import harvest_backbone_features
from alignment.plot import plot_tsne_embeddings, plot_projector_tsne

cfg_file = Path(__file__).resolve().parents[1] / "config.yaml"
cfg = OmegaConf.load(cfg_file)


def train_projector(
    dataset: HTRDataset,
    backbone: HTRNet,
    projector: Projector | list[Projector],
    num_epochs: int = cfg.projector_epochs,
    batch_size: int = cfg.projector_batch_size,
    lr: float = cfg.projector_lr,
    num_workers: int = cfg.projector_workers,
    weight_decay: float = cfg.projector_weight_decay,
    device: torch.device | str = cfg.device,
    plot_tsne: bool = cfg.plot_tsne,
) -> None:
    """Freeze ``backbone`` and train ``projector`` on cached descriptors."""
    device = torch.device(device)
    backbone = backbone.to(device).eval()
    projs = projector if isinstance(projector, (list, tuple)) else [projector]
    projs = [p.to(device).train() for p in projs]

    word_embs_cpu = dataset.external_word_embeddings
    if word_embs_cpu is None:
        raise RuntimeError("dataset.external_word_embeddings is required")

    probs_attr = getattr(dataset, "external_word_probs", None)
    if probs_attr is not None and len(probs_attr) > 0:
        word_probs_cpu = (
            torch.tensor(probs_attr, dtype=torch.float)
            if isinstance(probs_attr, list)
            else probs_attr.float()
        )
    else:
        v = word_embs_cpu.size(0)
        print("Warning: `dataset.external_word_probs` not found or is empty. Using uniform distribution.")
        word_probs_cpu = torch.full((v,), 1.0 / v)

    word_embs = word_embs_cpu.to(device)
    word_probs = word_probs_cpu.to(device)

    if plot_tsne:
        plot_tsne_embeddings(dataset, backbone=backbone, save_path="tests/figures/tsne_backbone.png", device=device)

    print("Harvesting image descriptors from the backbone...")
    feats_all, aligned_all = harvest_backbone_features(
        dataset,
        backbone,
        batch_size=64,
        num_workers=num_workers,
        device=device,
    )
    assert feats_all.shape[1] == backbone.output_dim, "Descriptor dimension mismatch after harvesting features."

    proj_loader = DataLoader(
        TensorDataset(feats_all, aligned_all),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )

    criterion = ProjectionLoss().to(device)
    print("Starting projector training...")
    for idx, proj in enumerate(projs):
        optimiser = optim.AdamW(proj.parameters(), lr=lr, weight_decay=weight_decay)
        for epoch in range(1, num_epochs + 1):
            running_loss: float = 0.0
            for feats_cpu, align_cpu in proj_loader:
                feats = feats_cpu.to(device)
                align = align_cpu.to(device)
                assert (0 <= align).all() and (align < len(dataset.external_words)).all(), "Alignment indices out of bounds."
                assert torch.isfinite(feats).all(), "Non-finite values detected in features fed to the projector."
                pred = proj(feats)
                loss = criterion.forward(pred, word_embs, align, word_probs)
                assert torch.isfinite(loss), f"Loss is not finite ({loss.item()}). Aborting."
                loss.backward()
                grad_ok = all(torch.isfinite(p.grad).all() for p in proj.parameters() if p.grad is not None)
                assert grad_ok, "gradient explosion in projector - contains NaN/Inf"
                torch.nn.utils.clip_grad_norm_(proj.parameters(), max_norm=1.0)
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)
                running_loss += loss.item()
            for p in proj.parameters():
                assert torch.isfinite(p).all(), "Parameter blow-up detected in projector."
        proj.eval()
        with torch.no_grad():
            proj_vecs = proj(feats_all.to(device)).cpu()
        if plot_tsne:
            plot_projector_tsne(proj_vecs, dataset, save_path=f"tests/figures/tsne_projections_{idx}.png")
    print("[Projector] training complete ✔")

