"""Plotting utilities for alignment module."""

from typing import Optional, Tuple, List, Sequence
from pathlib import Path

import os
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from omegaconf import OmegaConf
from types import SimpleNamespace

from htr_base.utils.htr_dataset import HTRDataset
from htr_base.models import HTRNet
from htr_base.utils.vocab import load_vocab
from alignment.features import harvest_backbone_features

cfg = OmegaConf.load("alignment/config.yaml")


def plot_dataset_augmentations(dataset: HTRDataset, save_path: str) -> None:
    """Save a figure showing three images and their augmentations side by side.

    Parameters
    ----------
    dataset : HTRDataset
        Dataset providing images and augmentation transforms.
    save_path : str
        Where to write the PNG figure.
    """

    if len(dataset) < 3:
        raise ValueError("dataset must contain at least three items")

    if getattr(dataset, "transforms", None) is None:
        raise ValueError("dataset.transforms must not be None")

    orig_transforms = dataset.transforms

    indices = random.sample(range(len(dataset)), 3)

    # Load original images with transforms disabled
    dataset.transforms = None
    originals = [dataset[i][0].squeeze().cpu().numpy() for i in indices]

    # Load augmented versions
    dataset.transforms = orig_transforms
    augments = [dataset[i][0].squeeze().cpu().numpy() for i in indices]

    # Restore dataset transforms
    dataset.transforms = orig_transforms

    fig, axes = plt.subplots(3, 2, figsize=(6, 9))
    for row, (orig, aug) in enumerate(zip(originals, augments)):
        axes[row, 0].imshow(orig, cmap="gray")
        axes[row, 0].axis("off")
        axes[row, 1].imshow(aug, cmap="gray")
        axes[row, 1].axis("off")
    axes[0, 0].set_title("original")
    axes[0, 1].set_title("augmented")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)


def plot_tsne_embeddings(
    dataset: HTRDataset,
    backbone: HTRNet,
    save_path: str,
    *,
    device: torch.device = torch.device(cfg.device),
) -> None:
    """Generate a coloured t-SNE plot of backbone embeddings and save it.

    Features and current alignment labels are harvested from ``dataset`` using
    ``backbone``. t-SNE then projects the descriptors to 2‑D and the scatter
    plot colours samples in blue when ``aligned == 1`` and black otherwise.

    Parameters
    ----------
    dataset : HTRDataset
        Dataset instance providing the images.
    backbone : HTRNet
        The visual backbone model to extract embeddings from.
    save_path : str
        Path where the generated t-SNE plot (PNG image) will be saved.
    """
    # Determine device
    device = torch.device(device)

    # print(f"Harvesting features for t-SNE plot on device: {device}")
    features, aligned = harvest_backbone_features(dataset, backbone, device=device)

    # print(f"Performing t-SNE transformation on {features.shape[0]} samples...")
    # Ensure features are on CPU and are NumPy arrays for scikit-learn
    features_np = features.cpu().numpy()

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(features_np)

    # print("t-SNE transformation complete.")

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = ["blue" if int(a.item()) == 1 else "black" for a in aligned]
    ax.scatter(tsne_results[:, 0], tsne_results[:, 1], s=5, c=colors)
    ax.set_title("t-SNE projection of backbone embeddings")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")

    print(f"Saving t-SNE plot to {save_path}...")
    # Ensure the directory exists
    if os.path.dirname(save_path): # Check if there is a directory part
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    # print("Plot saved successfully.")


def plot_projector_tsne(
    projections: torch.Tensor, dataset: HTRDataset, save_path: str
) -> None:
    """Plot t-SNE of projector outputs and word embeddings.

    Parameters
    ----------
    projections : torch.Tensor
        Output of the projector with shape ``(N, E)``.
    dataset : HTRDataset
        Provides ``external_word_embeddings`` of shape ``(V, E)``.
    save_path : str
        Destination path for the PNG figure.
    """

    word_embs = dataset.external_word_embeddings
    if word_embs is None:
        raise RuntimeError("dataset.external_word_embeddings is required")

    all_vecs = torch.cat([projections, word_embs], dim=0).cpu().numpy()
    perplexity = min(30, max(1, all_vecs.shape[0] // 3))
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    )
    tsne_res = tsne.fit_transform(all_vecs)
    n_proj = projections.size(0)
    proj_2d = tsne_res[:n_proj]
    word_2d = tsne_res[n_proj:]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(proj_2d[:, 0], proj_2d[:, 1], s=5, c="blue", label="projections")
    ax.scatter(word_2d[:, 0], word_2d[:, 1], s=20, c="black", label="words")
    ax.legend()
    ax.set_title("t-SNE of projector outputs")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_pretrained_backbone_tsne(dataset: HTRDataset, n_samples: int, save_path: str) -> None:
    """Plot t-SNE embeddings from the pretrained backbone.

    Parameters
    ----------
    dataset : HTRDataset
        Dataset instance providing images and alignment labels.
    n_samples : int
        Number of random samples to visualise.
    save_path : str
        Path where the PNG figure will be saved.
    """
    cfg = OmegaConf.load(Path(__file__).with_name("config.yaml"))
    arch_cfg = SimpleNamespace(**cfg["architecture"])
    c2i, _ = load_vocab()
    backbone = HTRNet(arch_cfg, nclasses=len(c2i) + 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(
        "htr_base/saved_models/pretrained_backbone.pt", map_location=device
    )
    backbone.load_state_dict(state)
    backbone.eval().to(device)

    indices = random.sample(range(len(dataset)), min(len(dataset), n_samples))
    orig_transforms = getattr(dataset, "transforms", None)
    dataset.transforms = None
    
    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    
    feats_buf: list[torch.Tensor] = []
    aligned_buf: list[torch.Tensor] = []
    transcriptions_buf: list[str] = []

    with torch.no_grad():
        for imgs, _txt, aligned in loader:
            imgs = imgs.to(device)
            feats = backbone(imgs, return_feats=True)[-1]
            feats_buf.append(feats.cpu())
            aligned_buf.append(aligned.cpu())
            transcriptions_buf.extend(_txt)

    dataset.transforms = orig_transforms

    feats_all = torch.cat(feats_buf, dim=0)
    aligned_all = torch.cat(aligned_buf, dim=0)

    perplexity = min(30, max(1, feats_all.size(0) // 3))
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    )
    tsne_res = tsne.fit_transform(feats_all.numpy())

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["blue" if int(a.item()) == 1 else "black" for a in aligned_all]
    ax.scatter(tsne_res[:, 0], tsne_res[:, 1], s=5, c=colors)

    for i, txt in enumerate(transcriptions_buf):
        ax.text(tsne_res[i, 0], tsne_res[i, 1], txt, fontsize=6)

    ax.set_title("t-SNE of pretrained backbone")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
