"""Utility functions for aligning dataset instances to external words."""

from typing import Optional, Tuple, List, Sequence
from pathlib import Path

import os
import random
import torch
import torch.nn as nn
import numpy as np
import ot
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial import distance
from omegaconf import OmegaConf

cfg = OmegaConf.load("alignment/config.yaml")

from htr_base.utils.htr_dataset import HTRDataset
from htr_base.models import HTRNet


def print_dataset_stats(dataset: HTRDataset) -> None:
    """Print basic statistics about a dataset instance.

    Parameters
    ----------
    dataset : HTRDataset
        Dataset to inspect.
    """
    n_samples = len(dataset)
    n_aligned = int((dataset.aligned != -1).sum().item())
    n_external = len(getattr(dataset, "external_words", []))
    vocab_size = len(getattr(dataset, "character_classes", []))
    n_in_dict = 0
    in_dict_pct = 0.0
    if hasattr(dataset, "is_in_dict") and dataset.is_in_dict.numel() > 0:
        n_in_dict = int(dataset.is_in_dict.sum().item())
        in_dict_pct = 100 * float(n_in_dict) / dataset.is_in_dict.numel()

    all_lower = all(t == t.lower() for t in getattr(dataset, "transcriptions", []))
    ext_lower = all(w == w.lower() for w in getattr(dataset, "external_words", []))
    if n_samples > 0:
        avg_len = sum(len(t) for t in dataset.transcriptions) / n_samples
    else:
        avg_len = 0.0

    print("Dataset statistics:")
    print(f"  subset: {dataset.subset}")
    print(f"  samples: {n_samples}")
    print(f"  aligned: {n_aligned}")
    print(f"  external vocab size: {n_external}")
    print(f"  vocabulary size: {vocab_size}")
    print(f"  in-dictionary samples: {n_in_dict}/{n_samples} ({in_dict_pct:.1f}%)")
    print(f"  transcriptions lowercase: {all_lower}")
    print(f"  external words lowercase: {ext_lower}")
    print(f"  avg transcription length: {avg_len:.2f}")


def harvest_backbone_features(
    dataset: HTRDataset,
    backbone: HTRNet,
    *,
    batch_size: int = 64,
    num_workers: int = 0,
    device: torch.device = torch.device(cfg.device),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return image descriptors and alignment info for the whole dataset.

    Dataset augmentations are temporarily disabled while collecting
    descriptors. The backbone is run in evaluation mode on ``device``.

    Parameters
    ----------
    dataset : HTRDataset
        Dataset providing images and alignment information.
    backbone : HTRNet
        Visual encoder used to extract per-image descriptors.
    batch_size : int, optional
        Mini-batch size when forwarding the dataset.
    num_workers : int, optional
        Worker processes used by the ``DataLoader``.
    device : torch.device | str, optional
        Device on which the backbone runs.

    Returns
    -------
    torch.Tensor
        Tensor of descriptors with shape ``(N, D)`` where ``N`` is the
        dataset size.
    torch.Tensor
        Alignment tensor of shape ``(N,)`` copied from the dataset.
    """

    device = torch.device(device)
    backbone = backbone.to(device).eval()

    orig_transforms = getattr(dataset, "transforms", None)
    dataset.transforms = None

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    feats_buf: list[torch.Tensor] = []
    align_buf: list[torch.Tensor] = []

    with torch.no_grad():
        for imgs, _txt, aligned in loader:
            imgs = imgs.to(device)
            feats = backbone(imgs, return_feats=True)[-1]
            if feats.dim() != 2:
                raise RuntimeError(
                    f"Expected (B, feat_dim) descriptors, got {feats.shape}"
                )
            feats_buf.append(feats.cpu())
            align_buf.append(aligned.cpu())

    dataset.transforms = orig_transforms

    feats_all = torch.cat(feats_buf, dim=0)
    aligned_all = torch.cat(align_buf, dim=0)

    return feats_all, aligned_all

def calculate_ot_projections(
    pa: np.ndarray,
    X: np.ndarray,
    pb: np.ndarray,
    Y: np.ndarray,
    reg: float = 0.1,
    *,
    unbalanced: bool = False,
    reg_m: float = 1.0,
    sinkhorn_kwargs: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute OT projections of ``X`` onto ``Y``.

    Parameters
    ----------
    pa : np.ndarray
        Source distribution over ``X`` of shape ``(N,)``.
    X : np.ndarray
        Source features of shape ``(N, D)``.
    pb : np.ndarray
        Target distribution over ``Y`` of shape ``(M,)``.
    Y : np.ndarray
        Target features of shape ``(M, D)``.
    reg : float, optional
        Entropic regularisation.
    unbalanced : bool, optional
        Use unbalanced OT formulation if ``True``.
    reg_m : float, optional
        Unbalanced mass regularisation.
    sinkhorn_kwargs : dict, optional
        Additional arguments for the OT solver.

    Returns
    -------
    projections : np.ndarray
        ``X`` projected in the space of ``Y`` (``(N, D)``).
    plan : np.ndarray
        Optimal transport plan of shape ``(N, M)``.
    """
    if sinkhorn_kwargs is None:
        sinkhorn_kwargs = {}

    M = distance.cdist(X, Y)

    if unbalanced:
        T = ot.unbalanced.sinkhorn_unbalanced(
            pa, pb, M, reg, reg_m, **sinkhorn_kwargs
        )
    else:
        T = ot.sinkhorn(pa, pb, M, reg=reg, **sinkhorn_kwargs)

    row_sum = T.sum(axis=1, keepdims=True)
    inv_row_sum = np.divide(1.0, row_sum, out=np.zeros_like(row_sum), where=row_sum != 0)
    projections = inv_row_sum * (T @ Y)
    assert np.all(np.isfinite(projections)), "Non-finite values in projections"

    return projections, T


def select_uncertain_instances(
    m: int,
    *,
    transport_plan: Optional[np.ndarray] = None,
    dist_matrix: Optional[np.ndarray] = None,
    metric: str = "gap",
) -> np.ndarray:
    """Return indices of the ``m`` most uncertain instances.

    Parameters
    ----------
    m : int
        Number of indices to return.
    transport_plan : np.ndarray, optional
        OT plan of shape ``(N, V)``. Required for ``metric='entropy'``.
    dist_matrix : np.ndarray, optional
        Pre-computed pairwise distances ``(N, V)``. Required for
        ``metric='gap'``.
    metric : str, optional
        Either ``'gap'`` or ``'entropy'`` selecting the uncertainty measure.

    Returns
    -------
    np.ndarray
        Array of ``m`` indices sorted by decreasing uncertainty.
    """
    if metric not in {"gap", "entropy"}:
        raise ValueError("metric must be 'gap' or 'entropy'")

    if m <= 0:
        raise ValueError("m must be positive")

    if metric == "gap":
        if dist_matrix is None:
            raise ValueError("dist_matrix required for metric='gap'")
        assert dist_matrix.ndim == 2, "dist_matrix must be 2D"
        n = dist_matrix.shape[0]
        assert m <= n, "m cannot exceed number of dataset instances"
        sorted_d = np.sort(dist_matrix, axis=1)
        gaps = sorted_d[:, 1] - sorted_d[:, 0]
        order = np.argsort(gaps)
    else:
        if transport_plan is None:
            raise ValueError("transport_plan required for metric='entropy'")
        assert transport_plan.ndim == 2, "transport_plan must be 2D"
        n = transport_plan.shape[0]
        assert m <= n, "m cannot exceed number of dataset instances"
        row_sum = transport_plan.sum(axis=1, keepdims=True)
        p = np.divide(transport_plan, row_sum, out=np.zeros_like(transport_plan), where=row_sum != 0)
        ent = -np.sum(p * np.log(p + 1e-12), axis=1)
        order = np.argsort(-ent)

    return order[:m]

@torch.no_grad()
def align_more_instances(
    dataset: HTRDataset,
    backbone: HTRNet,
    projectors: Sequence[nn.Module],
    *,
    batch_size: int = 512,
    device: str = cfg.device,
    reg: float = 0.1,
    unbalanced: bool = False,
    reg_m: float = 1.0,
    sinkhorn_kwargs: Optional[dict] = None,
    k: int = 0,
    metric: str = "entropy",          # 'gap' or 'entropy'  (assignment certainty)
    agree_threshold: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Use OT to project each projector's descriptors onto the external
    word-embedding space, then pseudo-label the **k most-certain yet-unaligned**
    images. Predictions are combined by majority vote and only accepted when the
    number of agreeing projectors meets ``agree_threshold``.

    Parameters
    ----------
    projectors : sequence of nn.Module
        One or more projector networks producing word-space descriptors.
    metric : {'gap', 'entropy'}
        How certainty is measured:
        * **gap**     – nearest-word margin  (large ⇒ confident)
        * **entropy** – OT row entropy       (small ⇒ confident)

    agree_threshold : int
        Minimum number of votes required for a pseudo-label to be accepted.

    Returns
    -------
    plan_torch   : (N, V) float32  – OT transport plan
    proj_feats_ot: (N, E) float32  – averaged barycentric projections
    moved_dist   : (N,)  float32   – L2 distance projector→projection
    """
    if metric not in {"gap", "entropy"}:
        raise ValueError("metric must be 'gap' or 'entropy'")
    if sinkhorn_kwargs is None:
        sinkhorn_kwargs = {}

    device = torch.device(device)
    word_embs = dataset.external_word_embeddings.to(device)  # (V, E)

    # --------------------------------------------------------- 1. descriptors
    feats_all, aligned_all = harvest_backbone_features(
        dataset, backbone,
        batch_size=batch_size, num_workers=0, device=device
    )                                     # both tensors on CPU
    projs = projectors if isinstance(projectors, Sequence) else [projectors]
    Z_list = []
    dist_list = []
    nearest_list = []
    moved_list = []
    plan = None
    for proj in projs:
        proj.eval().to(device)
        proj_feats = proj(feats_all.to(device)).cpu()        # (N, E)

        N, V = proj_feats.size(0), word_embs.size(0)
        a = np.full((N,), 1.0 / N, dtype=np.float64)
        if getattr(dataset, "external_word_probs", None):
            b = np.asarray(dataset.external_word_probs, dtype=np.float64)
            if not unbalanced:
                b /= b.sum()
        else:
            b = np.full((V,), 1.0 / V, dtype=np.float64)

        proj_np, plan = calculate_ot_projections(
            a, proj_feats.numpy(), b, word_embs.cpu().numpy(),
            reg, unbalanced=unbalanced, reg_m=reg_m,
            sinkhorn_kwargs=sinkhorn_kwargs,
        )
        proj_feats_ot = torch.from_numpy(proj_np).float()
        moved_dist = (proj_feats_ot - proj_feats).norm(p=2, dim=1)           # (N,)
        dist_matrix = torch.cdist(proj_feats_ot, word_embs.cpu())            # (N,V)
        nearest_word = dist_matrix.argmin(dim=1)

        Z_list.append(proj_feats_ot)
        dist_list.append(dist_matrix)
        nearest_list.append(nearest_word)
        moved_list.append(moved_dist)

    Z = torch.stack(Z_list)                             # (M,N,E)
    dist_matrix = torch.stack(dist_list).mean(dim=0)    # (N,V)
    moved_dist = torch.stack(moved_list).mean(dim=0)    # (N,)
    preds = torch.stack(nearest_list)                   # (M,N)
    nearest_word, _ = preds.mode(dim=0)
    counts = (preds == nearest_word.unsqueeze(0)).sum(dim=0)
    plan_torch = torch.from_numpy(plan).float() if plan is not None else torch.empty(0)

    # --------------------------------------------------------- 4. certainty
    # We first obtain all indices ordered *by decreasing uncertainty*,
    # then traverse that list backwards to pick the most-certain rows.
    order_unc = select_uncertain_instances(
        m=len(dataset),
        transport_plan=plan if metric == "entropy" else None,
        dist_matrix=dist_matrix.numpy() if metric == "gap" else None,
        metric=metric,
    )                                         # most-uncertain → least-certain
    order_cert = order_unc[::-1]              # least-uncertain first

    mask_new = (aligned_all == -1).numpy()
    chosen: List[int] = []
    for idx in order_cert:
        if mask_new[idx] and counts[idx].item() >= agree_threshold:
            chosen.append(idx)
            if len(chosen) == min(k, mask_new.sum()):
                break
    chosen_tensor = torch.tensor(chosen, dtype=torch.long)

    # --------------------------------------------------------- 5. label writing
    if chosen:
        dataset.aligned[chosen_tensor] = nearest_word[chosen_tensor].to(torch.int32)

    # --------------------------------------------------------- 6. logging
    if k > 0 and chosen:
        # accuracy on *just* the new items
        correct_new = sum(
            dataset.external_words[dataset.aligned[i].item()] ==
            dataset.transcriptions[i].strip()
            for i in chosen
        )
        print(
            f"[Align] newly pseudo-labelled {len(chosen)} items; "
            f"[Align] round accuracy {correct_new}/{len(chosen)} "
            f"({correct_new/len(chosen):.1%})"
        )

        # 10 random examples
        for idx in random.sample(chosen, min(10, len(chosen))):
            gt   = dataset.transcriptions[idx].strip()
            pred = dataset.external_words[dataset.aligned[idx].item()]
            print(f"[Align] sample: '{gt}' → '{pred}'")

        # stats
        md = moved_dist[chosen_tensor]
        print(f"[Align] mean moved distance: {md.mean():.4f} ± {md.std(unbiased=False):.4f}")

        if metric == "gap":
            top2 = torch.topk(dist_matrix[chosen_tensor], 2, largest=False).values
            gap_vals = (top2[:, 1] - top2[:, 0])
            print(f"[Align] mean NN-gap: {gap_vals.mean():.4f} ± {gap_vals.std(unbiased=False):.4f}")
        else:  # entropy
            row = plan[chosen] / plan[chosen].sum(axis=1, keepdims=True)
            ent = -(row * np.log(row + 1e-12)).sum(axis=1)
            print(f"[Align] mean row entropy: {ent.mean():.4f} ± {ent.std():.4f}")

    # cumulative accuracy
    aligned_idx = torch.nonzero(dataset.aligned != -1, as_tuple=True)[0]
    if aligned_idx.numel():
        correct_tot = sum(
            dataset.external_words[dataset.aligned[i].item()] ==
            dataset.transcriptions[i].strip()
            for i in aligned_idx.tolist()
        )
        print(
            f"[Align] cumulative accuracy: {correct_tot}/{aligned_idx.numel()} "
            f"({correct_tot/aligned_idx.numel():.1%}) – "
            f"{aligned_idx.numel()} of {len(dataset)} samples aligned"
        )

    for proj in projs:
        proj.train()
    backbone.train()
    proj_feats_mean = Z.mean(dim=0)
    return plan_torch, proj_feats_mean, moved_dist


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
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    from htr_base.utils.vocab import load_vocab

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
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, indices),
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    feats_buf: list[torch.Tensor] = []
    aligned_buf: list[torch.Tensor] = []
    with torch.no_grad():
        for imgs, _txt, aligned in loader:
            imgs = imgs.to(device)
            feats = backbone(imgs, return_feats=True)[-1]
            feats_buf.append(feats.cpu())
            aligned_buf.append(aligned.cpu())
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
    ax.set_title("t-SNE of pretrained backbone")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
