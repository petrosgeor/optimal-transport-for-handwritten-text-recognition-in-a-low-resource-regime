"""Utility functions for aligning dataset instances to external words."""

from typing import Optional

import os
import random
import torch
import numpy as np
import ot
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial import distance

from htr_base.utils.htr_dataset import HTRDataset
from htr_base.models import HTRNet, Projector


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
    device: torch.device = "cuda",
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


def align_more_instances(
    dataset: HTRDataset,
    backbone: HTRNet,
    projector: Projector,
    *,
    batch_size: int = 512,
    device = "cpu",
    reg: float = 0.1,
    unbalanced: bool = False,
    reg_m: float = 1.0,
    sinkhorn_kwargs: Optional[dict] = None,
    k: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Automatically align dataset images to external words using OT.

    Parameters
    ----------
    dataset : HTRDataset
        Dataset providing images and ``external_word_embeddings``.
    backbone : HTRNet
        Visual backbone used to extract per-image features.
    projector : Projector
        Module projecting features to the word embedding space.
    batch_size : int, optional
        Mini-batch size used when harvesting descriptors.
    device : torch.device | str, optional
        Device on which descriptors are processed. Both feature extraction and
        the projector run on this device.
    reg : float, optional
        Entropic regularisation strength for Sinkhorn.
    unbalanced : bool, optional
        If ``True`` uses the unbalanced OT formulation.
    reg_m : float, optional
        Additional unbalanced regularisation parameter.
    sinkhorn_kwargs : dict, optional
        Extra arguments forwarded to the Sinkhorn solver.
    k : int, optional
        How many of the least-moved descriptors to pseudo-label.

    Notes
    -----
    Dataset augmentations are temporarily disabled while collecting
    descriptors and re-enabled afterwards. Both ``backbone`` and
    ``projector`` are returned to training mode once alignment
    completes.

    Returns
    -------
    torch.Tensor
        Transport plan of shape ``(N, V)`` where ``N`` is the number of
        dataset samples and ``V`` the vocabulary size.
    torch.Tensor
        Projected descriptors after OT (``(N, E)``).
    torch.Tensor
        Distance moved by each descriptor (``(N,)``).
    """
    if sinkhorn_kwargs is None:
        sinkhorn_kwargs = {}

    word_embs = dataset.external_word_embeddings  # (V, E)
    if word_embs is None:
        raise RuntimeError("dataset.external_word_embeddings is required")

    device = torch.device(device)

    # Harvest features using the utility function
    # harvest_backbone_features handles backbone.eval(), torch.no_grad(),
    # and moving features to CPU. It also handles dataset transforms.
    feats, _ = harvest_backbone_features(
        dataset,
        backbone,
        batch_size=batch_size,
        num_workers=0, # Match current num_workers in the removed loader
        device=device
    )

    if feats.numel() == 0:
        raise RuntimeError("Dataset yielded no images")

    projector.eval().to(device)
    proj_feats = projector(feats.to(device)).detach().cpu()  # (N, E)
    assert torch.isfinite(proj_feats).all(), (
        "Non-finite values detected in projector features"
    )

    n = proj_feats.size(0)
    v = word_embs.size(0)

    a = np.full((n,), 1.0 / n, dtype=np.float64)
    if hasattr(dataset, "external_word_probs") and dataset.external_word_probs:
        b = np.asarray(dataset.external_word_probs, dtype=np.float64)
        if not unbalanced:
            s = b.sum()
            if s > 0:
                b = b / s
    else:
        b = np.full((v,), 1.0 / v, dtype=np.float64)

    proj_np, plan = calculate_ot_projections(
        a,
        proj_feats.numpy(),
        b,
        word_embs.numpy(),
        reg,
        unbalanced=unbalanced,
        reg_m=reg_m,
        sinkhorn_kwargs=sinkhorn_kwargs,
    )

    plan_torch = torch.from_numpy(plan).float()
    proj_feats_ot = torch.from_numpy(proj_np).float()

    old_aligned = dataset.aligned.clone()
    moved_distance = (proj_feats_ot - proj_feats).norm(p=2, dim=1)
    moved_distance_orig = moved_distance.clone()
    moved_distance[old_aligned != -1] = float("inf")

    # ── assign OT-projected descriptors to closest word embedding ────────────
    dists_to_words = torch.cdist(proj_feats_ot, word_embs)
    nearest_word = dists_to_words.argmin(dim=1)  # (N,)

    # ── pseudo‑label k least-moved instances ────────────────────────────────
    if k > 0:
        valid_idx = torch.nonzero(moved_distance.isfinite(), as_tuple=True)[0]
        if valid_idx.numel() > 0:
            k_sel = min(k, valid_idx.numel())
            sorted_idx = moved_distance[valid_idx].argsort()[:k_sel]
            chosen = valid_idx[sorted_idx]
            dataset.aligned[chosen] = nearest_word[chosen].to(torch.int32)

            # ---- print pseudo‑labelling accuracy ---------------------------------
            correct_new = 0
            for idx in chosen.tolist():
                pred = dataset.external_words[dataset.aligned[idx].item()]
                if pred == dataset.transcriptions[idx].strip():
                    correct_new += 1
            print(
                f"[Align] newly pseudo‑labelled {len(chosen)} items; "
                f"{correct_new} match ground truth"
            )
            if len(chosen) > 0:
                print(
                    f"[Align] round accuracy: {correct_new}/{len(chosen)} "
                    f"({correct_new / len(chosen):.1%})"
                )

    # Restore moved distances for pre-aligned items
    moved_distance[old_aligned != -1] = moved_distance_orig[old_aligned != -1]

    # Ensure already aligned entries remain unchanged
    assert torch.equal(
        dataset.aligned[old_aligned != -1],
        old_aligned[old_aligned != -1],
    ), "Pre-aligned items were modified"

    # Restore training modes
    # dataset.transforms is handled by harvest_backbone_features
    backbone.train()
    projector.train()

    # ---- overall alignment accuracy ---------------------------------------
    aligned_idx = torch.nonzero(dataset.aligned != -1, as_tuple=True)[0]
    correct_total = 0
    for idx in aligned_idx.tolist():
        pred = dataset.external_words[dataset.aligned[idx].item()]
        if pred == dataset.transcriptions[idx].strip():
            correct_total += 1
    if aligned_idx.numel() > 0:
        print(
            f"[Align] cumulative accuracy: {correct_total}/{aligned_idx.numel()} "
            f"({correct_total / aligned_idx.numel():.1%}) – "
            f"{aligned_idx.numel()} of {len(dataset)} samples aligned"
        )


    return plan_torch, proj_feats_ot, moved_distance


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


def plot_tsne_embeddings(dataset: HTRDataset, backbone: HTRNet, save_path: str) -> None:
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

