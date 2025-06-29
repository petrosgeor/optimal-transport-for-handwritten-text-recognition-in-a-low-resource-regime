"""Backwards-compatible imports for training utilities."""

from alignment.training import (
    maybe_load_backbone,
    refine_visual_backbone,
    train_projector,
    alternating_refinement,
    cfg,
)

from alignment.features import harvest_backbone_features, print_dataset_stats
from alignment.ot_utils import align_more_instances
from alignment.plot import plot_tsne_embeddings, plot_projector_tsne

__all__ = [
    "maybe_load_backbone",
    "refine_visual_backbone",
    "train_projector",
    "alternating_refinement",
    "harvest_backbone_features",
    "print_dataset_stats",
    "align_more_instances",
    "plot_tsne_embeddings",
    "plot_projector_tsne",
    "cfg",
]

if __name__ == "__main__":
    from types import SimpleNamespace
    from pathlib import Path
    from htr_base.utils.htr_dataset import HTRDataset
    from htr_base.models import HTRNet, Projector
    from htr_base.utils.transforms import aug_transforms

    proj_root = Path(__file__).resolve().parents[1]
    gw_folder = proj_root / "htr_base" / "data" / "GW" / "processed_words"
    if not gw_folder.exists():
        raise RuntimeError(
            "GW processed dataset not found – run the dataset preparation step before executing this dummy test."
        )

    class DummyCfg:
        k_external_words = 200
        n_aligned = cfg.n_aligned

    dataset = HTRDataset(
        str(gw_folder),
        subset="all",
        fixed_size=(64, 256),
        transforms=aug_transforms,
        config=DummyCfg(),
    )

    arch = SimpleNamespace(**cfg["architecture"])
    backbone = HTRNet(arch, nclasses=len(dataset.character_classes) + 1)
    maybe_load_backbone(backbone, cfg)
    projectors = [Projector(arch.feat_dim, dataset.word_emb_dim) for _ in range(cfg.ensemble_size)]

    alternating_refinement(dataset, backbone, projectors)

