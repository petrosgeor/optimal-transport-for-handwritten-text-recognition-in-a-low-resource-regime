from __future__ import annotations
from pathlib import Path
from typing import List
import os
from omegaconf import OmegaConf

from htr_base.utils.htr_dataset import HTRDataset
from htr_base.models import HTRNet
from alignment.training.backbone import maybe_load_backbone, refine_visual_backbone, cfg
from alignment.training.projector import train_projector
from alignment.features import print_dataset_stats
from alignment.ot_utils import align_more_instances


def alternating_refinement(
    dataset: HTRDataset,
    backbone: HTRNet,
    projectors: List[HRTNet | HTRNet] | List,
    *,
    rounds: int = cfg.alt_rounds,
    backbone_epochs: int = cfg.refine_epochs,
    projector_epochs: int = cfg.projector_epochs,
    refine_kwargs: dict | None = None,
    projector_kwargs: dict | None = None,
    align_kwargs: dict | None = None,
) -> None:
    """Alternately train ``backbone`` and one or more projectors with OT alignment."""

    maybe_load_backbone(backbone, cfg)

    print_dataset_stats(dataset)
    assert isinstance(projectors, (list, tuple)) and len(projectors) > 0, "Projectors must be a non-empty list or tuple."

    if refine_kwargs is None:
        refine_kwargs = {}
    if projector_kwargs is None:
        projector_kwargs = {}
    if align_kwargs is None:
        align_kwargs = {}
    align_kwargs.setdefault("batch_size", cfg.align_batch_size)
    align_kwargs.setdefault("device", cfg.align_device)
    align_kwargs.setdefault("reg", cfg.align_reg)
    align_kwargs.setdefault("unbalanced", cfg.align_unbalanced)
    align_kwargs.setdefault("reg_m", cfg.align_reg_m)
    align_kwargs.setdefault("k", cfg.align_k)
    align_kwargs.setdefault("agree_threshold", cfg.agree_threshold)

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
            for proj in projectors:
                for param in proj.parameters():
                    param.requires_grad_(True)

            backbone_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
            projectors_trainable = sum(p.numel() for proj in projectors for p in proj.parameters() if p.requires_grad)
            assert (
                backbone_trainable == 0 and projectors_trainable > 0
            ) or (
                backbone_trainable > 0 and projectors_trainable == 0
            ), "Exactly one module family (backbone or projectors) should be trainable."

            print(f"[Round {r + 1}/{rounds}] Training projector...")
            if projector_epochs > 0:
                _probs_backup = None
                if isinstance(getattr(dataset, "external_word_probs", None), list):
                    _probs_backup = dataset.external_word_probs
                    dataset.external_word_probs = torch.tensor(_probs_backup, dtype=torch.float)

                train_projector(
                    dataset,
                    backbone,
                    projectors,
                    num_epochs=projector_epochs,
                    **projector_kwargs,
                )

                if _probs_backup is not None:
                    dataset.external_word_probs = _probs_backup

            for param in backbone.parameters():
                param.requires_grad_(True)
            for proj in projectors:
                for param in proj.parameters():
                    param.requires_grad_(False)

            backbone_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
            projectors_trainable = sum(p.numel() for proj in projectors for p in proj.parameters() if p.requires_grad)
            assert (
                backbone_trainable == 0 and projectors_trainable > 0
            ) or (
                backbone_trainable > 0 and projectors_trainable == 0
            ), "Exactly one module family (backbone or projectors) should be trainable."

        print("[Cycle] Aligning more instances...")
        assert (dataset.aligned != -1).sum() > 0, "Cannot align more instances with zero seeds."
        align_more_instances(dataset, backbone, projectors, **align_kwargs)

