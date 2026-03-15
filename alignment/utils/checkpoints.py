"""Checkpoint parsing and backbone initialization helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from htr_base.models import HTRNet


def _is_raw_state_dict(candidate: dict) -> bool:
    """Check whether a dictionary looks like a raw tensor state dict.

    Args:
        candidate (dict): Dictionary to validate.

    Returns:
        bool: True when all values are torch.Tensor objects.
    """
    return bool(candidate) and all(isinstance(v, torch.Tensor) for v in candidate.values())


def _extract_state_dict(ckpt: object, checkpoint_path: str | Path) -> dict:
    """Extract model weights from supported checkpoint formats.

    Supported formats:
    - raw state dict: {name: tensor, ...}
    - wrapped dict: {"state_dict": {...}}
    - pretraining dict: {"model_state_dict": {...}}

    Args:
        ckpt (object): Checkpoint object returned by torch.load.
        checkpoint_path (str | Path): Path used only for readable error messages.

    Returns:
        dict: State dict mapping parameter names to tensors.
    """
    if isinstance(ckpt, dict):
        if isinstance(ckpt.get("state_dict"), dict):
            return ckpt["state_dict"]
        if isinstance(ckpt.get("model_state_dict"), dict):
            return ckpt["model_state_dict"]
        if _is_raw_state_dict(ckpt):
            return ckpt
        found = ", ".join(sorted(str(k) for k in ckpt.keys()))
        raise ValueError(
            "Unsupported checkpoint format for "
            f"{checkpoint_path}. Expected one of: raw state dict, "
            "{'state_dict': ...}, or {'model_state_dict': ...}. "
            f"Found top-level keys: [{found}]"
        )

    if hasattr(ckpt, "state_dict"):
        maybe_sd = ckpt.state_dict()
        if isinstance(maybe_sd, dict) and _is_raw_state_dict(maybe_sd):
            return maybe_sd

    raise ValueError(
        "Unsupported checkpoint object type for "
        f"{checkpoint_path}: {type(ckpt).__name__}. "
        "Expected dict-like checkpoint or object exposing a tensor state_dict()."
    )


def maybe_load_backbone(backbone: HTRNet, cfg) -> None:
    """Load a checkpoint while skipping incompatible head layers.

    Only parameters whose names exist in backbone.state_dict() AND whose
    tensor shapes match exactly are loaded. This allows reusing the backbone
    even when the vocabulary size (CTC head) or PHOC head configuration differs
    from the checkpoint.

    Args:
        backbone (HTRNet): Backbone model to initialize.
        cfg: Training config with preload flags and checkpoint path.
    """
    if not getattr(cfg, "load_pretrained_backbone", False):
        return

    path = cfg.pretrained_backbone_path
    # Load checkpoint to CPU; caller can move model to device afterwards
    ckpt = torch.load(path, map_location="cpu")

    # Resolve the actual state_dict inside various checkpoint formats
    sd = _extract_state_dict(ckpt, path)

    model_sd = backbone.state_dict()

    # Handle common prefixes from DDP or wrapper modules
    def _maybe_strip_prefix(d: dict, prefix: str) -> dict:
        """Strip a prefix from matching state-dict keys.

        Args:
            d (dict): Source state dict.
            prefix (str): Prefix to remove from key names.

        Returns:
            dict: State dict with prefix removed where applicable.
        """
        if any(k.startswith(prefix) for k in d.keys()):
            return {(k[len(prefix) :] if k.startswith(prefix) else k): v for k, v in d.items()}
        return d

    # Try progressively stripping known prefixes to maximise matches
    for pref in ("module.", "backbone."):
        if not (set(sd.keys()) & set(model_sd.keys())):
            sd = _maybe_strip_prefix(sd, pref)

    # Keep only keys present in model with matching tensor shape
    filtered = {}
    shape_mismatch = []
    checkpoint_only = []
    for k, v in sd.items():
        if k in model_sd:
            if hasattr(v, "shape") and getattr(model_sd[k], "shape", None) == v.shape:
                filtered[k] = v
            else:
                shape_mismatch.append(k)
        else:
            checkpoint_only.append(k)

    # Load the compatible subset
    incompatible = backbone.load_state_dict(filtered, strict=False)

    # User-friendly summary
    matched = len(filtered)
    total_ckpt = len(sd)
    print(
        f"[Init] Loaded {matched}/{total_ckpt} tensors from {path}. "
        f"Skipped {len(shape_mismatch)} shape-mismatched keys, "
        f"{len(checkpoint_only)} checkpoint-only keys, "
        f"{len(incompatible.missing_keys)} missing model keys, and "
        f"{len(incompatible.unexpected_keys)} unexpected keys."
    )
