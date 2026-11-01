"""Checkpoint parsing and backbone initialization helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from htr_base.models import HTRNet
from htr_base.utils.vocab import load_vocab

# Output layers have one row per character, so their size depends on the charset.
CTC_HEAD_PREFIXES = ("top.fnl.1.", "top.cnn.1.")
PHOC_HEAD_PREFIX = "phoc_head."


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


def _head_rows(
    key: str, src_c2i: dict, tgt_c2i: dict, phoc_blocks: int
) -> tuple[torch.Tensor, int] | None:
    """Pick the checkpoint rows that an output layer keeps for a smaller charset.

    A checkpoint trained on a larger charset is shrunk by keeping the rows of
    the target characters, in target order.

    Args:
        key (str): State-dict key of the tensor.
        src_c2i (dict): Character-to-index map of the checkpoint charset.
        tgt_c2i (dict): Character-to-index map of the target charset. Every
            character must also be in src_c2i.
        phoc_blocks (int): Number of PHOC segments, i.e. sum(phoc_levels).

    Returns:
        tuple[torch.Tensor, int] | None: (checkpoint rows to keep, row count the
        checkpoint tensor must have), or None if key is not an output layer.
    """
    tgt_chars = sorted(tgt_c2i, key=tgt_c2i.get)
    if key.startswith(CTC_HEAD_PREFIXES):
        # Row 0 is the CTC blank, row c2i[ch] is character ch.
        rows = [0] + [src_c2i[ch] for ch in tgt_chars]
        return torch.tensor(rows), len(src_c2i) + 1
    if key.startswith(PHOC_HEAD_PREFIX):
        # One block per PHOC segment, each with one row per character (no blank).
        rows = [
            block * len(src_c2i) + src_c2i[ch] - 1
            for block in range(phoc_blocks)
            for ch in tgt_chars
        ]
        return torch.tensor(rows), len(src_c2i) * phoc_blocks
    return None


def maybe_load_backbone(backbone: HTRNet, cfg, c2i: dict) -> None:
    """Load a pretrained checkpoint, shrinking its output layers to the target charset.

    Parameters whose names exist in backbone.state_dict() and whose shapes match
    are loaded as-is. The output layers (both CTC heads and the PHOC head) of a
    checkpoint pretrained on the ALL_DATASETS charset are shrunk to the rows of
    the characters in c2i. Any other mismatched tensor is skipped.

    Args:
        backbone (HTRNet): Backbone model to initialize.
        cfg: Training config with preload flags and checkpoint path.
        c2i (dict): Character-to-index map the backbone's output layers use.
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

    # Pretraining always uses the union charset.
    src_c2i, _ = load_vocab("ALL_DATASETS")
    phoc_blocks = sum(backbone.phoc_levels or [])
    can_shrink = set(c2i) <= set(src_c2i)

    # Keep keys present in model with matching tensor shape; shrink output layers
    filtered = {}
    shrunk = []
    shape_mismatch = []
    checkpoint_only = []
    for k, v in sd.items():
        if k not in model_sd:
            checkpoint_only.append(k)
            continue
        if hasattr(v, "shape") and getattr(model_sd[k], "shape", None) == v.shape:
            filtered[k] = v
            continue
        head = _head_rows(k, src_c2i, c2i, phoc_blocks) if can_shrink else None
        if head is not None:
            rows, expected_src_rows = head
            # Checkpoint was trained on ALL_DATASETS
            checkpoint_ok = v.shape[0] == expected_src_rows
            # Same layer input size (e.g. RNN hidden size, feat_dim)
            inputs_ok = v.shape[1:] == model_sd[k].shape[1:]
            # Model was built for the c2i charset
            model_ok = model_sd[k].shape[0] == len(rows)
            if checkpoint_ok and inputs_ok and model_ok:
                filtered[k] = v[rows]
                shrunk.append(k)
                continue
        shape_mismatch.append(k)

    # Load the compatible subset
    incompatible = backbone.load_state_dict(filtered, strict=False)

    # User-friendly summary
    matched = len(filtered)
    total_ckpt = len(sd)
    print(
        f"[Init] Loaded {matched}/{total_ckpt} tensors from {path} "
        f"({len(shrunk)} output-layer tensors shrunk to the target charset). "
        f"Skipped {len(shape_mismatch)} shape-mismatched keys, "
        f"{len(checkpoint_only)} checkpoint-only keys, "
        f"{len(incompatible.missing_keys)} missing model keys, and "
        f"{len(incompatible.unexpected_keys)} unexpected keys."
    )
    if shape_mismatch:
        print(f"[Init] Shape-mismatched keys left at random init: {shape_mismatch}")
