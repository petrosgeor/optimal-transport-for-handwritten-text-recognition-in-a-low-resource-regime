# Copyright
from __future__ import annotations
import os, sys, random, pickle
import json
import re
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone
from omegaconf import OmegaConf

CONFIG_PATH = Path(__file__).parent / "alignment_configs" / "pretraining_config.yaml"
CFG = OmegaConf.load(CONFIG_PATH)
os.environ["CUDA_VISIBLE_DEVICES"] = str(CFG.get("gpu_id", 0))

import torch

# ------------------------------------------------------------------
# Configuration parameters for pretraining
# ------------------------------------------------------------------
# Add project root to path for imports
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from htr_base.utils.htr_dataset import PretrainingHTRDataset
from htr_base.models import HTRNet
from htr_base.utils.transforms import aug_transforms
from htr_base.utils.metrics import CER
from htr_base.utils.vocab import load_vocab
from alignment.utils.ctc import (
    encode_for_ctc,
    greedy_ctc_decode,
    beam_search_ctc_decode,
)
from alignment.utils.checks import _assert_grad_finite
from alignment.losses import _ctc_loss_fn, SoftContrastiveLoss  # new class
from htr_base.utils.phoc import build_phoc_description
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_ID = int(CFG.get("gpu_id", 0)) if DEVICE == "cuda" else None
ENABLE_PHOC = bool(CFG.get("enable_phoc", False))
PHOC_LEVELS = tuple(CFG.get("phoc_levels", (1, 2, 3, 4)))
PHOC_W = float(CFG.get("phoc_loss_weight", 0.1))
ENABLE_CONTR = bool(CFG.get("contrastive_enable", False))
CONTR_W = float(CFG.get("contrastive_weight", 0.0))
CONTR_TAU = float(CFG.get("contrastive_tau", 0.07))
CONTR_TTXT = float(CFG.get("contrastive_text_T", 1.0))
# Default pretraining configuration
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "saved_models" / "pretrained_models"
LAST_CHECKPOINT_NAME = "last.pt"
BEST_CHECKPOINT_NAME = "best_cer.pt"
HPARAMS_FILENAME = "hparams.yaml"


def _resolve_path(value, *, required: bool = False, must_exist: bool = False) -> Path | None:
    """Resolve configuration paths relative to repository root.

    Args:
        value: Raw configuration value, usually str or Path.
        required (bool): Raise ValueError if value is empty.
        must_exist (bool): Raise FileNotFoundError if resolved path is missing.

    Returns:
        Path | None: Absolute resolved path, or None when value is empty and
            required=False.
    """
    if value in (None, ""):
        if required:
            raise ValueError(f"Configuration value is required; update {CONFIG_PATH}.")
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    if must_exist and not candidate.exists():
        raise FileNotFoundError(
            f"Resolved path '{candidate}' does not exist. Update {CONFIG_PATH}."
        )
    return candidate


def _validate_run_name(run_name: str) -> str:
    """Validate and normalize a pretraining run name.

    Args:
        run_name (str): Candidate run name from configuration.

    Returns:
        str: Stripped run name that is safe for directory creation.
    """
    normalized = str(run_name).strip()
    if not normalized:
        raise ValueError("save_path must be a non-empty run name.")
    if "/" in normalized or "\\" in normalized:
        raise ValueError("save_path must be a simple run name (no path separators).")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", normalized):
        raise ValueError(
            "save_path may only contain letters, numbers, dot, underscore, and hyphen."
        )
    return normalized


def _load_git_commit() -> str | None:
    """Return the current git commit hash if available.

    Returns:
        str | None: Commit hash or None if unavailable.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
        return commit or None
    except Exception:
        return None


def _atomic_torch_save(payload: dict, destination: Path) -> Path:
    """Write a checkpoint payload atomically.

    Args:
        payload (dict): Serializable checkpoint payload.
        destination (Path): Final checkpoint path.

    Returns:
        Path: Final checkpoint path after atomic replace.
    """
    tmp_path = destination.with_name(f"{destination.name}.tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, destination)
    return destination


def _save_last_checkpoint(
    run_dir: Path,
    *,
    epoch: int,
    net: torch.nn.Module,
    opt: optim.Optimizer,
    best_cer: float,
) -> Path:
    """Save the latest resume checkpoint for pretraining.

    Args:
        run_dir (Path): Run directory where checkpoints are stored.
        epoch (int): Completed 1-based epoch index.
        net (torch.nn.Module): Model to checkpoint.
        opt (optim.Optimizer): Optimizer to checkpoint.
        best_cer (float): Current best CER for this run.

    Returns:
        Path: Path to the saved latest checkpoint.
    """
    checkpoint_path = run_dir / LAST_CHECKPOINT_NAME
    payload = {
        "epoch": int(epoch),
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "best_cer": float(best_cer),
    }
    return _atomic_torch_save(payload, checkpoint_path)


def _save_best_checkpoint(
    run_dir: Path,
    *,
    epoch: int,
    net: torch.nn.Module,
    best_cer: float,
) -> Path:
    """Save the best-CER model checkpoint for pretraining.

    Args:
        run_dir (Path): Run directory where checkpoints are stored.
        epoch (int): Completed 1-based epoch index.
        net (torch.nn.Module): Model to checkpoint.
        best_cer (float): Best CER value at save time.

    Returns:
        Path: Path to the saved best checkpoint.
    """
    checkpoint_path = run_dir / BEST_CHECKPOINT_NAME
    payload = {
        "epoch": int(epoch),
        "model_state_dict": net.state_dict(),
        "best_cer": float(best_cer),
    }
    return _atomic_torch_save(payload, checkpoint_path)


def _write_hparams_yaml(
    run_dir: Path,
    *,
    run_name: str,
    effective_config: dict,
    architecture: dict,
    nclasses: int,
) -> Path:
    """Write run metadata and resolved hyperparameters to YAML.

    Args:
        run_dir (Path): Directory for this pretraining run.
        run_name (str): User-provided run identifier.
        effective_config (dict): Fully resolved training config in use.
        architecture (dict): Resolved architecture dictionary.
        nclasses (int): Number of output classes including CTC blank.

    Returns:
        Path: Path to the written hparams YAML file.
    """
    hparams_path = run_dir / HPARAMS_FILENAME
    payload = {
        "run_name": run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _load_git_commit(),
        "vocab_name": "ALL_DATASETS",
        "nclasses": int(nclasses),
        "architecture": architecture,
        "effective_config": effective_config,
        "checkpoint_policy": {
            "checkpoint_every": int(effective_config["checkpoint_every"]),
            "last_checkpoint_name": LAST_CHECKPOINT_NAME,
            "best_checkpoint_name": BEST_CHECKPOINT_NAME,
        },
        "runtime": {
            "torch_version": str(torch.__version__),
            "torch_cuda_runtime": (None if torch.version.cuda is None else str(torch.version.cuda)),
        },
    }
    OmegaConf.save(config=OmegaConf.create(payload), f=hparams_path)
    return hparams_path


def _load_hparams_yaml(hparams_path: Path) -> dict:
    """Load run hyperparameters from YAML.

    Args:
        hparams_path (Path): Path to an hparams YAML file.

    Returns:
        dict: Parsed hparams content as a plain dictionary.
    """
    loaded = OmegaConf.load(hparams_path)
    return OmegaConf.to_container(loaded, resolve=True)


PRETRAINING_CONFIG = {
    "list_file": CFG.get("list_file", None),
    "train_set_size": int(CFG.get("train_set_size", 30000)),
    "test_set_size": int(CFG.get("test_set_size", 10000)),
    "num_epochs": int(CFG.get("num_epochs", 10000)),
    "batch_size": int(CFG.get("batch_size", 128)),
    "learning_rate": float(CFG.get("learning_rate", 1e-3)),
    "base_path": CFG.get("base_path", None),
    "fixed_size": tuple(CFG.get("fixed_size", (64, 256))),
    "device": DEVICE,
    "use_augmentations": bool(CFG.get("use_augmentations", True)),
    "main_loss_weight": float(CFG.get("main_loss_weight", 1.0)),
    "aux_loss_weight": float(CFG.get("aux_loss_weight", 0.1)),
    # Save pretrained backbone using the union vocabulary across all datasets
    "save_path": CFG.get(
        "save_path",
        "default_run",
    ),
    "checkpoint_every": int(CFG.get("checkpoint_every", 10)),
    "save_backbone": bool(CFG.get("save_backbone", True)),
    "debug_nonfinite_probe": bool(CFG.get("debug_nonfinite_probe", False)),
    "debug_nonfinite_dump": bool(CFG.get("debug_nonfinite_dump", True)),
    "debug_nonfinite_dump_file": str(CFG.get("debug_nonfinite_dump_file", "nonfinite_report.json")),
    "debug_nonfinite_max_batches": CFG.get("debug_nonfinite_max_batches", None),
}
# Architecture configuration for the pretraining backbone
# Loaded from alignment/alignment_configs/pretraining_config.yaml to stay consistent with other scripts
ARCHITECTURE_CONFIG = CFG["architecture"]


class PretrainingRunner:
    """Run synthetic pretraining with explicit setup, train, eval, and save stages."""

    def __init__(self, config: dict | None = None):
        """Initialize runner state and configuration overlays.

        Args:
            config (dict | None): Optional runtime overrides.
        """
        self.runtime_overrides = {} if config is None else dict(config)
        self.config = {**PRETRAINING_CONFIG, **self.runtime_overrides}

        self.run_name: str | None = None
        self.run_dir: Path | None = None
        self.hparams_path: Path | None = None
        self.last_checkpoint_path: Path | None = None
        self.saved_architecture: dict | None = None

        self.list_file: str | None = None
        self.base_path: str | None = None
        self.n_random: int | None = None
        self.num_epochs: int | None = None
        self.batch_size: int | None = None
        self.lr: float | None = None
        self.fixed_size = None
        self.device: str | None = None
        self.use_augmentations: bool | None = None
        self.main_weight: float | None = None
        self.aux_weight: float | None = None
        self.save_backbone: bool | None = None
        self.checkpoint_every: int | None = None
        self.debug_nonfinite_probe: bool = False
        self.debug_nonfinite_dump: bool = True
        self.debug_nonfinite_dump_file: str = "nonfinite_report.json"
        self.debug_nonfinite_max_batches: int | None = None

        self.train_set = None
        self.test_set = None
        self.train_loader = None
        self.test_loader = None

        self.c2i = None
        self.i2c = None
        self.nclasses: int | None = None
        self.architecture: dict | None = None
        self.arch = None
        self.net = None
        self.opt = None
        self.contr_loss_fn = None

        self.start_epoch = 0
        self.best_cer = float("inf")
        self._global_step = 0
        self._probe_stopped_early = False

    def run(self) -> Path:
        """Execute the full pretraining workflow.

        Returns:
            Path: Path to the latest checkpoint (`last.pt`) for this run.
        """
        self._prepare_run_paths()
        self._load_or_resolve_config()
        self._build_datasets()
        self._build_model_and_optimizer()
        self._maybe_resume()
        self._write_hparams_once()

        print(f"[Pretraining] Starting training...")
        print(f"[Pretraining] PHOC loss enabled: {ENABLE_PHOC}")
        print(f"[Pretraining] Contrastive loss enabled: {ENABLE_CONTR}")

        assert self.num_epochs is not None
        if self.start_epoch >= self.num_epochs:
            print(
                f"[Pretraining] start_epoch ({self.start_epoch}) >= num_epochs ({self.num_epochs}); "
                "nothing to train."
            )
            assert self.last_checkpoint_path is not None
            return self.last_checkpoint_path

        for epoch in range(self.start_epoch, self.num_epochs):
            metrics = self._train_one_epoch(epoch)
            if self._probe_stopped_early:
                print(
                    "[Pretraining][Debug] Stopping early after "
                    f"{self.debug_nonfinite_max_batches} batches due to debug_nonfinite_max_batches."
                )
                break
            should_eval = ((epoch + 1) % 3 == 0) or (epoch == self.num_epochs - 1)
            if should_eval:
                lr_val = self.opt.param_groups[0]["lr"]
                assert lr_val > 0 and math.isfinite(lr_val), (
                    "learning rate underflow or became non-finite"
                )
                msg = f"[Pretraining] Epoch {epoch + 1:03d}/{self.num_epochs} - Loss: {metrics['avg_loss']:.4f}"
                if ENABLE_PHOC:
                    msg += f" - PHOC: {metrics['avg_phoc_loss']:.4f}"
                if ENABLE_CONTR:
                    msg += f" - Contr: {metrics['avg_contr_loss']:.4f}"
                msg += f" - lr: {lr_val:.2e}"
                print(msg)
                cer = self._evaluate_cer(self.test_loader)
                if self.save_backbone and math.isfinite(cer) and cer < self.best_cer:
                    self.best_cer = cer
                    best_path = _save_best_checkpoint(
                        self.run_dir,
                        epoch=epoch + 1,
                        net=self.net,
                        best_cer=self.best_cer,
                    )
                    print(f"[Pretraining] Updated best CER checkpoint: {best_path}")
                self._decode_random_samples(self.test_set)
            self._maybe_save_last_checkpoint(epoch)

        assert self.last_checkpoint_path is not None
        return self.last_checkpoint_path

    def _prepare_run_paths(self) -> None:
        """Create and validate filesystem paths for this run.
        """
        self.run_name = _validate_run_name(self.config["save_path"])
        self.run_dir = RUNS_ROOT / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.hparams_path = self.run_dir / HPARAMS_FILENAME
        self.last_checkpoint_path = self.run_dir / LAST_CHECKPOINT_NAME

    def _load_or_resolve_config(self) -> None:
        """Resolve final configuration, honoring resume-time drift protection.
        """
        assert self.hparams_path is not None
        assert self.run_name is not None
        debug_keys = {
            "debug_nonfinite_probe",
            "debug_nonfinite_dump",
            "debug_nonfinite_dump_file",
            "debug_nonfinite_max_batches",
        }
        if self.hparams_path.is_file():
            saved_hparams = _load_hparams_yaml(self.hparams_path)
            saved_effective = saved_hparams.get("effective_config", {})
            if saved_effective:
                allowed_runtime_keys = {"device", "gpu_id", *debug_keys}
                ignored = [
                    key for key in self.runtime_overrides.keys() if key not in allowed_runtime_keys
                ]
                if ignored:
                    print(
                        "[Pretraining] Ignoring runtime overrides on resume to avoid drift: "
                        + ", ".join(sorted(ignored))
                    )
                self.config = dict(saved_effective)
                for key in debug_keys:
                    if key in self.runtime_overrides:
                        self.config[key] = self.runtime_overrides[key]
                    else:
                        self.config[key] = PRETRAINING_CONFIG[key]
            self.saved_architecture = saved_hparams.get("architecture")
        else:
            self.saved_architecture = None

        self.config["save_path"] = self.run_name
        self.config["device"] = DEVICE
        self.checkpoint_every = int(self.config.get("checkpoint_every", 10))
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be >= 1")
        self.config["checkpoint_every"] = self.checkpoint_every

        list_file_path = _resolve_path(self.config["list_file"], required=True, must_exist=True)
        self.list_file = str(list_file_path)
        self.n_random = int(self.config.get("train_set_size", self.config.get("n_random", 0)))
        self.num_epochs = int(self.config["num_epochs"])
        self.batch_size = int(self.config["batch_size"])
        assert 0 < self.batch_size <= 1024, "batch_size must be between 1 and 1024"
        self.lr = float(self.config.get("learning_rate", 1e-4))
        assert self.lr > 0, "learning_rate must be positive"
        base_path_cfg = self.config.get("base_path", None)
        self.fixed_size = self.config["fixed_size"]
        assert self.fixed_size[0] > 0 and self.fixed_size[1] > 0, (
            "fixed_size dimensions must be positive"
        )
        self.device = self.config["device"]
        gpu_id = self.config.pop("gpu_id", None)
        if gpu_id is not None:
            print(
                f"[Pretraining] 'gpu_id' in runtime overrides is ignored; configure it in {CONFIG_PATH}"
            )
        self.use_augmentations = self.config.get("use_augmentations", True)
        self.main_weight = self.config.get("main_loss_weight", 1.0)
        self.aux_weight = self.config.get("aux_loss_weight", 0.1)
        self.save_backbone = self.config.get("save_backbone", False)
        self.debug_nonfinite_probe = bool(self.config.get("debug_nonfinite_probe", False))
        self.debug_nonfinite_dump = bool(self.config.get("debug_nonfinite_dump", True))
        self.debug_nonfinite_dump_file = str(
            self.config.get("debug_nonfinite_dump_file", "nonfinite_report.json")
        )
        max_batches_cfg = self.config.get("debug_nonfinite_max_batches", None)
        self.debug_nonfinite_max_batches = (
            None if max_batches_cfg in (None, "") else int(max_batches_cfg)
        )
        if self.debug_nonfinite_max_batches is not None and self.debug_nonfinite_max_batches < 1:
            raise ValueError("debug_nonfinite_max_batches must be >= 1 when set")
        print(f"[Pretraining] Starting with config:")
        print(f"  list_file: {self.list_file}")
        print(f"  train_set_size: {self.n_random}")
        print(f"  epochs: {self.num_epochs}")
        print(f"  batch_size: {self.batch_size}")
        print(f"  learning_rate: {self.lr}")
        print(f"  device: {self.device}")
        print(f"  augmentations: {self.use_augmentations}")
        print(f"  save_backbone: {self.save_backbone}")
        print(f"  run_name: {self.run_name}")
        print(f"  run_dir: {self.run_dir}")
        print(f"  checkpoint_every: {self.checkpoint_every}")
        print(f"  gpu_id: {GPU_ID if GPU_ID is not None else 'CPU'}")
        print(f"  debug_nonfinite_probe: {self.debug_nonfinite_probe}")
        print(f"  debug_nonfinite_dump: {self.debug_nonfinite_dump}")
        print(f"  debug_nonfinite_dump_file: {self.debug_nonfinite_dump_file}")
        print(f"  debug_nonfinite_max_batches: {self.debug_nonfinite_max_batches}")
        if base_path_cfg is None:
            self.base_path = str(list_file_path.parent)
        else:
            self.base_path = str(_resolve_path(base_path_cfg, must_exist=True))

    def _build_datasets(self) -> None:
        """Create train and test datasets and corresponding stats logs.
        """
        transforms = aug_transforms if self.use_augmentations else None
        self.train_set = PretrainingHTRDataset(
            self.list_file,
            fixed_size=self.fixed_size,
            base_path=self.base_path,
            transforms=transforms,
            n_random=self.n_random,
            preload_images=False,
            random_seed=0,
        )
        assert len(self.train_set) > 0, "training dataset is empty"
        assert self.train_set.fixed_size == self.fixed_size, "train_set fixed_size mismatch"
        self.test_set = PretrainingHTRDataset(
            self.list_file,
            fixed_size=self.fixed_size,
            base_path=self.base_path,
            transforms=None,
            n_random=self.config.get("test_set_size", 10000),
            preload_images=False,
            random_seed=1,
        )
        assert self.test_set.fixed_size == self.fixed_size, "test_set fixed_size mismatch"
        print(f"[Pretraining] Dataset size: {len(self.train_set)}")
        print(f"[Pretraining] Test set size: {len(self.test_set)}")

    def _build_model_and_optimizer(self) -> None:
        """Create vocabulary, model, loaders, and optimizer objects.
        """
        self.c2i, self.i2c = load_vocab("ALL_DATASETS")
        assert 0 not in self.c2i.values(), "blank index 0 found in c2i values"

        self.nclasses = len(self.c2i) + 1
        assert (len(self.c2i) + 1) == self.nclasses, "nclasses mismatch with vocab size"
        print(f"[Pretraining] Vocabulary size: {self.nclasses} (including blank)")
        self.architecture = (
            self.saved_architecture
            if isinstance(self.saved_architecture, dict)
            else OmegaConf.to_container(ARCHITECTURE_CONFIG, resolve=True)
        )
        self.arch = SimpleNamespace(**self.architecture)
        self.net = HTRNet(self.arch, nclasses=self.nclasses).to(self.device).train()
        n_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(f"[Pretraining] Network parameters: {n_params:,}")
        n_workers_train = 3
        n_workers_test = 3
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=n_workers_train,
        )
        self.test_loader = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=n_workers_test,
        )
        self.opt = optim.Adam(self.net.parameters(), lr=self.lr)
        self.contr_loss_fn = SoftContrastiveLoss(CONTR_TAU, CONTR_TTXT).to(self.device)

    def _maybe_resume(self) -> None:
        """Resume model and optimizer from the latest checkpoint when present.
        """
        assert self.last_checkpoint_path is not None
        if self.last_checkpoint_path.is_file():
            last_payload = torch.load(self.last_checkpoint_path, map_location=self.device)
            self.net.load_state_dict(last_payload["model_state_dict"])
            if "optimizer_state_dict" in last_payload:
                self.opt.load_state_dict(last_payload["optimizer_state_dict"])
            self.start_epoch = int(last_payload.get("epoch", 0))
            self.best_cer = float(last_payload.get("best_cer", float("inf")))
            print(
                f"[Pretraining] Resuming from {self.last_checkpoint_path} "
                f"(epoch={self.start_epoch}, best_cer={self.best_cer:.4f})"
            )

    def _write_hparams_once(self) -> None:
        """Persist run metadata and effective hyperparameters if missing.
        """
        effective_config = {
            "list_file": self.list_file,
            "base_path": self.base_path,
            "train_set_size": self.n_random,
            "test_set_size": int(self.config.get("test_set_size", 10000)),
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.lr,
            "fixed_size": list(self.fixed_size),
            "device": self.device,
            "gpu_id": GPU_ID,
            "use_augmentations": self.use_augmentations,
            "enable_phoc": ENABLE_PHOC,
            "phoc_levels": list(PHOC_LEVELS),
            "phoc_loss_weight": PHOC_W,
            "contrastive_enable": ENABLE_CONTR,
            "contrastive_weight": CONTR_W,
            "contrastive_tau": CONTR_TAU,
            "contrastive_text_T": CONTR_TTXT,
            "main_loss_weight": self.main_weight,
            "aux_loss_weight": self.aux_weight,
            "save_backbone": self.save_backbone,
            "checkpoint_every": self.checkpoint_every,
            "debug_nonfinite_probe": self.debug_nonfinite_probe,
            "debug_nonfinite_dump": self.debug_nonfinite_dump,
            "debug_nonfinite_dump_file": self.debug_nonfinite_dump_file,
            "debug_nonfinite_max_batches": self.debug_nonfinite_max_batches,
        }
        if not self.hparams_path.exists():
            written_hparams = _write_hparams_yaml(
                self.run_dir,
                run_name=self.run_name,
                effective_config=effective_config,
                architecture=self.architecture,
                nclasses=self.nclasses,
            )
            print(f"[Pretraining] Wrote run metadata to: {written_hparams}")

    def _tensor_stats(self, tensor: torch.Tensor | None) -> dict:
        """Return compact numerical diagnostics for a tensor.

        Args:
            tensor (torch.Tensor | None): Tensor to summarize, with any shape.

        Returns:
            dict: Shape/dtype/device and finite-aware scalar stats.
        """
        if tensor is None:
            return {"present": False}

        detached = tensor.detach()
        numel = int(detached.numel())
        stats = {
            "present": True,
            "shape": list(detached.shape),
            "dtype": str(detached.dtype),
            "device": str(detached.device),
            "numel": numel,
        }
        if numel == 0:
            stats.update(
                {
                    "isfinite_all": True,
                    "finite_count": 0,
                    "finite_ratio": 1.0,
                    "min": None,
                    "max": None,
                    "mean": None,
                    "std": None,
                }
            )
            return stats

        numeric = detached.to(torch.float32)
        finite_mask = torch.isfinite(numeric)
        finite_count = int(finite_mask.sum().item())
        finite_ratio = float(finite_count) / float(numel)
        stats.update(
            {
                "isfinite_all": finite_count == numel,
                "finite_count": finite_count,
                "finite_ratio": finite_ratio,
            }
        )
        if finite_count == 0:
            stats.update({"min": None, "max": None, "mean": None, "std": None})
            return stats

        finite_vals = numeric[finite_mask]
        stats.update(
            {
                "min": float(finite_vals.min().item()),
                "max": float(finite_vals.max().item()),
                "mean": float(finite_vals.mean().item()),
                "std": float(finite_vals.std(unbiased=False).item()),
            }
        )
        return stats

    def _loss_scalar(self, value: torch.Tensor | float | None) -> float | None:
        """Convert a scalar-like value to a Python float when possible.

        Args:
            value (torch.Tensor | float | None): Scalar tensor with shape ()
                or shape (1), Python float, or None.

        Returns:
            float | None: Scalar value or None when unavailable.
        """
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.numel() != 1:
                return None
            return float(value.detach().item())
        return float(value)

    def _build_nonfinite_report(
        self,
        *,
        offending_component: str,
        epoch: int,
        batch_idx: int,
        txts: list[str],
        inp_lens: torch.Tensor | None,
        lengths: torch.Tensor | None,
        targets: torch.Tensor | None,
        main_logits: torch.Tensor | None,
        aux_logits: torch.Tensor | None,
        features: torch.Tensor | None,
        phoc_logits: torch.Tensor | None,
        loss_main: torch.Tensor | None,
        loss_aux: torch.Tensor | None,
        loss_phoc: torch.Tensor | None,
        loss_contr: torch.Tensor | None,
        total_loss: torch.Tensor | None,
    ) -> dict:
        """Build a JSON-serializable non-finite diagnostics payload.

        Args:
            offending_component (str): Name of the first non-finite component.
            epoch (int): Zero-based epoch index.
            batch_idx (int): Zero-based batch index within the epoch.
            txts (list[str]): Raw transcriptions in the mini-batch.
            inp_lens (torch.Tensor | None): CTC input lengths with shape (B,).
            lengths (torch.Tensor | None): CTC target lengths with shape (B,).
            targets (torch.Tensor | None): Flattened CTC targets with shape (sum(L),).
            main_logits (torch.Tensor | None): Main logits with shape (T, B, C).
            aux_logits (torch.Tensor | None): Auxiliary logits with shape (T, B, C).
            features (torch.Tensor | None): Feature matrix with shape (B, D).
            phoc_logits (torch.Tensor | None): PHOC logits with shape (B, P).
            loss_main (torch.Tensor | None): Scalar loss tensor with shape ().
            loss_aux (torch.Tensor | None): Scalar loss tensor with shape ().
            loss_phoc (torch.Tensor | None): Scalar loss tensor with shape ().
            loss_contr (torch.Tensor | None): Scalar loss tensor with shape ().
            total_loss (torch.Tensor | None): Scalar total loss with shape ().

        Returns:
            dict: Structured diagnostics report for logging and debugging.
        """
        target_lengths = lengths.detach().cpu().tolist() if lengths is not None else []
        input_lengths = inp_lens.detach().cpu().tolist() if inp_lens is not None else []
        long_targets = 0
        if target_lengths and input_lengths:
            if len(input_lengths) == len(target_lengths):
                long_targets = sum(
                    1
                    for tgt_len, inp_len in zip(target_lengths, input_lengths)
                    if tgt_len > inp_len
                )
            else:
                max_input = max(input_lengths)
                long_targets = sum(1 for tgt_len in target_lengths if tgt_len > max_input)

        report = {
            "offending_component": offending_component,
            "run_name": self.run_name,
            "global_step": self._global_step,
            "epoch_zero_based": int(epoch),
            "epoch_one_based": int(epoch) + 1,
            "batch_idx": int(batch_idx),
            "device": self.device,
            "gpu_id": GPU_ID,
            "batch_size": len(txts),
            "transcriptions_head": [str(t) for t in txts[:8]],
            "transcription_lengths_head": [len(str(t)) for t in txts[:8]],
            "ctc": {
                "input_length_head": input_lengths[:8],
                "target_length_head": target_lengths[:8],
                "num_targets_longer_than_input": int(long_targets),
                "targets_numel": None if targets is None else int(targets.numel()),
            },
            "losses": {
                "loss_main": self._loss_scalar(loss_main),
                "loss_aux": self._loss_scalar(loss_aux),
                "loss_phoc": self._loss_scalar(loss_phoc),
                "loss_contr": self._loss_scalar(loss_contr),
                "total_loss": self._loss_scalar(total_loss),
            },
            "tensor_stats": {
                "main_logits": self._tensor_stats(main_logits),
                "aux_logits": self._tensor_stats(aux_logits),
                "features": self._tensor_stats(features),
                "phoc_logits": self._tensor_stats(phoc_logits),
                "loss_main": self._tensor_stats(loss_main),
                "loss_aux": self._tensor_stats(loss_aux),
                "loss_phoc": self._tensor_stats(loss_phoc),
                "loss_contr": self._tensor_stats(loss_contr),
                "total_loss": self._tensor_stats(total_loss),
            },
        }
        return report

    def _persist_nonfinite_report(self, report: dict) -> Path:
        """Persist diagnostics as JSON under the current run directory.

        Args:
            report (dict): Report produced by _build_nonfinite_report.

        Returns:
            Path: Output JSON path.
        """
        assert self.run_dir is not None
        out_path = self.run_dir / self.debug_nonfinite_dump_file
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=True)
        return out_path

    def _raise_nonfinite(
        self,
        *,
        offending_component: str,
        epoch: int,
        batch_idx: int,
        txts: list[str],
        inp_lens: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        main_logits: torch.Tensor | None = None,
        aux_logits: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
        phoc_logits: torch.Tensor | None = None,
        loss_main: torch.Tensor | None = None,
        loss_aux: torch.Tensor | None = None,
        loss_phoc: torch.Tensor | None = None,
        loss_contr: torch.Tensor | None = None,
        total_loss: torch.Tensor | None = None,
    ) -> None:
        """Raise a detailed runtime error for the first non-finite component.

        Args:
            offending_component (str): Name of the first non-finite component.
            epoch (int): Zero-based epoch index.
            batch_idx (int): Zero-based batch index.
            txts (list[str]): Batch transcriptions.
            inp_lens (torch.Tensor | None): CTC input lengths with shape (B,).
            lengths (torch.Tensor | None): CTC target lengths with shape (B,).
            targets (torch.Tensor | None): Flattened CTC targets with shape (sum(L),).
            main_logits (torch.Tensor | None): Main CTC logits with shape (T, B, C).
            aux_logits (torch.Tensor | None): Auxiliary CTC logits with shape (T, B, C).
            features (torch.Tensor | None): Feature matrix with shape (B, D).
            phoc_logits (torch.Tensor | None): PHOC logits with shape (B, P).
            loss_main (torch.Tensor | None): Scalar main loss with shape ().
            loss_aux (torch.Tensor | None): Scalar auxiliary loss with shape ().
            loss_phoc (torch.Tensor | None): Scalar PHOC loss with shape ().
            loss_contr (torch.Tensor | None): Scalar contrastive loss with shape ().
            total_loss (torch.Tensor | None): Scalar weighted total loss with shape ().
        """
        report = self._build_nonfinite_report(
            offending_component=offending_component,
            epoch=epoch,
            batch_idx=batch_idx,
            txts=txts,
            inp_lens=inp_lens,
            lengths=lengths,
            targets=targets,
            main_logits=main_logits,
            aux_logits=aux_logits,
            features=features,
            phoc_logits=phoc_logits,
            loss_main=loss_main,
            loss_aux=loss_aux,
            loss_phoc=loss_phoc,
            loss_contr=loss_contr,
            total_loss=total_loss,
        )
        dump_msg = ""
        if self.debug_nonfinite_dump:
            report_path = self._persist_nonfinite_report(report)
            dump_msg = f" Report written to: {report_path}."

        raise RuntimeError(
            "[Pretraining] Non-finite detected in "
            f"'{offending_component}' at epoch={epoch + 1}, batch_idx={batch_idx}, "
            f"global_step={self._global_step}.{dump_msg}"
        )

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        """Train on one epoch and return aggregated scalar metrics.

        Args:
            epoch (int): Zero-based epoch index.

        Returns:
            dict[str, float]: Averaged train losses for this epoch.
        """
        epoch_loss = 0.0
        num_batches = 0
        phoc_epoch_loss = 0.0
        running_contr = 0.0
        for batch_idx, (imgs, txts) in enumerate(self.train_loader):
            if (
                self.debug_nonfinite_probe
                and self.debug_nonfinite_max_batches is not None
                and batch_idx >= self.debug_nonfinite_max_batches
            ):
                self._probe_stopped_early = True
                break

            self._global_step += 1
            imgs = imgs.to(self.device)
            out = self.net(imgs)
            main_logits, aux_logits, features = out[:3]
            assert features.dim() == 2 and features.size(1) == self.arch.feat_dim, (
                "features tensor has wrong dimensions or feature size"
            )
            for name, tens in {
                "main_logits": main_logits,
                "aux_logits": aux_logits,
                "features": features,
            }.items():
                if not torch.isfinite(tens).all():
                    self._raise_nonfinite(
                        offending_component=name,
                        epoch=epoch,
                        batch_idx=batch_idx,
                        txts=list(txts),
                        main_logits=main_logits,
                        aux_logits=aux_logits,
                        features=features,
                    )

            targets, lengths = encode_for_ctc(list(txts), self.c2i, device=imgs.device)
            assert targets.max() < self.nclasses, "target contains index out of nclasses range"
            inp_lens = torch.full(
                (imgs.size(0),), main_logits.size(0), dtype=torch.int32, device=self.device
            )

            loss_main = _ctc_loss_fn(main_logits, targets, inp_lens, lengths)
            loss_aux = _ctc_loss_fn(aux_logits, targets, inp_lens, lengths)

            loss_contr = torch.tensor(0.0, device=self.device)
            if ENABLE_CONTR:
                loss_contr = self.contr_loss_fn(features, targets, lengths.cpu())
                assert loss_contr >= 0 and torch.isfinite(loss_contr), (
                    "Contrastive loss is negative or non-finite"
                )

            if ENABLE_PHOC:
                phoc_logits = out[-1]
                phoc_targets = (
                    build_phoc_description(list(txts), self.c2i, levels=PHOC_LEVELS)
                    .float()
                    .to(self.device)
                )
                assert phoc_logits.shape == phoc_targets.shape, "shape mismatch"
                if not torch.isfinite(phoc_logits).all():
                    self._raise_nonfinite(
                        offending_component="phoc_logits",
                        epoch=epoch,
                        batch_idx=batch_idx,
                        txts=list(txts),
                        inp_lens=inp_lens,
                        lengths=lengths,
                        targets=targets,
                        main_logits=main_logits,
                        aux_logits=aux_logits,
                        features=features,
                        phoc_logits=phoc_logits,
                    )
                loss_phoc = torch.nn.functional.binary_cross_entropy_with_logits(
                    phoc_logits, phoc_targets
                )
                if not torch.isfinite(loss_phoc).all():
                    self._raise_nonfinite(
                        offending_component="loss_phoc",
                        epoch=epoch,
                        batch_idx=batch_idx,
                        txts=list(txts),
                        inp_lens=inp_lens,
                        lengths=lengths,
                        targets=targets,
                        main_logits=main_logits,
                        aux_logits=aux_logits,
                        features=features,
                        phoc_logits=phoc_logits,
                        loss_main=loss_main,
                        loss_aux=loss_aux,
                        loss_phoc=loss_phoc,
                        loss_contr=loss_contr,
                    )
            else:
                phoc_logits = None
                loss_phoc = torch.tensor(0.0, device=self.device)

            for component_name, component_value in (
                ("loss_main", loss_main),
                ("loss_aux", loss_aux),
                ("loss_contr", loss_contr),
            ):
                if not torch.isfinite(component_value).all():
                    self._raise_nonfinite(
                        offending_component=component_name,
                        epoch=epoch,
                        batch_idx=batch_idx,
                        txts=list(txts),
                        inp_lens=inp_lens,
                        lengths=lengths,
                        targets=targets,
                        main_logits=main_logits,
                        aux_logits=aux_logits,
                        features=features,
                        phoc_logits=phoc_logits,
                        loss_main=loss_main,
                        loss_aux=loss_aux,
                        loss_phoc=loss_phoc,
                        loss_contr=loss_contr,
                    )

            total_loss = (
                self.main_weight * loss_main
                + self.aux_weight * loss_aux
                + PHOC_W * loss_phoc
                + CONTR_W * loss_contr
            )
            if not torch.isfinite(total_loss).all():
                self._raise_nonfinite(
                    offending_component="total_loss",
                    epoch=epoch,
                    batch_idx=batch_idx,
                    txts=list(txts),
                    inp_lens=inp_lens,
                    lengths=lengths,
                    targets=targets,
                    main_logits=main_logits,
                    aux_logits=aux_logits,
                    features=features,
                    phoc_logits=phoc_logits,
                    loss_main=loss_main,
                    loss_aux=loss_aux,
                    loss_phoc=loss_phoc,
                    loss_contr=loss_contr,
                    total_loss=total_loss,
                )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
            _assert_grad_finite(self.net, "net", include_param_names=True)
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            epoch_loss += total_loss.item()
            phoc_epoch_loss += loss_phoc.item()
            running_contr += loss_contr.item()
            num_batches += 1

        return {
            "avg_loss": epoch_loss / max(1, num_batches),
            "avg_phoc_loss": phoc_epoch_loss / max(1, num_batches),
            "avg_contr_loss": running_contr / max(1, num_batches),
        }

    def _evaluate_cer(self, loader) -> float:
        """Compute and print CER for the provided loader.

        Args:
            loader: DataLoader to evaluate.

        Returns:
            float: Character Error Rate or NaN when dataset is empty.
        """
        if len(loader.dataset) == 0:
            print("[Eval] Test set is empty, skipping CER evaluation.")
            return float("nan")
        self.net.eval()
        metric = CER()
        with torch.no_grad():
            for imgs, trans in loader:
                imgs = imgs.to(self.device)
                logits = self.net(imgs, return_feats=False)[0]
                preds = greedy_ctc_decode(logits, self.i2c)
                for pred, target in zip(preds, trans):
                    metric.update(pred.strip(), target.strip())
        self.net.train()
        score = metric.score()
        print(f"[Eval] CER: {score:.4f}")
        return score

    def _decode_random_samples(self, ds) -> None:
        """Print model predictions for up to ten random dataset samples.

        Args:
            ds: Dataset to sample from.
        """
        self.net.eval()
        with torch.no_grad():
            assert len(ds) > 0, "empty dataset for random samples"
            indices = random.sample(range(len(ds)), min(10, len(ds)))
            imgs, gts = zip(*(ds[i] for i in indices))
            batch = torch.stack(imgs, dim=0).to(self.device)
            logits = self.net(batch, return_feats=False)[0]
            greedy_preds = greedy_ctc_decode(logits, self.i2c)
            beam_preds = beam_search_ctc_decode(logits, self.i2c, beam_width=5)
            for gt, greedy, beam in zip(gts, greedy_preds, beam_preds):
                print(f"GT: '{gt.strip()}' | greedy: '{greedy}' | beam5: '{beam}'")
        self.net.train()

    def _maybe_save_last_checkpoint(self, epoch: int) -> None:
        """Persist last checkpoint according to configured save cadence.

        Args:
            epoch (int): Zero-based epoch index.
        """
        should_save = ((epoch + 1) % self.checkpoint_every == 0) or (epoch + 1 == self.num_epochs)
        if self.save_backbone and should_save:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            assert os.access(self.run_dir, os.W_OK), "run_dir is not writable"
            last_path = _save_last_checkpoint(
                self.run_dir,
                epoch=epoch + 1,
                net=self.net,
                opt=self.opt,
                best_cer=self.best_cer,
            )
            print(f"[Pretraining] Saved latest checkpoint: {last_path}")


def main(config: dict | None = None) -> Path:
    """Train an HTRNet on a synthetic list using the ALL_DATASETS charset.

    Args:
        config (dict | None): Optional overrides for pretraining settings.

    Returns:
        Path: Path to the latest checkpoint (`last.pt`) for the selected run.
    """
    runner = PretrainingRunner(config=config)
    return runner.run()


if __name__ == "__main__":
    main()
