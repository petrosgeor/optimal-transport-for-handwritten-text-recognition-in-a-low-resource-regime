# Optimal Transport for Handwritten Text Recognition in a Low-Resource Regime

This repository accompanies the paper *"Optimal Transport for Handwritten Text Recognition in a Low-Resource Regime"*. It implements an alternating pipeline that grows supervision with minimal ground truth by combining CTC fine-tuning, optimal-transport-based projection, and pseudo-labelling.

## Table of Contents
- [Abstract](#abstract)
- [Repository Highlights](#repository-highlights)
- [Pipeline Overview](#pipeline-overview)
- [Guardrails and Assumptions](#guardrails-and-assumptions)
- [Getting Started](#getting-started)
- [Configuration at a Glance](#configuration-at-a-glance)
- [Running the Pipeline](#running-the-pipeline)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Evaluation and Reporting](#evaluation-and-reporting)
- [Logging and Artifacts](#logging-and-artifacts)
- [Project Layout](#project-layout)
- [API Reference](#api-reference)
- [Experiments and Tests](#experiments-and-tests)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Abstract
We iteratively project backbone descriptors into a word-embedding space via optimal transport, selectively pseudo-label the most confident unlabeled images, and refine a CTC-based handwritten text recogniser. The procedure delivers competitive CER/WER in low-resource settings while never exposing the model to the ground-truth transcripts of unaligned samples.

## Repository Highlights
- Alternating optimisation loop that cycles between CTC refinement, projector training, and OT-based pseudo-labelling.
- Dataset utilities that track aligned/unaligned indices, word priors, and embeddings for large vocabularies.
- Guardrails that prevent leakage: the backbone trains only on aligned items, pseudo-labelling draws exclusively from unaligned instances, and evaluation uses a clean test split with `n_aligned = 0`.
- Self-contained configuration through `OmegaConf` YAML files and reproducible experiment scripts in `alignment/`.
- Visualisation helpers (t-SNE plots, augmentations) and metric reporting (CER/WER) aligned with the paper.

## Pipeline Overview
```mermaid
flowchart LR
    A["HTRDataset<br/>(aligned, unique_words,<br/>word embeddings/probs)"] --> B["Refine Backbone<br/>CTC on aligned only"]
    A -->|all images (eval)| C["Harvest Descriptors"]
    C --> D["Train Projector(s)<br/>ProjectionLoss (OT + supervised MSE)"]
    D --> E["OTAligner.align()<br/>compute OT plan + projections"]
    E --> F["Select Candidates<br/>metric: gap/entropy + agreement"]
    F --> G["Update dataset.aligned<br/>(pseudo-labels)"]
    G -->|loop| B
    A --> H["Test split (n_aligned=0)"]
    H --> I["compute_cer / compute_wer"]
```

**Loop narrative:** Each round fine-tunes the backbone on the currently aligned subset, freezes it to train projector(s) with OT-regularised ProjectionLoss, then leverages the OT transport plan to pseudo-label a small batch of unaligned samples that meet confidence and agreement thresholds. CER/WER are logged on a clean test split after each cycle until no unaligned items remain or a stopping condition is met.

## Guardrails and Assumptions
- **Aligned-only supervision:** `refine_visual_backbone` filters the dataset so the CTC head only observes samples with `aligned >= 0`.
- **Unaligned-only pseudo-labelling:** `OTAligner` masks out already aligned instances before computing uncertainty scores.
- **Leak-free evaluation:** The test split is instantiated with `n_aligned = 0` to avoid using target labels during pseudo-labelling.
- **CUDA requirement:** Projector training asserts a CUDA-capable device; alignments are GPU-backed for performance.
- **Variance metric:** Ensemble variance is handled inside `OTAligner`; `select_uncertain_instances` accepts `"gap"` or `"entropy"` only.

## Getting Started
### Prerequisites
- Python 3.10+
- CUDA-capable GPU (required for projector training)
- [POT](https://pythonot.github.io/), SciPy, PyTorch, Albumentations, pyctcdecode, editdistance, OmegaConf (listed in `requirements.txt`)

### Installation
```bash
# clone the renamed repository
git clone https://github.com/petrosgeor/optimal-transport-for-handwritten-text-recognition-in-a-low-resource-regime.git
cd optimal-transport-for-handwritten-text-recognition-in-a-low-resource-regime

# install python dependencies
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Dataset Preparation
1. Download IAM/GW/CVL handwritten word images following the paper protocol.
2. Arrange the data under `data/<DATASET_NAME>/{train,val,test}` (structure configurable through `trainer_config.yaml → dataset.basefolder`).
3. Optionally prepare the synthetic 90k dataset (see `PretrainingHTRDataset`) if you plan to mix synthetic samples during backbone refinement.
4. Build word embeddings and priors on first run; the dataset utilities cache them on disk.

## Configuration at a Glance
The main experiment knob lives in `alignment/alignment_configs/trainer_config.yaml`. The snippet below highlights the paper defaults:

```yaml
refine:
  epochs: 5
  batch_size: 32
  lr: 1.0e-4
  main_weight: 1.0
  aux_weight: 0.3
  synthetic_mix: 0.2
projector:
  epochs: 4
  batch_size: 512
  lr: 5.0e-4
  ensemble_size: 3
  device: cuda
align:
  k: 32
  metric: entropy
  agree_threshold: 2
  reg: 0.05
  unbalanced: false
experiment:
  alt_rounds: 2
  enable_pseudo_labeling: true
  device: cuda:0
  dataset:
    basefolder: data/IAM
    subset: all
    n_aligned: 24
    word_prob_strategy: empirical
```

Update the YAML to match your hardware (e.g., GPU id, batch sizes) and dataset paths. For ablations, keep a copy of the exact config used in appendix tables.

## Running the Pipeline
1. **Create the datasets:**
   ```bash
   python - <<'PY'
   from alignment.data_entry import build_dataset_bundle
   cfg_path = "alignment/alignment_configs/trainer_config.yaml"
   build_dataset_bundle(cfg_path)
   PY
   ```
2. **Launch alternating refinement:**
   ```bash
   python -m alignment.trainer \
     --config alignment/alignment_configs/trainer_config.yaml
   ```
   The script orchestrates backbone refinement, projector training, OT pseudo-labelling, and metric logging.
3. **Backbone-only baseline:**
   ```bash
   python -m alignment.trainer \
     --config alignment/alignment_configs/trainer_config.yaml \
     --override "experiment.enable_pseudo_labeling=false"
   ```
   This skips projector training and OT, mirroring the supervised ablation in the paper.

## Reproducing Paper Results
1. **Seed logging:** run the pipeline once to ensure caches (embeddings, priors, dataset splits) are initialised.
2. **Paper configuration:** duplicate the YAML used in the manuscript (e.g., `trainer_config_paper.yaml`) and commit it to guarantee traceability.
3. **Train:** execute the alternating loop with the paper config. Collect per-round CER/WER from `results/metrics_round_*.tsv`.
4. **Evaluate checkpoints:** after convergence, run the evaluation utilities on the saved backbone to compute final CER/WER.
5. **Pseudo-label audit:** inspect `results/pseudo_labels_round_*.tsv` for agreement statistics and confirm that only previously unaligned indices were updated.
6. **Report:** aggregate metrics into the tables referenced in the manuscript (e.g., CER@round3, WER@round_final).

## Evaluation and Reporting
```bash
python - <<'PY'
from alignment.eval import compute_cer, compute_wer
from alignment.trainer import load_backbone_for_eval
from htr_base.utils.htr_dataset import HTRDataset

backbone = load_backbone_for_eval("results/checkpoints/backbone_final.pt")
dataset = HTRDataset(basefolder="data/IAM", subset="test", config={"n_aligned": 0})
cer = compute_cer(dataset, backbone, batch_size=64, device="cuda")
wer = compute_wer(dataset, backbone, batch_size=64, device="cuda")
print(f"CER: {cer:.2f}%  WER: {wer:.2f}%")
PY
```
This script mirrors the evaluation protocol used for the paper tables. `compute_wer` and `compute_cer` accept `decode="beam"` if you wish to replicate the KenLM-free beam search variant.

## Logging and Artifacts
- **Metrics:** `results/metrics_round_*.tsv` (round index, CER, WER, correct pseudo labels).
- **Pseudo-labels:** `results/pseudo_labels_round_*.tsv` (image id, new label, confidence stats).
- **Checkpoints:** `results/checkpoints/` (backbone, projector ensemble, optimizer state).
- **Visualisations:** Optional t-SNE plots saved under `results/plots/` when enabled.
- **Config snapshots:** Every launcher run stores the resolved config under `results/config_used.yaml` for reproducibility.

## Project Layout
| Path | Description |
| ---- | ----------- |
| `alignment/trainer.py` | Entry point for the alternating refinement loop and evaluation helpers. |
| `alignment/alignment_utilities.py` | Optimal transport aligner, candidate selection, pseudo-label validation. |
| `alignment/losses.py` | ProjectionLoss (OT + supervised) and optional soft-contrastive loss. |
| `alignment/ctc_utils.py` | CTC encoding/decoding helpers and probability utilities. |
| `htr_base/utils/htr_dataset.py` | Dataset wrappers for aligned/unaligned handling, word embeddings, priors. |
| `htr_base/models.py` | Backbone (HTRNet), projector MLPs, and heads (CTC variants). |
| `alignment/alignment_configs/` | YAML configuration files for training and pretraining. |
| `results/` | Default output directory for metrics, pseudo-label TSVs, and checkpoints. |
| `tests/` | Smoke tests and fixtures for quick regressions. |

## API Reference

### Classes
#### HTRDataset (`htr_base/utils/htr_dataset.py`)
Loads handwriting images, transcriptions, word embeddings, and maintains the `aligned` map that drives the alternating pipeline.

**Args (constructor):**
- `basefolder` (str): Root directory with `train/`, `val/`, and `test` subsets.
- `subset` (str): `'train'`, `'val'`, `'test'`, `'all'`, or `'train_val'`.
- `fixed_size` (tuple[int, int]): Image resize target `(height, width or None)`.
- `transforms` (list | None): Optional Albumentations pipeline (two-view compatible).
- `character_classes` (list[str] | None): Override vocabulary; defaults to dataset-specific charsets.
- `config` (Any): Extra configuration, including `n_aligned` and sampling strategy.
- `two_views` (bool): Return two augmented views for contrastive training when `True`.

**Attributes:**
- `data` (list[tuple]): File path and transcription pairs.
- `transcriptions` (list[str]): Ground-truth words per sample.
- `unique_words` (list[str]): Deduplicated vocabulary built from training data.
- `unique_word_probs` (torch.Tensor): Prior probability of each word (empirical, uniform, or wordfreq).
- `unique_word_embeddings` (torch.Tensor): Word embeddings cached on disk.
- `aligned` (torch.IntTensor): `-1` for unaligned entries, otherwise index into `unique_words`.
- `is_in_dict` (torch.IntTensor): Flags words that exist in the vocabulary.

**Methods:**
- `__len__() -> int`
- `__getitem__(index: int) -> tuple`
- `get_dataset_name() -> str`
- `word_frequencies() -> tuple[list[str], list[float]]`
- `get_test_indices() -> torch.Tensor`

**Example:**
```python
from htr_base.utils.htr_dataset import HTRDataset
train_ds = HTRDataset(basefolder="data/IAM", subset="all", config={"n_aligned": 24})
image, transcription, aligned_id = train_ds[0]
assert aligned_id in (-1, *range(len(train_ds.unique_words)))
```

#### PretrainingHTRDataset (`htr_base/utils/htr_dataset.py`)
Wraps the synthetic 90k OCR dataset used for optional backbone warm-up.

**Args (constructor):**
- `list_file` (str): Path to file listing image relative paths and labels.
- `fixed_size` (tuple[int, int]): Image resize size.
- `base_path` (str): Base directory for image paths.
- `transforms` (list | None): Optional Albumentations pipeline.
- `n_random` (int | None): If set, enforce exactly `n_random` valid samples.
- `random_seed` (int): Seed for deterministic sampling order.
- `preload_images` (bool): Preload images into memory if `True`.

**Attributes:**
- `img_paths` (list[str]): Absolute paths to images.
- `transcriptions` (list[str]): Labels associated with each image.
- `preload_images` (bool): Indicates whether images were cached in memory.

**Methods:**
- `__len__() -> int`
- `__getitem__(index: int) -> tuple`
- `save_image(index: int, out_dir: str, filename: str | None = None) -> str`
- `loaded_image_shapes() -> list[tuple[int, int]]`

**Example:**
```python
from htr_base.utils.htr_dataset import PretrainingHTRDataset
syn_ds = PretrainingHTRDataset(list_file="data/90k/imlist.txt", n_random=10000)
img, label = syn_ds[0]
print(label)
```

#### HTRNet (`htr_base/models.py`)
Backbone network with configurable CNN, optional RNN/Transformer heads, holistic feature projection, and PHOC auxiliary branch.

**Args (constructor):**
- `arch_cfg` (omegaconf.DictConfig): Contains `cnn_cfg`, `head_type`, `rnn_type`, `feat_dim`, `feat_pool`, `phoc_levels`, etc.
- `nclasses` (int): Vocabulary size including the CTC blank symbol.

**Attributes:**
- `features` (nn.Module): CNN feature extractor.
- `top` (nn.Module): Chosen head (CNN/RNN/Both/Transformer) for CTC logits.
- `feat_head` (nn.Module | None): Projects holistic descriptor when `feat_dim` > 0.
- `phoc_head` (nn.Module | None): PHOC prediction layer for auxiliary supervision.

**Methods:**
- `forward(x: torch.Tensor, *, return_feats: bool = True) -> tuple`

**Example:**
```python
from htr_base.models import HTRNet
from omegaconf import OmegaConf
arch_cfg = OmegaConf.create({"cnn_cfg": [[2, 64], "M", [3, 128]], "head_type": "rnn", "rnn_type": "gru", "feat_dim": 512})
model = HTRNet(arch_cfg, nclasses=96)
logits, feats = model(torch.randn(1, 1, 128, 512))
```

#### Projector (`htr_base/models.py`)
Three-layer MLP that maps backbone descriptors to the word embedding space used by the OT aligner.

**Args (constructor):**
- `input_dim` (int): Descriptor dimensionality from the backbone.
- `output_dim` (int): Word embedding dimension.
- `dropout` (float): Dropout rate (applied after the first two activations).

**Attributes:**
- `input_dim` (int)
- `output_dim` (int)
- `sequential` (nn.Sequential): MLP layers.

**Methods:**
- `forward(x: torch.Tensor) -> torch.Tensor`

**Example:**
```python
from htr_base.models import Projector
proj = Projector(input_dim=512, output_dim=1024, dropout=0.2)
projected = proj(torch.randn(32, 512))
```

#### OTAligner (`alignment/alignment_utilities.py`)
Runs optimal transport between projected descriptors and word embeddings, selects confident candidates, and updates the dataset alignment map.

**Args (constructor):**
- `dataset` (HTRDataset): Dataset providing descriptors and alignment metadata.
- `backbone` (HTRNet): Frozen backbone used to harvest descriptors.
- `projectors` (Sequence[nn.Module]): Projector ensemble.
- `batch_size` (int): Mini-batch size when extracting descriptors.
- `device` (str): Device used for descriptor harvesting and OT computations.
- `reg` (float): Entropic regularisation strength for Sinkhorn.
- `unbalanced` (bool): Enable unbalanced OT when `True`.
- `reg_m` (float): Mass regularisation term for unbalanced OT.
- `sinkhorn_kwargs` (dict | None): Extra keyword arguments forwarded to POT.
- `k` (int): Number of least-moved descriptors promoted per call.
- `metric` (str): Candidate ranking metric (`"gap"` or `"entropy"`).
- `agree_threshold` (int): Minimum number of projectors that must agree.

**Attributes:**
- `word_embs` (torch.Tensor): Word embeddings cached on the chosen device.
- `aligned_before` (torch.Tensor): Snapshot of alignment prior to each call.
- `transport_plan` (torch.Tensor | None): Last computed transport matrix.

**Methods:**
- `align() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]`
- `validate_pseudo_labels(edit_threshold: int, batch_size: int, decode_cfg: dict, num_workers: int) -> int`

**Example:**
```python
from alignment.alignment_utilities import OTAligner
aligner = OTAligner(dataset=train_ds, backbone=model, projectors=[proj], batch_size=256, device="cuda", k=16)
plan, projections, moved = aligner.align()
print(plan.shape)
```

### Key Functions
#### `alternating_refinement` (`alignment/trainer.py`)
**Purpose:** Orchestrates the alternating training loop that refines the backbone, trains projectors, and pseudo-labels new samples.

**Args:**
- `dataset` (HTRDataset): Dataset with alignment metadata.
- `backbone` (HTRNet): CTC backbone to refine.
- `projectors` (Sequence[nn.Module]): Projector ensemble.
- `rounds` (int): Upper bound on alternating rounds when pseudo-labelling is disabled.
- `enable_pseudo_labeling` (bool): Toggle OT pseudo-labelling stage.
- `logger` (logging.Logger | None): Optional logger for progress.
- `**kwargs`: Additional overrides (batch sizes, number of OT calls, validation hooks).

**Returns:**
- `None`

**Example:**
```python
from alignment.trainer import alternating_refinement
alternating_refinement(dataset=train_ds, backbone=model, projectors=[proj1, proj2], rounds=6, enable_pseudo_labeling=True)
```

#### `refine_visual_backbone` (`alignment/trainer.py`)
**Purpose:** Fine-tunes the backbone on currently aligned samples using CTC, optional PHOC, and soft-contrastive losses.

**Args:**
- `dataset` (HTRDataset): Supplies aligned samples and augmentation flags.
- `backbone` (HTRNet): Model to optimise.
- `epochs` (int): Number of epochs.
- `batch_size` (int): Mini-batch size.
- `lr` (float): Learning rate.
- `weights` (dict): Loss weights (`main`, `aux`, `phoc`, `contrastive`).
- `pretrain_ds` (PretrainingHTRDataset | None): Optional synthetic dataset mix.

**Returns:**
- `None`

**Example:**
```python
from alignment.trainer import refine_visual_backbone
refine_visual_backbone(dataset=train_ds, backbone=model, epochs=3, batch_size=32, lr=1e-4, weights={"main": 1.0, "aux": 0.3})
```

#### `train_projector` (`alignment/trainer.py`)
**Purpose:** Freezes the backbone, caches descriptors, and optimises the projector ensemble with OT-based ProjectionLoss and supervised alignment term.

**Args:**
- `dataset` (HTRDataset): Supplies descriptors and alignment info.
- `backbone` (HTRNet): Frozen backbone used for descriptor harvesting.
- `projectors` (Sequence[nn.Module]): Ensemble to train.
- `epochs` (int): Number of epochs.
- `batch_size` (int): Mini-batch size when iterating descriptors.
- `lr` (float): Learning rate.
- `weight_decay` (float): Optimiser weight decay.
- `device` (str): CUDA device (required).

**Returns:**
- `None`

**Example:**
```python
from alignment.trainer import train_projector
train_projector(dataset=train_ds, backbone=model, projectors=[proj], epochs=4, batch_size=512, lr=5e-4, weight_decay=0.0, device="cuda")
```

#### `align_more_instances` (`alignment/alignment_utilities.py`)
**Purpose:** Convenience wrapper around `OTAligner` that optionally validates pseudo-labels and returns OT statistics.

**Args:**
- `dataset` (HTRDataset)
- `backbone` (HTRNet)
- `projectors` (Sequence[nn.Module])
- `batch_size` (int)
- `device` (str)
- `reg` (float)
- `unbalanced` (bool)
- `reg_m` (float)
- `sinkhorn_kwargs` (dict | None)
- `k` (int)
- `metric` (str): `"gap"` or `"entropy"`.
- `agree_threshold` (int)

**Returns:**
- `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`: transport plan, projected descriptors, moved distances.

**Example:**
```python
from alignment.alignment_utilities import align_more_instances
plan, projections, moved = align_more_instances(dataset=train_ds, backbone=model, projectors=[proj], k=32, metric="entropy", agree_threshold=2)
```

#### `select_uncertain_instances` (`alignment/alignment_utilities.py`)
**Purpose:** Rank transport-plan rows by uncertainty (`"gap"` or `"entropy"`) to prioritise pseudo-labelling.

**Args:**
- `m` (int): Number of indices to return.
- `transport_plan` (numpy.ndarray | None): OT transport plan (required for `metric="entropy"`).
- `dist_matrix` (numpy.ndarray | None): Distance matrix between descriptors and embeddings (required for `metric="gap"`).
- `metric` (str): Either `"gap"` or `"entropy"`.

**Returns:**
- `numpy.ndarray`: Indices of the top-`m` most uncertain samples.

**Example:**
```python
from alignment.alignment_utilities import select_uncertain_instances
indices = select_uncertain_instances(10, transport_plan=plan.cpu().numpy(), metric="entropy")
```

#### `compute_cer` (`alignment/eval.py`)
**Purpose:** Compute the character error rate (CER) on a dataset using greedy or beam decoding.

**Args:**
- `dataset` (torch.utils.data.Dataset): Provides `(image, transcription, _)` tuples.
- `model` (torch.nn.Module): Backbone with CTC head.
- `batch_size` (int): Mini-batch size.
- `device` (str): Device for inference.
- `decode` (str): `'greedy'` or `'beam'`.

**Returns:**
- `float`: CER percentage (not rounded).

**Example:**
```python
from alignment.eval import compute_cer
cer = compute_cer(dataset=test_ds, model=model, batch_size=64, device="cuda", decode="beam")
```

#### `compute_wer` (`alignment/eval.py`)
**Purpose:** Compute the word error rate (WER) analogue of `compute_cer`.

**Args:**
- `dataset` (torch.utils.data.Dataset)
- `model` (torch.nn.Module)
- `batch_size` (int)
- `device` (str)
- `decode` (str)

**Returns:**
- `float`: WER percentage.

**Example:**
```python
from alignment.eval import compute_wer
wer = compute_wer(dataset=test_ds, model=model, batch_size=64, device="cuda")
```

#### `encode_for_ctc` (`alignment/ctc_utils.py`)
**Purpose:** Convert a list of strings into compact CTC targets.

**Args:**
- `transcriptions` (list[str]): Words or phrases to encode.
- `c2i` (dict[str, int]): Character-to-index mapping (blank reserved at index `0`).
- `device` (str | torch.device | None): Optional target device.

**Returns:**
- `Tuple[torch.IntTensor, torch.IntTensor]`: Flattened targets and per-sample lengths.

**Example:**
```python
from alignment.ctc_utils import encode_for_ctc
c2i = {"<blank>": 0, "a": 1, "b": 2}
targets, lengths = encode_for_ctc(["abba"], c2i)
print(targets.tolist(), lengths.tolist())
```

## Experiments and Tests
- **Smoke test:** `python -m tests.test_alignment_loop` (ensures dataset loading, OT aligner, and trainer interoperate).
- **Toy example:**
  ```bash
  python - <<'PY'
  from alignment.trainer import alternating_refinement
  from htr_base.utils.htr_dataset import HTRDataset
  from htr_base.models import HTRNet, Projector
  from omegaconf import OmegaConf

  cfg = OmegaConf.load("alignment/alignment_configs/trainer_config.yaml")
  ds = HTRDataset(basefolder=cfg.dataset.basefolder, subset="train", config={"n_aligned": cfg.dataset.n_aligned})
  backbone = HTRNet(cfg.architecture, nclasses=len(ds.character_classes) + 1)
  projector = Projector(cfg.architecture.feat_dim, ds.unique_word_embeddings.shape[1])
  alternating_refinement(ds, backbone, [projector], rounds=1, enable_pseudo_labeling=False)
  PY
  ```
  This minimal run validates configuration integrity without modifying saved models.

## Troubleshooting
- **`RuntimeError: expected CUDA device`:** Ensure `projector.device` and `experiment.device` point to a valid GPU and that `torch.cuda.is_available()` is `True`.
- **Few pseudo-labels per round:** Increase `align.k`, lower `agree_threshold`, or confirm that `dataset.aligned` still contains `-1` entries.
- **Sinkhorn divergence:** Tune `align.reg` or provide `sinkhorn_kwargs={"max_iter": 200}`.
- **Beam decoder missing KenLM:** The project defaults to KenLM-free decoding; warnings are suppressed. Install a KenLM language model only if required for ablations.

## License
Distributed under the MIT License. See `LICENSE` for details.
