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
- Clear entry point at `alignment/trainer.py`, which coordinates the full alternating loop and is the script reviewers should run to reproduce results.

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
2. Decode them into word crops and create `gt.txt` files using the provided scripts (for GW see `htr_base/prepare_gw.py`; the other corpora follow the same recipe).
3. Place the processed data under `htr_base/data/<DATASET>/processed_words/{train,val,test}` with PNG images and matching `gt.txt` lines `<filename> <transcription>`. IAM also includes optional `.uttlist` files under `splits/`.
4. Optional: prepare the synthetic 90k dataset (`PretrainingHTRDataset`) if you plan to mix synthetic samples during backbone refinement.
5. Build word embeddings and priors on first run; the dataset utilities cache them on disk.

## Configuration at a Glance
`alignment/alignment_configs/trainer_config.yaml` drives the experiments. Tune these groups:
- **Backbone refinement:** `refine_batch_size`, `refine_epochs`, `refine_lr`, `syn_batch_ratio`, plus optional PHOC/contrastive toggles.
- **Architecture:** entries under `architecture.*` choose CNN depth, head type, feature dimension, and pooling source.
- **Projector training:** `projector_batch_size`, `projector_epochs`, `projector_lr`, `projector_weight_decay`.
- **Alternating loop:** `alt_rounds`, `enable_pseudo_labeling`, and devices (`device`, `gpu_id`).
- **OT alignment:** `align_k`, `metric`, `agree_threshold`, `align_reg`, `align_unbalanced`, `align_reg_m`.
- **Datasets & extras:** `dataset.basefolder`, `dataset.n_aligned`, `word_prob_strategy`, `load_pretrained_backbone`, and the optional `synthetic_dataset` block.



## Running the Pipeline
`alignment/trainer.py` is the entry point; invoke it directly (or via `python -m alignment.trainer`) after preparing the datasets.

1. **Prepare data:** run your preprocessing scripts so `htr_base/data/...` matches the layout above (or call `alignment.data_entry.build_dataset_bundle` if you keep a helper script).
2. **Launch alternating refinement:** `python -m alignment.trainer --config alignment/alignment_configs/trainer_config.yaml`.
3. **Monitor outputs:** metrics, pseudo-label TSVs, and checkpoints land under `results/`.


## Reproducing Paper Results
1. Duplicate the YAML used in the paper runs (e.g., `trainer_config_paper.yaml`) and store it alongside the repo.
2. Train with that config (`python -m alignment.trainer --config ...`) and retain the artefacts in `results/` (metrics TSVs, checkpoints, pseudo-label logs).
3. Evaluate the final backbone with `alignment.eval.compute_cer/compute_wer` to populate the reported CER/WER tables.

## Evaluation and Reporting
Use `alignment.trainer.load_backbone_for_eval` to restore the trained model, then call `alignment.eval.compute_cer` / `compute_wer` on the test split. Both functions accept `decode="beam"` when you need the KenLM-free beam search reported in the paper.

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

## API Reference (Quick Map)
- `alignment/trainer.py`
  - `alternating_refinement`: orchestrates the backbone/projector/OT loop.
  - `refine_visual_backbone`, `train_projector`, `align_more_instances`: building blocks for bespoke training schedules.
  - `load_backbone_for_eval`: restore checkpoints for CER/WER evaluation.
- `alignment/alignment_utilities.py`
  - `OTAligner`: optimal-transport pseudo-labelling engine (feature harvesting, Sinkhorn, candidate selection, dataset updates).
  - `select_uncertain_instances`: entropy/gap ranking helper for custom selection policies.
- `alignment/eval.py`: `compute_cer` and `compute_wer` for reporting (supports greedy or KenLM-free beam decoding).
- `alignment/ctc_utils.py`: CTC utilities (`encode_for_ctc`, `greedy_ctc_decode`, `beam_search_ctc_decode`).
- `htr_base/utils/htr_dataset.py`
  - `HTRDataset`: core loader for the processed word crops; handles alignment metadata, word embeddings, and prior estimation.
  - `PretrainingHTRDataset`: synthetic 90k dataset wrapper for optional backbone warm-up.
- `htr_base/models.py`
  - `HTRNet`: configurable CNN+CTC backbone with optional PHOC and descriptor heads.
  - `Projector`: MLP that maps backbone descriptors into the word-embedding space used by OT.

> Each module ships with detailed docstrings—use `pydoc` or open the source for full signatures.
## Troubleshooting
- **`RuntimeError: expected CUDA device`:** Ensure `projector.device` and `experiment.device` point to a valid GPU and that `torch.cuda.is_available()` is `True`.
- **Few pseudo-labels per round:** Increase `align.k`, lower `agree_threshold`, or confirm that `dataset.aligned` still contains `-1` entries.
- **Sinkhorn divergence:** Tune `align.reg` or provide `sinkhorn_kwargs={"max_iter": 200}`.
- **Beam decoder missing KenLM:** The project defaults to KenLM-free decoding; warnings are suppressed. Install a KenLM language model only if required for ablations.

## License
Distributed under the MIT License. See `LICENSE` for details.
