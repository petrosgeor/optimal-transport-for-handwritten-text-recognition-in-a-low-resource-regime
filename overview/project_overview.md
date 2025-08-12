# Handwritten Text Recognition through Iterative Visual‑Semantic Alignment

## Project Goal

This project tackles a core challenge in handwritten text recognition (HTR): training robust character‑level recognizers when we do not know in advance which words are present. Our approach bridges visual features with semantic word representations and refines models through an alternating, EM‑only workflow.

---

## Core Challenge & Solution Strategy

### The Problem

Traditional HTR requires extensive character‑level annotations. In many settings (e.g., historical documents), we have images of handwritten text but limited or no aligned ground truth. Fully supervised approaches become impractical in these low‑resource scenarios.

### The Solution

We use a bootstrapping framework that:

1. Seeds training with a small set of confidently aligned word instances.
2. Leverages dataset word‑frequency priors (and optionally external priors) to inform alignment.
3. Uses optimal transport (OT) to relate distributions of visual features and word embeddings.
4. Runs EM‑only refinement using soft responsibilities over vocabulary words (no hard pseudo‑labels).
5. Alternates training between visual and semantic modules for mutual refinement.
6. Trains a backbone that generalizes beyond the initial seed vocabulary.

---

## Key Methodological Components

### 1. Visual Feature Space & HTRNet Backbone

- A configurable HTRNet backbone (CNN + optional CTC heads) extracts descriptors from word or line images.
- Features can be projected into an embedding space; a global descriptor per image is available depending on the architecture.
- Assumes a simple source marginal over descriptors for OT computations.

### 2. Dataset Vocabulary & Language Priors

- Build a vocabulary from the dataset, with empirical word frequencies as priors.
- Compute word embeddings (e.g., via MDS over edit‑distance space), capturing lexical structure among vocabulary entries.
- Restrict the vocabulary to valid characters seen in the dataset.

### 3. Iterative Refinement Loop (EM‑only)

Training proceeds in alternating cycles tightly coupled by OT‑derived responsibilities, without any hard alignment expansion:

#### Phase A: Backbone Refinement (M‑step)

- Fine‑tune the backbone on aligned items with standard CTC losses.
- On unaligned items, use EM losses built from soft responsibilities: expected word loss and optional EM‑PHOC loss.
- This improves recognition of known words and provides expectation‑based supervision for unknowns.

#### Phase B: Projector Training (E‑step support)

- Freeze the backbone and train an MLP projector to map backbone features to the word‑embedding space.
- Compute responsibilities with `compute_ot_responsibilities`: row‑normalized OT plans (optionally top‑K‑sparsified), mean‑fused across projector ensembles when applicable.

#### No Hard Alignment Expansion

- The previous pseudo‑labelling step (e.g., OTAligner, align_more_instances) has been removed.
- The loop uses soft responsibilities only; no dataset indices are hard‑assigned during training.

---

## Project Structure and Components

- `htr_base/`: Data loading, models (CNN, CTC heads), backbone training utilities.
- `alignment/`: Responsibilities via OT, projector training, refinement losses, and utilities for EM‑only training (no pseudo‑labelling).
- `tests/`: Minimal tests and scripts; includes CER checks and functional unit tests.
- YAML configs: Architecture, training, and alignment parameters for reproducibility.
- `requirements.txt`: Python dependencies.

---

## Pipeline Diagram

```
+-------------------+       +-----------------------+
|   HTRDataset      |       | PretrainingHTRDataset |
| (htr_base/utils)  |       |    (htr_base/utils)   |
| - Loads images    |       | - Loads synthetic     |
| - Transcriptions  |       |   images              |
| - Augmentations   |       | - Optional preload    |
+---------+---------+       +-----------+-----------+
          |                             |
          | Data Loading & Preprocessing
          v
+---------------------------------------------------+
|                 HTRNet Model                      |
|               (htr_base/models.py)                |
| - Configurable CNN Backbone                       |
| - CTC Heads (cnn, rnn, both, transf)              |
| - Optional PHOC branch, Feature Projection        |
+-------------------------+-------------------------+
          |
          |
          v
+---------------------------------------------------+
|                 Pretraining Phase                 |
|                 (optional / config)               |
| - Trains backbone on synthetic images             |
| - Optional PHOC and Contrastive losses            |
+-------------------------+-------------------------+
          |
          | Refined Backbone
          v
+---------------------------------------------------+
|              Alternating Refinement               |
|               (alignment/trainer.py)              |
| - `alternating_refinement` (Iterative Loop)       |
| - Starts with pretrained HTRNet backbone          |
| - Process:                                        |
|   1. `train_projector`: train projector(s) for    |
|      feature → word‑embedding mapping (E‑step).   |
|   2. `refine_visual_backbone`: fine‑tune HTRNet   |
|      with CTC on aligned items and EM losses on   |
|      unaligned items using responsibilities.      |
|      Responsibilities come from                   |
|      `compute_ot_responsibilities`.               |
+-------------------------+-------------------------+
          |
          | Metrics & Qualitative Plots
          v
+---------------------------------------------------+
|                 Evaluation & Utilities            |
| - `ctc_utils.py`: encode_for_ctc, greedy decode   |
| - `alignment_utilities.py`: harvest features,     |
|   compute_ot_responsibilities                      |
| - `plot.py`: TSNE embeddings, projector visuals   |
| - Metrics: CER, WER, silhouette score             |
+---------------------------------------------------+
```

---

## End Goal

Achieve a low character error rate (CER) on the test split. We pretrain the backbone on synthetic data (optional), then run alternating EM‑only refinement rounds: train projector(s), compute OT‑based responsibilities, and fine‑tune the backbone with CTC on aligned items and EM losses on unaligned ones. The final model generalizes to historical or out‑of‑vocabulary words, enabling downstream research in digital humanities and archival transcription.

