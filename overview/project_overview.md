# Handwritten Text Recognition through Iterative Visual-Semantic Alignment

## Project Goal

This project tackles the fundamental challenge in handwritten text recognition (HTR): **training robust character-level recognition models when we don't know in advance which specific words authors have written**. The approach uses a sophisticated iterative refinement strategy that bridges visual features with semantic word representations.

---

## Core Challenge & Solution Strategy

### The Problem

Traditional HTR approaches require extensive character-level annotations, but we often have images of handwritten text without knowing the exact vocabulary used by historical authors or the document’s domain. This makes fully supervised approaches impractical, especially for historical or low-resource datasets.

### The Solution

The project’s core solution is a bootstrapping framework that:

1. **Seeds the process with a small set of confidently aligned word instances** (i.e., words that exactly match a known external vocabulary),
2. **Leverages language priors** from modern English (or other target languages) to inform alignment,
3. **Uses optimal transport theory** to match distributions of visual features and word semantics,
4. **Iteratively expands the set of aligned images** (pseudo-labeling new instances based on model confidence),
5. **Alternates training between visual and semantic modules** for optimal mutual refinement,
6. **Trains a backbone network for robust character recognition** that ultimately generalizes beyond the initial seed vocabulary.

---

## Key Methodological Components

### 1. Visual Feature Space & HTRNet Backbone

- Uses a custom **HTRNet backbone** (CNN + optional RNN or Transformer CTC heads) to extract feature descriptors from line or word images.
- Visual features are projected into a learned embedding space. Optionally, a global feature vector is produced for each image (configurable via architecture YAML).
- Assumes a **uniform distribution** over the visual feature manifold to make optimal transport (OT) calculations tractable.
- The network architecture, data flow, and all training parameters are highly configurable for research flexibility【7†README.md】.

### 2. External Vocabulary & Language Priors

- Employs a list of the top-k most frequent English words as the **external vocabulary** (k typically 200+; can be adjusted).
- **Word frequency statistics** from modern English corpora provide prior probabilities for each word.
- **Word embeddings** for vocabulary are computed using Multi-Dimensional Scaling (MDS) on pairwise Levenshtein (edit) distances, preserving semantic similarity in embedding space.
- The vocabulary is filtered so that every external word consists only of characters present in the training set, ensuring compatibility with the dataset at hand.
- This design allows the model to benefit from statistical language knowledge even when the dataset’s actual vocabulary is unknown【7†README.md】.

### 3. Iterative Refinement Loop

The overall training proceeds in alternating cycles of three phases, tightly coupled by the OT-based alignment machinery:

#### Phase A: Backbone Refinement

- Fine-tune the visual backbone **only on currently aligned word instances** (i.e., images whose transcriptions match an external vocabulary word).
- Trains with standard CTC loss (main and auxiliary heads if applicable).
- Increases the backbone’s ability to recognize “known” words with high certainty【7†README.md】.

#### Phase B: Projector Training

- **Freezes the visual backbone.**
- Collects visual descriptors for all images, then trains a separate MLP “projector” that maps backbone features into the word embedding space.
- **Loss:** Unsupervised OT loss between all (projected) features and word embeddings, plus a supervised MSE loss for already aligned pairs.
- **Objective:** Make the projected visual features “semantic,” i.e., close to the word embeddings they should correspond to.

#### Phase C: Alignment Expansion (Pseudo-labeling)

- Computes an **optimal transport plan** to align current visual features with word embeddings.

- **Selects the most confident new matches** (least-moved instances under the OT plan), and pseudo-labels them as additional aligned pairs.

- **Updates the alignment list** for the next iteration, growing the pool of “trusted” training data step by step.

- This loop repeats for several rounds, typically with increasingly aggressive pseudo-labeling as model confidence grows【9†project\_overview\.md】【7†README.md】.

---

## Project Structure and Components

- **htr_base/**: Core data loading utilities, model definitions (CNN, RNN, Transformer heads), backbone training and evaluation scripts. Includes basic data preparation for IAM and George Washington datasets.
- **alignment/**: Alignment routines (optimal transport, pseudo-labeling, projector training), losses, and utilities for matching visual features to semantic space and expanding alignment set.
- **tests/**: Scripts for evaluating the effect of length, split, and pseudo-labeling policies. Includes character error rate (CER) analysis by subset.
- **YAML config files**: All model architecture, training, and alignment parameters are specified in config files for full reproducibility and experiment tracking.
- **requirements.txt**: All necessary Python dependencies are listed for out-of-the-box installation.

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
|               (alignment/pretraining.py)          |
| - `pretraining.main`                              |
| - Trains HTRNet backbone on image lists           |
| - Optional PHOC and Contrastive losses            |
+-------------------------+-------------------------+
          |
          | Refined Backbone
          v
+---------------------------------------------------+
|              Alternating Refinement               |
|            (alignment/alignment_trainer.py)       |
| - `alternating_refinement` (Iterative Loop)       |
| - Starts with pretrained HTRNet backbone          |
| - Process:                                        |
|   1. `refine_visual_backbone`: Fine-tunes HTRNet  |<--+
|      on aligned samples.                          |   |
|   2. `train_projector`: Trains projector(s) to    |   |
|      map features to embedding space.             |   |
|   3. `align_more_instances` (via OTAligner):      |   |
|      Aligns more dataset instances using Optimal  |---+
|      Transport.                                   |
+-------------------------+-------------------------+
          |
          | Aligned Data & Metrics
          v
+---------------------------------------------------+
|                 Evaluation & Utilities            |
| - `ctc_utils.py`: encode_for_ctc, greedy_ctc_decode|
| - `alignment_utilities.py`: print_dataset_stats,  |
|   harvest_backbone_features, select_uncertain_instances|
| - `plot.py`: plot_dataset_augmentations,          |
|   plot_projector_tsne, plot_tsne_embeddings       |
| - Metrics: CER, WER, word_silhouette_score        |
+---------------------------------------------------+

---

## End Goal

The ultimate outcome is to achieve a **low character error rate (CER) on the `HTRDataset` test split**. To reach this target we first pretrain the HTRNet backbone on a large synthetic dataset, providing a strong initialization. We then iteratively pseudo-label images from the `train_val` portion of `HTRDataset`&mdash;only a fraction of which has ground-truth labels&mdash;expanding the training set at each round. This cycle of pretraining, pseudo-labeling, and refinement yields a backbone capable of transcribing historical or out-of-vocabulary words and enables new research in digital humanities, archival transcription, and beyond.
