````markdown
# Handwritten Text Recognition (HTR) tools

This repository contains a minimal implementation for handwritten text recognition using PyTorch. The code is structured around a backbone network with a CTC head, data loading utilities and simple training scripts.

The `htr_base` directory acts mainly as a collection of helper utilities and network components.
Most of the logic for alignment and model training lives in the `alignment` directory.

## HTRNet

`htr_base/models.py` defines the main neural network used throughout the codebase. `HTRNet` is composed of a CNN backbone followed by different CTC heads. The architecture is configurable through a YAML file or a small namespace object.

Key features:

- **Configurable CNN** using residual blocks and pooling as defined by `cnn_cfg`.
- **CTC heads**: `cnn`, `rnn`, `both` or `transf` (transformer-based). See `CTCtopC`, `CTCtopR`, `CTCtopB` and `CTCtopT` in the same file.
- Optional **feature projection** producing a global feature vector per image (`feat_dim`).
- The `forward` method returns CTC logits and optionally image descriptors.
  For the transformer head, provide `transf_d_model`, `transf_nhead`,
  `transf_layers` and `transf_dim_ff` in the architecture config.

Example usage:
```python
from types import SimpleNamespace
from htr_base.models import HTRNet

arch = SimpleNamespace(
    cnn_cfg=[[2, 64], 'M', [3, 128], 'M', [2, 256]],
    head_type='both',  # or 'rnn', 'cnn', 'transf'
    rnn_type='lstm',
    rnn_layers=3,
    rnn_hidden_size=256,
    flattening='maxpool',
    stn=False,
    feat_dim=None,
)
model = HTRNet(arch, nclasses=80 + 1)
````

## HTRDataset

Located in `htr_base/utils/htr_dataset.py`, `HTRDataset` loads line images and corresponding transcriptions. It automatically builds the character vocabulary if not provided. Main arguments include:

* `basefolder`: root folder containing `train/`, `val/` and `test/` subdirectories with a `gt.txt` file inside each.
* `subset`: which portion of the dataset to load (`train`, `val`, `test` or `all`).
  Using `all` merges the three splits and applies the same augmentation policy as the training split.
* **Data path**: the default configuration expects processed line images under `./data/IAM/processed_lines`.  A small sample dataset for the unit tests lives in `htr_base/data/GW/processed_words`.
* `fixed_size`: target `(height, width)` used to resize images.
* `transforms`: optional Albumentations augmentation pipeline applied to the images.
* `character_classes`: list of characters. If `None`, the dataset infers it from the data.
* `word_emb_dim`: dimensionality of the MDS word embeddings (default `512`).
* `two_views`: if `True`, `__getitem__` returns two randomly augmented views of the same line image.
* `concat_prob`: probability of concatenating a sample with itself when fetching a datum.
* The external vocabulary is automatically filtered so that all words only contain characters present in the dataset.
* If none of the dataset's transcriptions overlap with the selected `k_external_words`,
  the dataset will contain **zero** pre-aligned samples so `n_aligned` has no effect and
  refinement cannot start.

* External vocabulary words are stored in lowercase.

* `prior_char_probs`: mapping of character frequencies computed from the 50,000 most common English words.

`letter_priors(transcriptions=None, n_words=50000)` builds this mapping. If no transcriptions are provided it relies on `wordfreq` to return probabilities for `a-z0-9`.

If `two_views` is `False`, `__getitem__` returns `(img_tensor, transcription, alignment_id)`.
Otherwise it returns `((img1, img2), transcription, alignment_id)` where `img1` and `img2` are independent views of the same image.

If `concat_prob` is greater than `0`, the line image may be stitched with itself horizontally and resized back to `fixed_size`. The transcription then contains the ground-truth string twice, separated by **three** spaces.


## encode\_for\_ctc

Defined in `alignment/ctc_utils.py`. This helper converts a batch of strings into the flattened `(targets, lengths)` representation required by `torch.nn.CTCLoss`.

```python
def encode_for_ctc(transcriptions, c2i, device=None):
    """Convert a batch of texts to CTC targets and per-sample lengths."""
```

* `transcriptions`: list of strings **already padded with spaces** (the dataset wraps each line with leading and trailing spaces).
* `c2i`: dictionary mapping characters to indices where index `0` is reserved for the CTC blank.
* `device`: optional torch device for the returned tensors.

Returns `targets` (concatenated label indices) and `lengths` (original length of each transcription).

## greedy\_ctc\_decode

Also in `alignment/ctc_utils.py`. This performs a simple best‑path decoding of CTC logits.

```python
def greedy_ctc_decode(logits, i2c, blank_id=0, time_first=True):
    """Greedy decode a batch of CTC outputs."""
```

* `logits`: tensor either `(T, B, C)` or `(B, T, C)`; only the `argmax` over `C` is used.
* `i2c`: mapping from class index to character (excluding the blank).
* `blank_id`: id reserved for the CTC blank (default `0`).
* `time_first`: set to `True` if the tensor is `(T, B, C)`.

Returns a list of decoded strings, one for each element in the batch.

## align\_more\_instances

Located in `alignment/alignment_utilities.py`. This routine automatically assigns dataset images to external words via optimal transport.

```python
def align_more_instances(dataset, backbone, projector, *, batch_size=512,
                         device="cpu", reg=0.1, unbalanced=False, reg_m=1.0,
                         sinkhorn_kwargs=None, k=0):
    """Automatically align dataset images to external words using OT."""
```

* `dataset`: instance of `HTRDataset` providing images and `external_word_embeddings`.
* `backbone`: `HTRNet` used to extract visual descriptors.
* `projector`: projects descriptors to the embedding space.
* `batch_size`: mini-batch size when harvesting descriptors.
* `device`: device used for feature extraction, descriptor processing and
  the projector.
* `reg`: entropic regularisation for Sinkhorn.
* `unbalanced`: use unbalanced OT formulation.
* `reg_m`: additional unbalanced regularisation parameter.
* `sinkhorn_kwargs`: extra arguments for the Sinkhorn solver.
* `k`: number of least-moved descriptors to pseudo-label.

After each call, the function now reports round-wise pseudo-labelling accuracy
and the cumulative accuracy over all aligned samples.

The underlying OT projection step handles rows with zero mass safely, avoiding
`inf` or `nan` values when some descriptors receive no transport mass.

Returns the OT transport plan, the projected descriptors after OT and the distance moved by each descriptor.

## select_uncertain_instances

Located in `alignment/alignment_utilities.py`. Given either a distance matrix or a transport plan, this helper returns the indices of the most uncertain dataset instances.

```python
def select_uncertain_instances(m, *, transport_plan=None, dist_matrix=None, metric="gap"):
    """Return indices of the ``m`` most uncertain dataset items."""
```

* `m`: how many indices to return.
* `transport_plan`: OT matrix used when `metric="entropy"`.
* `dist_matrix`: pairwise distances used when `metric="gap"`.
* `metric`: either `'gap'` (smallest nearest-neighbour gap) or `'entropy'`.

## plot_dataset_augmentations

Also in `alignment/alignment_utilities.py`. Saves a figure with three dataset
images and their augmented versions side by side.

```python
def plot_dataset_augmentations(dataset, save_path):
    """Save a figure of three images and their augmentations."""
```

* `dataset`: `HTRDataset` instance with augmentation transforms.
* `save_path`: where to write the resulting PNG file.

## plot_tsne_embeddings

Also in `alignment/alignment_utilities.py`. This helper projects backbone descriptors with t-SNE and colours samples by their alignment label.

```python
def plot_tsne_embeddings(dataset, backbone, save_path):
    """Generate a coloured t-SNE plot of backbone embeddings."""
```

* `dataset`: dataset providing images and `aligned` flags.
* `backbone`: visual encoder used to compute descriptors.
* `save_path`: destination for the resulting PNG file.

Points with `aligned == 1` appear in **blue** while the rest are **black**.



## print_dataset_stats

Located in `alignment/alignment_utilities.py`. Given an `HTRDataset` instance,
this helper prints useful information such as:

- total number of samples and how many are already aligned,
- size of the external vocabulary,
- number and percentage of dataset items found in that vocabulary,
- whether all transcriptions and external words are lowercase,
- average transcription length.

```python
def print_dataset_stats(dataset):
    """Print basic statistics about *dataset*."""
```

## harvest_backbone_features

Also in `alignment/alignment_utilities.py`. This routine harvests image descriptors from the backbone for each dataset item and stores their current alignment labels.

```python
def harvest_backbone_features(dataset, backbone, *, batch_size=512,
                              num_workers=0, device="cuda"):
    """Return (descriptors, alignment) tensors for the whole dataset."""
```

* `dataset`: dataset providing images and `aligned` flags.
* `backbone`: network used to compute descriptors.
* `batch_size`: mini-batch size during feature extraction.
* `num_workers`: data loader workers used while harvesting.
* `device`: computation device for feature extraction.

Dataset augmentations are disabled while features are harvested.

## refine\_visual\_backbone

Defined in `alignment/alignment_trainer.py`. It fine-tunes the visual backbone on the subset of images already aligned to external words. Only those pre-aligned samples are loaded during training.

```python
def refine_visual_backbone(dataset, backbone, num_epochs, *, batch_size=128,
                           lr=1e-4, main_weight=1.0, aux_weight=0.1):
    """Fine‑tune *backbone* only on words already aligned to external words."""
```

* `dataset`: training dataset with alignment information.
* `backbone`: network to refine.
* `num_epochs`: number of optimisation epochs.
* `batch_size`: mini-batch size.
* `lr`: learning rate.
* `main_weight`/`aux_weight`: weights for the main and auxiliary CTC losses.
* External words are automatically wrapped with spaces before encoding so that
  no persistent changes are made to `dataset.external_words`.

## train\_projector

Also in `alignment/alignment_trainer.py`. This freezes the backbone, collects image descriptors and trains a separate projector using an OT-based loss.

```python
def train_projector(dataset, backbone, projector, num_epochs=150,
                    batch_size=512, lr=1e-4, num_workers=0,
                    weight_decay=1e-4, device="cuda"):
    """Freeze *backbone*, collect image descriptors -> train *projector*."""
```

* `dataset`: dataset with `external_word_embeddings`.
* `backbone`: frozen encoder producing descriptors.
* `projector`: learnable mapping to the embedding space.
* `num_epochs`, `batch_size`, `lr`: training hyperparameters.
* `num_workers`: data loading workers during descriptor harvesting.
* `weight_decay`: weight decay for the optimiser.
* `device`: computation device for training.

## alternating\_refinement

Also in `alignment/alignment_trainer.py`. This helper repeatedly refines the
visual backbone and projector while progressively aligning more dataset
instances using Optimal Transport.

```python
def alternating_refinement(dataset, backbone, projector, *, rounds=4,
                           backbone_epochs=2, projector_epochs=100,
                           refine_kwargs=None, projector_kwargs=None,
                           align_kwargs=None):
    """Alternately train ``backbone`` and ``projector`` with OT alignment."""
```

* `rounds`: number of backbone/projector cycles per alignment pass.
* `backbone_epochs`: epochs for each backbone refinement round.
* `projector_epochs`: epochs for each projector training round.
* `refine_kwargs`: extra keyword arguments forwarded to
  `refine_visual_backbone`.
* `projector_kwargs`: keyword arguments for `train_projector`.
* `align_kwargs`: parameters for `align_more_instances`.

## tee_output

Also in `alignment/alignment_trainer.py`. This context manager duplicates
`stdout` to a file while it is active, recreating the file on each run.

```python
from alignment.alignment_trainer import tee_output

with tee_output("results.txt"):
    alternating_refinement(dataset, backbone, projector)
```

## alignment/config.yaml

Additional hyperparameters for the alignment workflow are stored in
`alignment/config.yaml`. A new option, `n_aligned`, controls how many
dataset samples are initially marked as aligned to external words.
These pre-aligned items give a warm start to backbone refinement.

## train\_by\_length.py

`tests/train_by_length.py` contains helper routines for fine tuning models on subsets of ground-truth words selected by length. The `_evaluate_cer` function reports character error rate for words shorter and longer than a chosen threshold. It now also prints the total number of characters contained in the true transcriptions of each subset.

## Requirements

Python dependencies are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

## License

This project is released under the MIT license.


