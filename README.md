# Handwritten Text Recognition (HTR) tools

This repository contains a minimal implementation for handwritten text recognition using PyTorch. The code is structured around a backbone network with a CTC head, data loading utilities and simple training scripts.

## HTRNet

`htr_base/models.py` defines the main neural network used throughout the codebase. `HTRNet` is composed of a CNN backbone followed by different CTC heads. The architecture is configurable through a YAML file or a small namespace object.

Key features:

- **Configurable CNN** using residual blocks and pooling as defined by `cnn_cfg`.
- **CTC heads**: `cnn`, `rnn` or `both` (a combination of convolutional and recurrent heads). See `CTCtopC`, `CTCtopR` and `CTCtopB` in the same file.
- Optional **feature projection** producing a global feature vector per image (`feat_dim`).
- The `forward` method returns CTC logits and optionally image descriptors.

Example usage:
```python
from types import SimpleNamespace
from htr_base.models import HTRNet

arch = SimpleNamespace(
    cnn_cfg=[[2, 64], 'M', [3, 128], 'M', [2, 256]],
    head_type='both',  # or 'rnn', 'cnn'
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
* `fixed_size`: target `(height, width)` used to resize images.
* `transforms`: optional Albumentations augmentation pipeline applied to the images.
* `character_classes`: list of characters. If `None`, the dataset infers it from the data.

Each `__getitem__` call returns `(img_tensor, transcription, alignment_id)` where `alignment_id` is `-1` if the sample is not aligned to an external word.

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

## train_by_length.py

`tests/train_by_length.py` contains helper routines for fine tuning models on subsets of ground-truth words selected by length. The `_evaluate_cer` function reports character error rate for words shorter and longer than a chosen threshold. It now also prints the total number of characters contained in the true transcriptions of each subset.



## Requirements

Python dependencies are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

## License

This project is released under the MIT license.

```
```
