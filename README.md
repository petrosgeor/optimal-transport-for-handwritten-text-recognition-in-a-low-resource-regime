# Handwritten Text Recognition (HTR) tools

This repository contains a minimal implementation for handwritten text recognition using PyTorch. The code is structured around a backbone network with a CTC head, data loading utilities and simple training scripts.

## Table of Contents
- [Overview](#overview)
- [API Reference](#api-reference)
  - [Model Components](#model-components)
  - [Data Handling](#data-handling)
  - [Alignment Utilities](#alignment-utilities)
  - [Loss Functions](#loss-functions)
  - [Metrics](#metrics)
  - [CTC Utilities](#ctc-utilities)
  - [Plotting Utilities](#plotting-utilities)
  - [Vocabulary Utilities](#vocabulary-utilities)
  - [Training Utilities](#training-utilities)
  - [Configuration Files](#configuration-files)
- [Requirements](#requirements)
- [Knowledge Graph](#knowledge-graph)
- [License](#license)

## Overview
Most of the logic for alignment and model training lives in the `alignment` directory. The `htr_base` directory acts mainly as a collection of helper utilities and network components.

## API Reference

### Model Components

#### HTRNet

Located in: `htr_base/models.py`

HTRNet backbone with optional global feature projection.

```python
class HTRNet(nn.Module):
    def __init__(self, arch_cfg, nclasses):
```

*   `arch_cfg`: Configuration object containing architecture parameters (e.g., `cnn_cfg`, `head_type`, `rnn_type`, `feat_dim`, `feat_pool`, `phoc_levels`).
*   `nclasses`: Number of output classes for the CTC head (including the blank token).

**Attributes:**
*   `features` (CNN): Convolutional feature extractor.
*   `top` (nn.Module): CTC head chosen by `arch_cfg.head_type`.
*   `feat_head` (nn.Module | None): Global descriptor pooling module.
*   `phoc_head` (nn.Module | None): Optional PHOC prediction layer.

**Methods:**
*   `forward(x, *, return_feats=True)`: Returns a tuple containing logits and, optionally, features and PHOC logits. The exact output depends on the model configuration and the `return_feats` flag.


#### Projector

Located in: `htr_base/models.py`

A simple MLP used to map backbone descriptors to the word embedding space.

```python
class Projector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.2) -> None:
```

*   `input_dim`: Dimensionality of the input features from the backbone.
*   `output_dim`: Dimensionality of the target embedding space (e.g., word embedding dimension).
*   `dropout`: Dropout rate applied after the first two activations.

**Attributes:**
*   `input_dim` (int): Stored input dimension.
*   `output_dim` (int): Stored output dimension.
*   `sequential` (nn.Sequential): Three-layer MLP.

**Methods:**
*   `forward(x) -> torch.Tensor`: Passes `x` through the MLP without altering state.




#### BasicBlock

Located in: `htr_base/models.py`

A basic building block for the CNN, typically used in ResNet-like architectures.

```python
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
```

*   `in_planes`: Number of input channels.
*   `planes`: Number of output channels.
*   `stride`: Stride for the convolutional layers.
**Attributes:**
*   `conv1` (nn.Conv2d): First convolution.
*   `bn1` (nn.BatchNorm2d): Batch norm after ``conv1``.
*   `conv2` (nn.Conv2d): Second convolution.
*   `bn2` (nn.BatchNorm2d): Batch norm after ``conv2``.
*   `shortcut` (nn.Sequential): Identity or projection path.

**Methods:**
*   `forward(x) -> torch.Tensor`: Returns the residual output tensor.

#### CNN

Located in: `htr_base/models.py`

Configurable Convolutional Neural Network (CNN) backbone.

```python
class CNN(nn.Module):
    def __init__(self, cnn_cfg, flattening='maxpool'):
```

*   `cnn_cfg`: Configuration list defining the CNN layers (e.g., `[[2, 64], "M", [3, 128]]`).
*   `flattening`: Method to flatten the CNN output (`'maxpool'` or `'concat'`).
**Attributes:**
*   `k` (int): Fixed temporal kernel size used during max-pooling.
*   `flattening` (str): Output flattening mode.
*   `features` (nn.ModuleList): Sequence of convolutional blocks.

**Methods:**
*   `forward(x) -> torch.Tensor`: Returns the CNN feature map.


#### AttentivePool

Located in: `htr_base/models.py`

Collapses a feature map via learnable attention weights.

```python
class AttentivePool(nn.Module):
    def __init__(self, ch: int, dim_out: int) -> None:
```

*   `ch`: Number of input channels.
*   `dim_out`: Output dimension after pooling.

#### CTCtopC

Located in: `htr_base/models.py`

CTC head using a convolutional layer.

```python
class CTCtopC(nn.Module):
    def __init__(self, input_size, nclasses, dropout=0.0):
```

*   `input_size`: Number of input features.
*   `nclasses`: Number of output classes.
*   `dropout`: Dropout rate.
**Attributes:**
*   `dropout` (nn.Dropout): Dropout layer.
*   `cnn_top` (nn.Conv2d): Convolutional classifier.

**Methods:**
*   `forward(x) -> torch.Tensor`: Returns network logits.


#### CTCtopR

Located in: `htr_base/models.py`

CTC head using a Recurrent Neural Network (RNN).

```python
class CTCtopR(nn.Module):
    def __init__(self, input_size, rnn_cfg, nclasses, rnn_type='gru'):
```

*   `input_size`: Number of input features.
*   `rnn_cfg`: Tuple containing `(hidden_size, num_layers)` for the RNN.
*   `nclasses`: Number of output classes.
*   `rnn_type`: Type of RNN (`'gru'` or `'lstm'`).
**Attributes:**
*   `rec` (nn.Module): Bidirectional RNN encoder.
*   `fnl` (nn.Sequential): Final linear classifier.

**Methods:**
*   `forward(x) -> torch.Tensor`: Returns sequence logits.


#### CTCtopB

Located in: `htr_base/models.py`

CTC head combining both RNN and CNN components.

```python
class CTCtopB(nn.Module):
    def __init__(self, input_size, rnn_cfg, nclasses, rnn_type='gru'):
```

*   `input_size`: Number of input features.
*   `rnn_cfg`: Tuple containing `(hidden_size, num_layers)` for the RNN.
*   `nclasses`: Number of output classes.
*   `rnn_type`: Type of RNN (`'gru'` or `'lstm'`).
**Attributes:**
*   `rec` (nn.Module): Bidirectional RNN encoder.
*   `fnl` (nn.Sequential): Linear classifier after the RNN.
*   `cnn` (nn.Sequential): Auxiliary convolutional head.

**Methods:**
*   `forward(x) -> Tuple[torch.Tensor, torch.Tensor]`: Returns main and auxiliary logits.


#### CTCtopT

Located in: `htr_base/models.py`

Transformer-based CTC head.

```python
class CTCtopT(nn.Module):
    def __init__(self, input_size, transf_cfg, nclasses):
```

*   `input_size`: Number of input features.
*   `transf_cfg`: Tuple containing `(d_model, nhead, nlayers, dim_ff)` for the Transformer.
*   `nclasses`: Number of output classes.
**Attributes:**
*   `proj` (nn.Linear): Linear projection before the Transformer.
*   `encoder` (nn.TransformerEncoder): Transformer encoder.
*   `fc` (nn.Linear): Final classification layer.

**Methods:**
*   `forward(x) -> torch.Tensor`: Returns sequence logits.


### Data Handling

#### HTRDataset

Located in: `htr_base/utils/htr_dataset.py`

Loads handwritten text images and optional alignment info.


```python
class HTRDataset(Dataset):
    def __init__(
        self,
        basefolder: str = 'IAM/',
        subset: str = 'train',
        fixed_size: tuple =(128, None),
        transforms: list = None,
        character_classes: list = None,
        config=None,
        two_views: bool = False,
        ):
```

*   `basefolder` (str): Root folder containing `train/`, `val/` and `test/`.
*   `subset` (str): Portion of the dataset to load (`'train'`, `'val'`, `'test'`, `'all'`, `'train_val'`).
*   `fixed_size` (tuple): `(height, width)` used to resize images.
*   `transforms` (list | None): Optional Albumentations pipeline.
*   `character_classes` (list | None): Characters making up the vocabulary.
*   `config` (Any): Optional configuration object with alignment parameters.
*   `two_views` (bool): Return two augmented views when `True`.
*   `word_prob_mode` (str): Determines how unique-word priors are computed.
  • 'empirical' – probabilities from dataset counts
  • 'wordfreq'  – probabilities from the external wordfreq corpus (normalised)
**Attributes:**
*   `data` (list[tuple]): Pairs of image paths and transcriptions.
*   `transcriptions` (list[str]): Text strings for each image.
*   `character_classes` (list[str]): Dataset vocabulary of characters.
*   `prior_char_probs` (dict): Prior probabilities for each character in the vocabulary.
*   `unique_words` (list[str]): Unique words present in the dataset.
*   `unique_word_probs` (list[float]): Prior probability of each `unique_word`; source controlled by `word_prob_mode`.
*   `unique_word_embeddings` (torch.Tensor): Embeddings for the unique words.
*   `is_in_dict` (torch.IntTensor): ``1`` if a transcription is in `unique_words`.
*   `aligned` (torch.IntTensor): Alignment indices or ``-1`` when unknown.
    If ``aligned[i] = k`` and ``k != -1``, ``image[i]`` is aligned with ``unique_words[k]``.
*   `word_prob_mode` (str): How `unique_word_probs` are computed (`empirical` or `wordfreq`).

**Methods:**
*   `__len__()` -> int: Dataset size.
*   `__getitem__(index)` -> tuple: Returns processed image(s), text and alignment id.
*   `letter_priors(transcriptions=None, n_words=50000)`: returns prior probabilities for characters.
*   `find_word_embeddings(word_list, n_components=512)`: returns tensor of embeddings.
*   `save_image(index, out_dir, filename=None)`: saves a preprocessed image to disk.
*   `external_word_histogram(save_dir='tests/figures', filename='external_word_hist.png', dpi=200)`: saves a bar plot of unique-word usage.
*   `word_frequencies(mode=None)` -> (unique_words, probs): Returns the dataset
    vocabulary and its probability vector. When mode is None (default) the
    computation honours `self.word_prob_mode`.




#### PretrainingHTRDataset

Located in: `htr_base/utils/htr_dataset.py`

Lightweight dataset for image-only pretraining.

```python
class PretrainingHTRDataset(Dataset):
    def __init__(
        self,
        list_file: str = '/gpu-data3/pger/handwriting_rec/mnt/ramdisk/max/90kDICT32px/imlist.txt',
        fixed_size: tuple = (64, 256),
        base_path: str = '/gpu-data3/pger/handwriting_rec/mnt/ramdisk/max/90kDICT32px',
        transforms: list = None,
        n_random: int = None,
        random_seed: int = 0,
        preload_images: bool = False,
    ):
```

*   `list_file` (str): Path to a text file with relative image paths.
*   `fixed_size` (tuple): `(height, width)` for resizing.
*   `base_path` (str): Root directory prepended to each path in `list_file`.
*   `transforms` (list | None): Optional Albumentations pipeline.
*   `n_random` (int | None): If given, keep only `n_random` entries.
*   `random_seed` (int): Seed controlling the random subset selection.
*   `preload_images` (bool): Load all images into memory on init.



**Attributes:**
*   `img_paths` (list[str]): Absolute paths to images.
*   `transcriptions` (list[str]): Corresponding labels.
*   `preload_images` (bool): Whether images were loaded into memory.

**Methods:**
*   `process_paths(filtered_list)` -> tuple: Returns absolute paths and labels.
*   `__len__()` -> int: Number of items.
*   `__getitem__(index)` -> tuple: Returns an image tensor and transcription.
*   `save_image(index, out_dir, filename=None)` -> str: Save preprocessed image.
*   `loaded_image_shapes()` -> List[tuple]: Shapes of cached images.


#### FusedHTRDataset

Located in: `htr_base/utils/htr_dataset.py`

Combine real and synthetic datasets.

```python
class FusedHTRDataset(HTRDataset, PretrainingHTRDataset):
    def __init__(
        self,
        real_ds: HTRDataset,
        syn_ds: PretrainingHTRDataset,
        n_aligned: int = 0,
        random_seed: int = 0,
    ):
```

*   `real_ds` (HTRDataset): Real dataset to draw ground-truth images from.
*   `syn_ds` (PretrainingHTRDataset): Synthetic dataset providing extra images.
*   `n_aligned` (int): Number of real samples to align initially.
*   `random_seed` (int): Seed controlling which real samples are aligned.

**Attributes:**
*   `data` (list[tuple]): Combined `(path, transcription)` pairs.
*   `transcriptions` (list[str]): Labels for all images.
*   `unique_words` (list[str]): Joint vocabulary of both datasets.
*   `unique_word_probs` (list[float]): Probability vector for `unique_words`.
*   `unique_word_embeddings` (torch.Tensor): Embeddings for `unique_words`.
*   `real_word_indices` (torch.IntTensor): Indices of `unique_words` from the real dataset.
*   `synth_word_indices` (torch.IntTensor): Indices of `unique_words` from the synthetic dataset.
*   `aligned` (torch.IntTensor): Alignment index for each sample.
*   `weights` (torch.FloatTensor): Per-sample curriculum weights ∈ [0, 1].
*   `character_classes` (list[str]): Characters from `real_ds`.
*   `prior_char_probs` (dict): Character priors copied from `real_ds`.
*   `word_prob_mode` (str): Probability mode copied **and locked** from real_ds.
*   `_is_real` (torch.BoolTensor): Mask of real samples.
*   `_is_syn` (torch.BoolTensor): Mask of synthetic samples.
*   `basefolder` (str | None): Root folder from the real dataset.
*   `fixed_size` (tuple | None): Target image size from the real dataset.
*   `config` (Any | None): Configuration object forwarded from the real dataset.

When used with `align_more_instances`, only words from `real_ds` are considered
for new pseudo-labels. Synthetic samples keep their provided labels.

**Methods:**
*   `__len__()` -> int: Total number of items.
*   `__getitem__(index)` -> tuple: Item as provided by underlying datasets plus the sample weight.
*   `set_weights(indices, values)` -> None: Update sample weights in-place.
*   `word_frequencies` -> -> (unique_words, probs):
    Returns the joint vocabulary and probability vector.
    If mode is supplied it **must equal** self.word_prob_mode; otherwise a
    ValueError is raised.



### Alignment Utilities

#### ProjectionAligner

Located in: `alignment/alignment_utilities.py`

Helper class implementing pseudo-labelling based on projector outputs only.

```python
class ProjectionAligner:
    def __init__(
        self,
        dataset: FusedHTRDataset,
        backbone: HTRNet,
        projectors: Sequence[nn.Module],
        *,
        batch_size: int = 512,
        device: str = cfg.device,
        k: int = 0,
        debug_checks: bool = True,
        agree_threshold: int = 1,
        use_agreement: bool = False,
    ) -> None:
```

*   `dataset` (FusedHTRDataset): Dataset providing images and alignment info.
*   `backbone` (HTRNet): Visual backbone network used for descriptors.
*   `projectors` (Sequence[nn.Module]): Ensemble mapping descriptors to the word embedding space.
*   `batch_size` (int): Mini-batch size during descriptor harvesting.
*   `device` (str): Device used for alignment.
*   `k` (int): Number of least-moved descriptors to pseudo-label.
*   `debug_checks` (bool): Whether to run extra sanity checks.
*   `agree_threshold` (int): Require this many projector votes for a word.
*   `use_agreement` (bool): Enable agreement-filtering when ``True``.

**Attributes:**
*   `dataset` (FusedHTRDataset): Stored dataset with alignment information.
*   `backbone` (HTRNet): Backbone used for feature extraction.
*   `projectors` (Sequence[nn.Module]): Ensemble of projector networks.
*   `batch_size` (int): Mini-batch size when harvesting descriptors.
*   `device` (str): Device on which computations run.
*   `k` (int): Number of samples to pseudo-label per call.
*   `word_embeddings` (torch.Tensor): Word embedding matrix on ``device``.
*   `real_word_indices` (torch.IntTensor): Indices of real words in the vocabulary.
*   `synth_word_indices` (torch.IntTensor): Indices of synthetic-only words.
*   `is_real` (torch.BoolTensor): Mask of real dataset items.
*   `is_syn` (torch.BoolTensor): Mask of synthetic dataset items.
*   `debug_checks` (bool): Whether runtime assertions are enabled.
*   `agree_threshold` (int): Minimum votes needed for a consensus.
*   `use_agreement` (bool): Whether agreement filtering is active.

**Methods:**
*   `_get_projector_features() -> Tuple[torch.Tensor, torch.Tensor]`: Harvests
    descriptors and returns projector outputs and alignment flags.
*   `align() -> Tuple[torch.Tensor, torch.Tensor]`: Pseudo-labels unaligned samples and
    returns mean projector features and moved distances.
*   `align_agreement() -> Tuple[torch.Tensor, torch.Tensor]`: Pseudo-labels using agreement filtering.
*   `_assert_alignment_invariants(prev_aligned, prev_real_vocab, prev_syn_vocab, vocab_size_before) -> None`: Ensure dataset integrity after alignment.
* `csls_euclidean(projections, K) -> torch.Tensor*`: Returns a `(N, V)` matrix of CSLS (Euclidean) similarities, shifted to the 0-1 range. `projections` is a tensor of projector outputs; `K` is the neighbourhood size. Columns belonging to synthetic words are masked with `∞`.
*   `_log_results(new_indices: torch.Tensor) -> None`:  Prints the current accuracy for the epoch we are on. Also prints 5 random predictions (index, predicted word, ground truth word) from the newly pseudo-labelled real samples for the current round.




#### align_more_instances

Located in: `alignment/alignment_utilities.py`

Automatically assigns dataset images to unique words via projector outputs. This is a wrapper over `ProjectionAligner` (no second OT pass). When `pseudo_label_validation.enable` is set in the configuration, it calls `validate_pseudo_labels` after alignment once `align_more_instances` has been invoked at least `start_iteration` times.

```python
def align_more_instances(
    dataset: FusedHTRDataset,
    backbone: HTRNet,
    projectors: Sequence[nn.Module],
    *,
    batch_size: int = 512,
    device: str = cfg.device,
    k: int = 0,
    use_agreement: bool = False,
    agree_threshold: int = 1,
    debug_checks: bool = True,
    **_ignored,
) -> Tuple[torch.Tensor, torch.Tensor]:
```

*   `dataset` (FusedHTRDataset): Dataset providing images and `unique_word_embeddings`.
*   `backbone` (HTRNet): `HTRNet` used to extract visual descriptors.
*   `projectors` (Sequence[nn.Module]): List of projectors mapping descriptors to the embedding space.
*   `batch_size` (int): Mini-batch size when harvesting descriptors.
*   `device` (str): Device used for feature extraction, descriptor processing and the projector.
*   `k` (int): Number of least-moved descriptors to pseudo-label.
*   `use_agreement` (bool): Enable agreement-filtering when `True`.
*   `agree_threshold` (int): Minimum number of agreeing projectors for a pseudo-label.
*   `debug_checks` (bool): Whether to run extra sanity checks.


**Returns:**
*   `torch.Tensor`: Mean projector descriptors for the dataset.
*   `torch.Tensor`: Distance moved by each descriptor.

#### harvest_backbone_features

Located in: `alignment/alignment_utilities.py`

Harvests image descriptors from the backbone for each dataset item and stores their current alignment labels.

```python
def harvest_backbone_features(
    dataset: HTRDataset,
    backbone: HTRNet,
    *,
    batch_size: int = 64,
    num_workers: int = 0,
    device: torch.device = torch.device(cfg.device),
) -> tuple[torch.Tensor, torch.Tensor]:
```

*   `dataset` (HTRDataset): Dataset providing images and `aligned` flags.
*   `backbone` (HTRNet): Network used to compute descriptors.
*   `batch_size` (int): Mini-batch size during feature extraction.
*   `num_workers` (int): Data loader workers used while harvesting.
*   `device` (torch.device | str): Computation device for feature extraction.

**Returns:**
*   `torch.Tensor`: Tensor of descriptors with shape `(N, D)` where `N` is the dataset size.
*   `torch.Tensor`: Alignment tensor of shape `(N,)` copied from the dataset.


### Loss Functions

#### ProjectionLoss

Located in: `alignment/losses.py`

Combines Optimal Transport with a supervised distance term for descriptors that have known word alignments.

```python
class ProjectionLoss(torch.nn.Module):
    def __init__(
        self,
        reg: float = 0.1,
        *,
        unbalanced: bool = False,
        reg_m: float = 1.0,
        supervised_weight: float = 1.0,
        **sinkhorn_kwargs,
    ):
```

*   `reg` (float): Entropic regularisation strength.
*   `unbalanced` (bool): If `True` use the unbalanced OT formulation.
*   `reg_m` (float): Unbalanced mass regularisation.
*   `supervised_weight` (float): Scale for the supervised descriptor distance term.
*   `sinkhorn_kwargs` (dict): Extra keyword arguments forwarded to the solver.
**Attributes:**
*   `reg` (float): OT regularisation strength.
*   `unbalanced` (bool): Flag for unbalanced OT formulation.
*   `reg_m` (float): Mass regularisation when unbalanced.
*   `supervised_weight` (float): Weight for supervised term.

**Methods:**
*   `forward(descriptors, word_embeddings, aligned, tgt_probs, weights=None) -> torch.Tensor`: returns the combined weighted loss.
*   `train_projector` restricts the OT computation to the real dataset vocabulary by
    creating a subset tensor of word embeddings and re-normalised probabilities.


#### SoftContrastiveLoss

Located in: `alignment/losses.py`

InfoNCE‑style loss that pulls together image descriptors whose transcripts have small Levenshtein distance.

```python
class SoftContrastiveLoss(torch.nn.Module):
    def __init__(self, tau: float = .07, T_txt: float = 1.0, eps: float = 1e-8):
```

*   `tau` (float): Temperature in image space (distance → similarity).
*   `T_txt` (float): Temperature in transcript space (controls softness).
*   `eps` (float): Numeric stability.
**Attributes:**
*   `tau` (float): Temperature for descriptor similarity.
*   `T_txt` (float): Temperature for transcript similarity.
*   `eps` (float): Numerical stability term.

**Methods:**
*   `forward(feats, targets, lengths) -> torch.Tensor`: Computes the contrastive loss; no side effects.
### Metrics

#### CER

Located in: `htr_base/utils/metrics.py`

Character error rate accumulator.

```python
class CER:
    def __init__(self) -> None:
```

*   No arguments. Initializes counters to zero.

**Attributes:**
*   `total_dist` (float): Accumulated edit distance.
*   `total_len` (int): Total target length.

**Methods:**
*   `update(prediction, target) -> None`: Adds edit distance of a new sample.
*   `score() -> float`: Returns current CER.
*   `reset() -> None`: Clears accumulated statistics.

#### WER

Located in: `htr_base/utils/metrics.py`

Word error rate metric with optional tokenization.

```python
class WER:
    def __init__(self, mode: str = 'tokenizer') -> None:
```

*   `mode` (str): Either `tokenizer` or `space` controlling tokenization.

**Attributes:**
*   `mode` (str): Tokenization mode.
*   `total_dist` (float): Cumulative edit distance.
*   `total_len` (int): Total number of reference words.

**Methods:**
*   `update(prediction, target) -> None`: Update statistics with one sample.
*   `score() -> float`: Return current WER.
*   `reset() -> None`: Reset internal counters.





### CTC Utilities

#### encode_for_ctc

Located in: `alignment/ctc_utils.py`

Converts a batch of raw string transcriptions to the `(targets, lengths)` format expected by `nn.CTCLoss`.

```python
def encode_for_ctc(
    transcriptions: List[str],
    c2i: Dict[str, int],
    device: torch.device | str = None
) -> Tuple[torch.IntTensor, torch.IntTensor]:
```

*   `transcriptions` (list[str]): Each element is a single line/word already wrapped with leading and trailing spaces.
*   `c2i` (dict[str, int]): Character-to-index mapping where index 0 is reserved for CTC blank.
*   `device` (torch.device, optional): If given, the returned tensors are moved to this device.

**Returns:**
*   `targets` (torch.IntTensor): All label indices concatenated in batch order.
*   `lengths` (torch.IntTensor): The original length (in characters) of every element in `transcriptions`.

#### _ctc_loss_fn

Located in: `alignment/losses.py`

A thin wrapper around `torch.nn.functional.ctc_loss` that takes `logits`.

```python
def _ctc_loss_fn(
    logits: torch.Tensor,
    targets: torch.IntTensor,
    inp_lens: torch.IntTensor,
    tgt_lens: torch.IntTensor,
) -> torch.Tensor:
```

*   `logits` (torch.Tensor): Input logits from the CTC head.
*   `targets` (torch.IntTensor): Ground truth targets for CTC loss.
*   `inp_lens` (torch.IntTensor): Lengths of the input sequences.
*   `tgt_lens` (torch.IntTensor): Lengths of the target sequences.

**Returns:**
*   `torch.Tensor`: Per-sample CTC loss values.

#### _unflatten_targets

Located in: `alignment/ctc_utils.py`

Converts flattened CTC targets to a list of lists.

```python
def _unflatten_targets(targets: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
```

*   `targets` (torch.Tensor): A 1D tensor of concatenated label indices.
*   `lengths` (torch.Tensor): A 1D tensor specifying the length of each sequence in the batch.

**Returns:**
*   `list[list[int]]`: A list of lists, where each inner list is a sequence of integer labels.

#### greedy_ctc_decode

Located in: `alignment/ctc_utils.py`

Greedy-decodes a batch of CTC network outputs.

```python
def greedy_ctc_decode(
    logits: torch.Tensor,
    i2c: Dict[int, str],
    blank_id: int = 0,
    time_first: bool = True,
) -> List[str]:
```

*   `logits` (torch.Tensor): Tensor shaped either `(T, B, C)` if `time_first` is `True` or `(B, T, C)` if `time_first` is `False`.
*   `i2c` (dict[int, str]): Index-to-character mapping that complements the `c2i` used during encoding. It must not contain the blank id.
*   `blank_id` (int, optional): Integer assigned to the CTC blank (defaults to 0).
*   `time_first` (bool, optional): Set `True` if logits are `(T, B, C)`; otherwise set `False` for `(B, T, C)`.

**Returns:**
*   `List[str]`: One decoded string per element in the mini-batch.

#### beam_search_ctc_decode

Located in: `alignment/ctc_utils.py`

Beam-search decoding for CTC outputs using `pyctcdecode`.

```python
def beam_search_ctc_decode(
    logits: torch.Tensor,
    i2c: Dict[int, str],
    *,
    beam_width: int = 10,
    blank_id: int = 0,
    time_first: bool = True,
) -> List[str]:
```

*   `logits` (torch.Tensor): Network output – either `(T, B, C)` if `time_first` or `(B, T, C)`.
*   `i2c` (dict[int, str]): Index-to-character map excluding the blank id.
*   `beam_width` (int, optional): Number of prefixes kept after every time-step.
*   `blank_id` (int, optional): Integer assigned to the CTC blank (defaults to `0`).
*   `time_first` (bool, optional): `True` if `logits` are `(T, B, C)`, else `False` for `(B, T, C)`.

**Returns:**
*   `list[str]`: Best-scoring transcription for every element in the mini-batch.

#### ctc_target_probability

Located in: `alignment/ctc_utils.py`

Probability that a single logit sequence collapses to a given transcription.

```python
def ctc_target_probability(
    logits: torch.Tensor,
    target: str,
    c2i: Dict[str, int],
    *,
    blank_id: int = 0,
) -> float:
```

*   `logits` (torch.Tensor): Two-dimensional `(T, C)` tensor with raw scores for one sample.
*   `target` (str): String whose probability we wish to compute.
*   `c2i` (dict[str, int]): Character-to-index map used to encode the string.
*   `blank_id` (int, optional): Index of the CTC blank (defaults to `0`).

**Returns:**
*   `float`: Probability that `logits` decode to `target` according to CTC.

### Plotting Utilities

#### plot_dataset_augmentations

Located in: `alignment/plot.py`

Saves a figure showing three images and their augmentations side by side.

```python
def plot_dataset_augmentations(dataset: HTRDataset, save_path: str) -> None:
```

*   `dataset` (HTRDataset): Dataset providing images and augmentation transforms.
*   `save_path` (str): Where to write the PNG figure.

#### plot_tsne_embeddings

Located in: `alignment/plot.py`

Generates a coloured t-SNE plot of backbone embeddings and saves it.

```python
def plot_tsne_embeddings(
    dataset: HTRDataset,
    backbone: HTRNet,
    save_path: str,
    *,
    device: torch.device = torch.device(cfg.device),
    n_samples: int = 1000,
) -> None:
```

*   `dataset` (HTRDataset): Dataset instance providing the images.
*   `backbone` (HTRNet): The visual backbone model to extract embeddings from.
*   `save_path` (str): Path where the generated t-SNE plot (PNG image) will be saved.
*   `device` (torch.device | str): Device on which the backbone runs.
*   `n_samples` (int): The number of random samples to visualise. Defaults to 1000.

#### plot_projector_tsne

Located in: `alignment/plot.py`

Plots t-SNE of projector outputs and word embeddings.

```python
def plot_projector_tsne(
    projections: torch.Tensor, dataset: HTRDataset, save_path: str
) -> None:
```

*   `projections` (torch.Tensor): Output of the projector with shape `(N, E)`.
*   `dataset` (HTRDataset): Provides `unique_word_embeddings` of shape `(V, E)`.
*   `save_path` (str): Destination path for the PNG figure.

#### plot_pretrained_backbone_tsne

Located in: `alignment/plot.py`

Plots t-SNE embeddings from the pretrained backbone.

```python
def plot_pretrained_backbone_tsne(dataset: HTRDataset, n_samples: int, save_path: str) -> None:
```

*   `dataset` (HTRDataset): Dataset instance providing images and alignment labels.
*   `n_samples` (int): Number of random samples to visualise.
*   `save_path` (str): Path where the PNG figure will be saved.

### Vocabulary Utilities

#### create_vocab

Located in: `htr_base/utils/vocab.py`

Creates the default vocabulary pickles and returns the dictionaries.

```python
def create_vocab() -> Tuple[Dict[str, int], Dict[int, str]]:
```

**Returns:**
*   `Tuple[Dict[str, int], Dict[int, str]]`: A tuple containing the character-to-index (`c2i`) and index-to-character (`i2c`) dictionaries.

#### load_vocab

Located in: `htr_base/utils/vocab.py`

Loads `c2i` and `i2c` dictionaries from `saved_models`. If the pickle files do not exist, `create_vocab` is called to generate them first.

```python
def load_vocab() -> Tuple[Dict[str, int], Dict[int, str]]:
```

**Returns:**
*   `Tuple[Dict[str, int], Dict[int, str]]`: A tuple containing the character-to-index (`c2i`) and index-to-character (`i2c`) dictionaries.

### Training Utilities

* `pretraining.py` – trains a CTC-based network on the synthetic 90k dataset.


#### refine_visual_backbone

Located in: `alignment/trainer.py`

Refines the backbone on aligned instances from a `FusedHTRDataset`.
Pseudo‑labels come from ``dataset.unique_words`` using the ``aligned`` indices
instead of the ground‑truth transcriptions.

```python
def refine_visual_backbone(
    dataset: FusedHTRDataset,
    backbone: nn.Module,
    *,
    batch_size: int,
    epochs: int,
    device: torch.device,
    logger: Logger | None = None,
) -> None:
```

```python
refine_visual_backbone(fused_ds, backbone, batch_size=32, epochs=1, device=torch.device("cuda"))
```

#### _shuffle_batch

Located in: `alignment/trainer.py`

Randomly permutes an image tensor and list of transcriptions with the same order.

```python
def _shuffle_batch(images: torch.Tensor, words: List[str]) -> Tuple[torch.Tensor, List[str]]:
```

```python
imgs, texts = _shuffle_batch(imgs, texts)
```

#### _aligned_to_words

Located in: `alignment/trainer.py`

Convert a tensor of alignment indices into pseudo‑labels using the dataset
vocabulary.

```python
def _aligned_to_words(aligned: torch.Tensor, vocab: List[str]) -> List[str]:
```

```python
pseudo = _aligned_to_words(torch.tensor([0, 1]), ["a", "b"])
```

#### _parse_pseudo_files

Located in: `tests/train_pseudo_labels.py`

Parse pseudo-label log files and count how many predictions match the original ground truth.

```python
def _parse_pseudo_files(results_dir: str) -> Tuple[Dict[int, str], int]:
```

```python
mapping, correct = _parse_pseudo_files("results")
```

#### train_projector

Located in: `alignment/trainer.py`

Optimises a projector to map descriptors onto the word embedding space.

```python
def train_projector(
    dataset: FusedHTRDataset | HTRDataset,
    backbone: HTRNet,
    projector: nn.Module | List[nn.Module],
    num_epochs: int = cfg.projector_epochs,
    batch_size: int = cfg.projector_batch_size,
    lr: float = cfg.projector_lr,
    num_workers: int = cfg.projector_workers,
    weight_decay: float = cfg.projector_weight_decay,
    device: torch.device | str = cfg.device,
    plot_tsne: bool = cfg.plot_tsne,
) -> None:
```

```python
train_projector(ds, backbone, proj)
```

#### alternating_refinement

Located in: `alignment/trainer.py`

Runs alternating cycles of backbone refinement, projector training and pseudo‑labelling. Initial seed alignments are saved to `pseudo_labels_round_0.txt` before the loop starts. After each round it prints the CER on both the training and test sets. `dataset` must be a `FusedHTRDataset` that merges the real corpus with a synthetic pre‑training set; the synthetic portion is considered pre‑aligned.

```python
def alternating_refinement(
    dataset: FusedHTRDataset,
    backbone: HTRNet,
    projectors: List[nn.Module],
    *,
    rounds: int = cfg.alt_rounds,
    backbone_epochs: int = cfg.refine_epochs,
    projector_epochs: int = cfg.projector_epochs,
    refine_kwargs: dict | None = None,
    projector_kwargs: dict | None = None,
    align_kwargs: dict | None = None,
) -> None:
```

```python
alternating_refinement(ds, backbone, projectors)
```

### Configuration Files

#### trainer_config.yaml

Located in: `alignment/alignment_configs/trainer_config.yaml`

Hyperparameters for backbone refinement, projector training, and overall alignment.

**Key parameters:**
*   `refine_batch_size` (int): Mini-batch size for backbone fine-tuning.
*   `refine_lr` (float): Learning rate during backbone refinement.
*   `refine_main_weight` (float): Weight for the main CTC loss branch.
*   `refine_aux_weight` (float): Weight for the auxiliary CTC loss.
*   `refine_epochs` (int): Epochs spent refining the backbone.
*   `syn_batch_ratio` (float): Fraction of each batch drawn from `PretrainingHTRDataset`.
*   `enable_phoc` (bool): Turn PHOC loss on/off.
*   `phoc_levels` (list): Descriptor levels for PHOC.
*   `phoc_loss_weight` (float): Scaling factor for the PHOC loss.
*   `contrastive_enable` (bool): Turn on to activate the soft-contrastive loss.
*   `contrastive_weight` (float): Scales the soft-contrastive loss inside total_loss.
*   `contrastive_tau` (float): Temperature for descriptor similarities.
*   `contrastive_text_T` (float): Softness in edit-distance space.
*   `architecture` (dict): Defines the HTRNet backbone parameters (e.g., `cnn_cfg`, `head_type`, `rnn_type`, `feat_dim`, `feat_pool`, `phoc_levels`).
*   `projector_epochs` (int): Number of epochs for the projector network.
*   `projector_batch_size` (int): Mini-batch size when collecting descriptors.
*   `projector_lr` (float): Learning rate for projector optimisation.
*   `projector_workers` (int): Dataloader workers used during descriptor caching.
*   `projector_weight_decay` (float): Weight decay for the projector optimiser.
*   `plot_tsne` (bool): Save t-SNE plots during projector training.
*   `device` (str): Target device for training (e.g., `'cuda'`, `'cpu'`).
*   `gpu_id` (int): Which CUDA device index to use.
*   `alt_rounds` (int): Number of backbone/projector cycles per pass.
*   `align_batch_size` (int): Mini-batch size when harvesting descriptors for alignment.
*   `align_device` (str): Device used during alignment post-processing.
*   `align_k` (int): Pseudo-label this many least-moved descriptors.
*   `align_use_agreement` (bool): Enable agreement-filtering during alignment.
*   `eval_batch_size` (int): Mini-batch size during CER evaluation.
*   `dataset` (dict): Parameters for `HTRDataset` (e.g., `basefolder`, `subset`, `fixed_size`, `n_aligned`, `word_emb_dim`, `two_views`).
*   `dataset.word_prob_mode` (str): `'empirical'` or `'wordfreq'` for word priors.
*   `n_aligned` (int): Number of initially aligned instances.
*   `ensemble_size` (int): Size of the projector ensemble.
*   `agree_threshold` (int): Minimum number of projectors that must agree for pseudo-labeling.
*   `supervised_weight` (int): Weight for supervised loss component.
*   `load_pretrained_backbone` (bool): Load weights for the backbone at startup.
*   `pretrained_backbone_path` (str): Path to the pretrained backbone model.
*   `synthetic_dataset` (dict): Parameters for `PretrainingHTRDataset` (e.g., `list_file`, `base_path`, `n_random`, `fixed_size`, `preload_images`, `random_seed`).
*   `pseudo_label_validation` (dict): Optional sanity check configuration with keys `enable`, `edit_distance` and `start_iteration`.

#### pretraining_config.yaml

Located in: `alignment/alignment_configs/pretraining_config.yaml`

Architecture and pretraining options used by `pretraining.py`.

**Key parameters:**
*   `enable_phoc` (bool): Turn PHOC loss on/off.
*   `phoc_levels` (list): Same default as HTRNet.
*   `phoc_loss_weight` (float): Loss scaling factor.
*   `architecture` (dict): Defines the HTRNet backbone parameters (e.g., `cnn_cfg`, `head_type`, `rnn_type`, `feat_dim`, `feat_pool`, `phoc_levels`).
*   `contrastive_enable` (bool): Turn on to activate the soft-contrastive loss.
*   `contrastive_weight` (float): Scales the new loss inside total_loss.
*   `contrastive_tau` (float): Temperature for descriptor similarities.
*   `contrastive_text_T` (float): Softness in edit-distance space.
*   `load_pretrained_backbone` (bool): Load weights for the backbone at startup.
*   `pretrained_backbone_path` (str): Path to the pretrained backbone model.
*   `device` (str): Target device.
*   `gpu_id` (int): CUDA device index.
*   `list_file` (str): Path to the image list file.
*   `train_set_size` (int): Number of random training images.
*   `test_set_size` (int): Number of random test images.
*   `batch_size` (int): Batch size.
*   `num_epochs` (int): Number of training epochs.
*   `learning_rate` (float): Learning rate.
*   `fixed_size` (list): Fixed size for images `[height, width]`.
*   `base_path` (str): Base path for images (defaults to list_file directory if null).
*   `use_augmentations` (bool): Whether to use augmentations.
*   `main_loss_weight` (float): Weight for the main CTC loss.
*   `aux_loss_weight` (float): Weight for the auxiliary CTC loss.
*   `save_path` (str): Path to save the trained model.
*   `save_backbone` (bool): Whether the trained backbone should be saved.
*   `results_file` (bool): Whether to save results to a file.
*   `dataset.word_prob_mode` (str): `'empirical'` or `'wordfreq'` for word priors.

## Requirements
Python dependencies are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

## Knowledge Graph
Run `python scripts/generate_knowledge_graph.py` to create `knowledge_graph.graphml` describing module and function relationships. See `docs/knowledge_graph.md` for details.

The script exposes `build_repo_graph(root_dirs, graphml_path)` which returns the generated file path.

## License
This project is released under the MIT license.