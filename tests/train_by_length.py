from __future__ import annotations
import os, sys, random
from types import SimpleNamespace
from pathlib import Path
# ------------------------------------------------------------------
# Hyper
# parameters controlling training and evaluation. The values
# below are largely defaults used when running this script directly.
#
#   gpu_id              – CUDA device identifier.
#   max_length/min_length – inclusive range of word lengths used for
#                            selecting training samples (spaces ignored).
#   eval_k              – length threshold used when reporting CER.
#   n_aligned           – maximum number of aligned ground
# truth words
#                          sampled for training.P
#   num_epochs          – number of fine
# tuning epochs.
#   batch_size          – mini
# batch size during training.
#   learning_rate       – optimiser learning rate.
#   main/aux_loss_weight – weights for the main and auxiliary CTC losses.
#   dataset_fixed_size  – target (H, W) used when resizing dataset images.
#   architecture_config – parameters defining the HTRNet architecture.
#   dataset_base_folder_name – dataset folder containing processed words.
#   figure_output_dir/filename – where to write diagnostic figures.
# ------------------------------------------------------------------
GPU_ID = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

MAX_LENGTH = 20
MIN_LENGTH = 4
EVAL_K = 4
N_ALIGNED = 800
NUM_EPOCHS = 600
BATCH_SIZE = 128
SYN_BATCH_RATIO = 0.7 # if 0 then we only use gt samples. If 1 then we use only synthetic samples
LEARNING_RATE = 1e-3
MAIN_LOSS_WEIGHT = 1.0
AUX_LOSS_WEIGHT = 0.1
DATASET_FIXED_SIZE = (64, 256)
ARCHITECTURE_CONFIG = {
    "cnn_cfg": [[2, 64], "M", [3, 128], "M", [2, 256]],
    "head_type": "both",
    "rnn_type": "gru",
    "rnn_layers": 3,
    "rnn_hidden_size": 256,
    "flattening": "maxpool",
    "stn": False,
    "feat_dim": None,
}
DATASET_BASE_FOLDER_NAME = "GW"
FIGURE_OUTPUT_DIR = "tests/figures"
FIGURE_FILENAME = "long.png"
LOAD_PRETRAINED_BACKBONE = True
DECODE_CONFIG = {
    "method": "beam",  # 'greedy' or 'beam'
    "beam_width": 3,
}
from typing import Dict, Tuple, List
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from collections import Counter
root = Path(__file__).resolve().parents[1]  # project root for local imports
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from htr_base.utils.htr_dataset import HTRDataset, PretrainingHTRDataset  # for the priors
from htr_base.models import HTRNet
from htr_base.utils.metrics import CER
from htr_base.utils.vocab import load_vocab
from htr_base.utils.transforms import aug_transforms
from alignment.ctc_utils import (
    encode_for_ctc,
    greedy_ctc_decode,
    beam_search_ctc_decode,
)
from alignment.losses import _ctc_loss_fn

# ---------------------------------------------------------------------
# Optional loading of a pretrained backbone before fine-tuning.
# ---------------------------------------------------------------------
def maybe_load_pretrained(net, device,
                          path="htr_base/saved_models/pretrained_backbone.pt") -> None:
    """Load *net* weights from ``path`` when ``LOAD_PRETRAINED_BACKBONE`` is ``True``."""
    if LOAD_PRETRAINED_BACKBONE:
        state = torch.load(path, map_location=device)
        print('Backbone is loaded')
        net.load_state_dict(state, strict=False)
# ---------------------------------------------------------------------
# Save a histogram of characters appearing in the provided strings to a
# PNG file.  Useful for quickly visualising the dataset distribution.
# ---------------------------------------------------------------------
def save_char_histogram_png(
    strings: List[str],
    output_dir: str = FIGURE_OUTPUT_DIR,
    filename: str = FIGURE_FILENAME
) -> None:
    """
    Build a histogram of all characters (a–z, 0–9, and space) appearing in the
    given list of strings, plot it using matplotlib, and save as a PNG in tests/figures.
    Parameters
    ----------
    strings : List[str]
        List of input strings to count characters from.
    output_dir : str
        Directory where the PNG will be written. Defaults to "tests/figures".
    filename : str
        Name of the output PNG file. Defaults to "char_histogram.png".
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Initialize histogram with all keys set to 0
    keys = [chr(c) for c in range(ord('a'), ord('z') + 1)] + \
           [str(d) for d in range(10)] + \
           [' ']
    histogram = {k: 0 for k in keys}
    # Count only valid characters (lowercased for letters)
    counter = Counter()
    for s in strings:
        for ch in s:
            if ch.isalpha():
                ch_lower = ch.lower()
                if 'a' <= ch_lower <= 'z':
                    counter[ch_lower] += 1
            elif ch.isdigit():
                counter[ch] += 1
            elif ch == ' ':
                counter[' '] += 1
            # ignore all other characters
    # Fill histogram with counts (zero for missing keys)
    counts = [int(counter.get(k, 0)) for k in keys]
    # Plot bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(keys, counts)
    plt.xlabel("Character")
    plt.ylabel("Count")
    plt.title("Character Frequency Histogram")
    plt.xticks(rotation=45)  # Rotate labels to improve readability (space will appear blank)
    plt.tight_layout()
    # Save figure
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()
    print(f"Character histogram PNG saved to: {output_path}")
# ---------------------------------------------------------------------
# Construct dictionaries that map characters to integer IDs and back. The
# blank symbol required by CTC occupies index 0.
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Evaluate character error rate on the given loader.  A few prediction
# examples are printed, as well as CER by word length and overall.
# ---------------------------------------------------------------------
def _evaluate_cer(model: HTRNet, loader: DataLoader, i2c: Dict[int, str],
                  device, k: int, show_max: int = 5,
                  *, seed: int | None = None) -> float:
    """Compute CER over *loader* and print a random selection of pairs.
    In addition to the global CER, compute per-word-length CERs and
    display the relative proportion of each length in the dataset.
    Also calculate and print CER for transcriptions with length <= k and > k.
    The number of samples contributing to each CER is printed as well.
    If ``seed`` is given, the random examples are reproducible.
    """
    model.eval()
    cer_less_equal_k = CER()
    cer_greater_k = CER()
    cer_total = CER()
    pairs: List[Tuple[str, str]] = []
    per_len: Dict[int, Tuple[CER, int]] = {}
    num_le_k = 0
    num_gt_k = 0
    num_total = 0
    chars_le_k = 0
    chars_gt_k = 0
    with torch.no_grad():
        for imgs, transcrs, _ in loader:
            imgs = imgs.to(device)
            logits = model(imgs, return_feats=False)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if DECODE_CONFIG.get("method", "greedy") == "beam":
                preds = beam_search_ctc_decode(
                    logits,
                    i2c,
                    beam_width=DECODE_CONFIG.get("beam_width", 10),
                )
            else:
                preds = greedy_ctc_decode(logits, i2c)
            for p, t in zip(preds, transcrs):
                gt_stripped = t.strip()
                pred_stripped = p.strip()
                pairs.append((gt_stripped, pred_stripped))
                cer_total.update(pred_stripped, gt_stripped)
                num_total += 1
                transcription_len_no_spaces = len(gt_stripped.replace(" ", ""))
                if transcription_len_no_spaces <= k:
                    cer_less_equal_k.update(pred_stripped, gt_stripped)
                    num_le_k += 1
                    chars_le_k += transcription_len_no_spaces
                else:
                    cer_greater_k.update(pred_stripped, gt_stripped)
                    num_gt_k += 1
                    chars_gt_k += transcription_len_no_spaces
                l = len(gt_stripped.replace(" ", ""))
                if l not in per_len:
                    per_len[l] = (CER(), 0)
                per_len[l][0].update(pred_stripped, gt_stripped)
                per_len[l] = (per_len[l][0], per_len[l][1] + 1)
    model.train()

    # Show a random subset of prediction examples
    rng = random.Random(seed) if seed is not None else random
    for gt, pr in rng.sample(pairs, k=min(show_max, len(pairs))):
        print(f"GT: '{gt}'\nPR: '{pr}'\n")
    print(
        f"\nCER for transcriptions with length <= {k}: {cer_less_equal_k.score():.4f} (n={num_le_k})"
    )
    print(
        f"CER for transcriptions with length > {k}: {cer_greater_k.score():.4f} (n={num_gt_k})"
    )
    print(f"Overall CER: {cer_total.score():.4f} (n={num_total})\n")
    print(
        f"Total characters for length <= {k}: {chars_le_k}"
    )
    print(
        f"Total characters for length > {k}: {chars_gt_k}\n"
    )
    total = sum(v[1] for v in per_len.values()) or 1
    for l in sorted(per_len):
        pct = 100 * per_len[l][1] / total
        print(f"[Eval] len={l:2d} ({pct:5.2f}%): CER={per_len[l][0].score():.4f}")
    return cer_total.score()


# ---------------------------------------------------------------------
# Fine-tune a visual model using only ground-truth words whose lengths fall
# within a specified range.  Evaluation is performed periodically using CER.
# ---------------------------------------------------------------------
def refine_visual_model(dataset: HTRDataset,
                        backbone: HTRNet,
                        num_epochs: int,
                        batch_size: int = 128,
                        lr: float = 1e-4,
                        main_weight: float = 1.0,
                        aux_weight: float = 0.1,
                        max_length: int = 4,
                        min_length: int = 0,
                        eval_k: int | None = None,
                        *,
                        pretrain_ds: PretrainingHTRDataset | None = None,
                        syn_batch_ratio: float | None = None) -> None:
    """Fine-tune *backbone* on a subset of ground-truth words.
    Only words whose length (ignoring spaces) lies in the inclusive range
    ``[min_length, max_length]`` are used for training.  At most ``n_aligned``
    such words are randomly selected.  ``eval_k`` controls the length
    threshold used when computing CER during evaluation.
    """
    device = next(backbone.parameters()).device
    print(
        f"[Refine] min_len={min_length} max_len={max_length} "
        f"epochs={num_epochs} batch={batch_size} lr={lr}"
    )
    # Build vocabulary
    c2i, i2c = load_vocab()
    # Test loader
    test_set = HTRDataset(dataset.basefolder, subset="test",
                          fixed_size=dataset.fixed_size, transforms=None,
                          config=dataset.config)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=(device.type == "cuda"))
    # Optimiser & scheduler
    opt = optim.AdamW(backbone.parameters(), lr=lr, weight_decay=1e-4)
    sched = lr_scheduler.StepLR(opt, step_size=150, gamma=0.5)
    n_aligned = getattr(dataset.config, "n_aligned", len(dataset))
    # how many ground-truth words to keep in the training subset
    # Pre-compute indices by length and build a fixed subset
    transcrs = [t.strip() for t in dataset.transcriptions]
    valid_idx = [
        i
        for i, t in enumerate(transcrs)
        if min_length <= len(t.replace(" ", "")) <= max_length
    ]
    subset_idx = random.sample(valid_idx, k=min(n_aligned, len(valid_idx)))
    # randomly choose the desired number of samples from the valid indices
    subset_ds = Subset(dataset, subset_idx)
    unique_words_in_subset = {transcrs[i].strip().lower() for i in subset_idx}
    print(
        f"[Refine] Training on {len(subset_idx)} samples, which correspond to {len(unique_words_in_subset)} unique transcriptions."
    )

    print("\nSelected transcriptions for training:")
    for i in subset_idx:
        print(f"  - {transcrs[i]}")
    print("-" * 20)

    pretrain_only = False
    if pretrain_ds is not None and syn_batch_ratio is not None:
        syn_bs = int(batch_size * syn_batch_ratio)
        gt_bs = batch_size - syn_bs
        print(f"Batch composition: HTRDataset samples: {gt_bs if gt_bs > 0 else 0}, PretrainingHTRDataset samples: {syn_bs if syn_bs > 0 else 0}")
        if syn_bs <= 0:
            train_loader = DataLoader(
                subset_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=(device.type == "cuda"),
            )
            pre_iter = None
        elif gt_bs <= 0:
            train_loader = DataLoader(
                pretrain_ds,
                batch_size=syn_bs,
                shuffle=True,
                num_workers=2,
                pin_memory=(device.type == "cuda"),
            )
            pre_iter = None
            pretrain_only = True
        else:
            train_loader = DataLoader(
                subset_ds,
                batch_size=gt_bs,
                shuffle=True,
                num_workers=2,
                pin_memory=(device.type == "cuda"),
            )
            pretrain_loader = DataLoader(
                pretrain_ds,
                batch_size=syn_bs,
                shuffle=True,
                num_workers=2,
                pin_memory=(device.type == "cuda"),
            )
            from itertools import cycle
            pre_iter = cycle(pretrain_loader)
    else:
        print(f"Batch composition: HTRDataset samples: {batch_size}, PretrainingHTRDataset samples: 0")
        train_loader = DataLoader(
            subset_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        pre_iter = None
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0; effective_batches = 0
        epoch_ctc = 0.0
        for batch in train_loader:
            if pretrain_only:
                imgs, trans = batch
            else:
                imgs, trans, _ = batch
            imgs = imgs.to(device)
            if pre_iter is not None:
                imgs_pt, trans_pt = next(pre_iter)
                imgs_pt = imgs_pt.to(device)
                imgs = torch.cat([imgs, imgs_pt], dim=0)
                trans = list(trans) + list(trans_pt)
            targets_s = [t if t.startswith(" ") else f" {t.strip()} " for t in trans]
            outputs = backbone(imgs, return_feats=False)
            if isinstance(outputs, (tuple, list)):
                main_logits = outputs[0]
                aux_logits = outputs[1] if len(outputs) > 1 else None
            else:
                main_logits = outputs
                aux_logits = None
            T, B, _ = main_logits.shape
            targets, tgt_lens = encode_for_ctc(targets_s, c2i, device="cpu")
            inp_lens = torch.full((B,), T, dtype=torch.int32)
            loss_m = _ctc_loss_fn(main_logits, targets, inp_lens, tgt_lens).mean()
            loss_a = (
                _ctc_loss_fn(aux_logits, targets, inp_lens, tgt_lens).mean()
                if aux_logits is not None
                else torch.tensor(0.0, device=main_logits.device)
            )
            total_loss = (
                main_weight * loss_m + aux_weight * loss_a
            )
            opt.zero_grad(set_to_none=True)
            total_loss.backward()
            opt.step()
            epoch_loss += total_loss.item()
            epoch_ctc += loss_m.item()
            effective_batches += 1
        sched.step()
        avg_loss = epoch_loss / max(1, effective_batches)
        avg_ctc = epoch_ctc / max(1, effective_batches)
        print(
            f"Epoch {epoch:03}/{num_epochs}  loss={avg_loss:.4f}  lr={sched.get_last_lr()[0]:.2e} "
            f"CTC={avg_ctc:.4f}"
        )
        if (epoch + 1) % 20 == 0 or epoch == num_epochs:
            k_eval = eval_k if eval_k is not None else max_length
            cer = _evaluate_cer(backbone, test_loader, i2c, device, k=k_eval)
            print(f"[Eval] CER @ epoch {epoch}: {cer:.4f}")
    print("[Refine] finished.")
def count_transcriptions_by_length(dataset: HTRDataset, k: int) -> int:
    """Counts the number of transcriptions in the dataset with length <= k."""
    count = 0
    for transcription in dataset.transcriptions:
        if len(transcription.strip().replace(" ", "")) <= k:
            count += 1
    return count

if __name__ == "__main__":
    proj_root = Path(__file__).resolve().parents[1]  # repository root
    gw_folder = proj_root / "htr_base" / "data" / DATASET_BASE_FOLDER_NAME / "processed_words"
    if not gw_folder.exists():
        raise RuntimeError("GW processed dataset not found – generate it first!")
    class DummyCfg:
        def __init__(self):
            self.n_aligned = N_ALIGNED

    train_set = HTRDataset(
        str(gw_folder),
        subset="train_val",
        fixed_size=DATASET_FIXED_SIZE,
        transforms=aug_transforms,
        config=DummyCfg(),
    )
    corp_root = Path("/gpu-data3/pger/handwriting_rec/mnt/ramdisk/max/90kDICT32px")
    list_file = corp_root / "imlist.txt"
    pretrain_set = PretrainingHTRDataset(
        str(list_file),
        fixed_size=DATASET_FIXED_SIZE,
        base_path=str(corp_root),
        transforms=aug_transforms,
        n_random=10000,
        preload_images=False
    )
    c2i, _ = load_vocab()
    arch_cfg_dict = ARCHITECTURE_CONFIG
    net = HTRNet(SimpleNamespace(**arch_cfg_dict), nclasses=len(c2i) + 1)
    net.to("cuda")
    maybe_load_pretrained(
        net,
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    refine_visual_model(
        train_set,
        net,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        main_weight=MAIN_LOSS_WEIGHT,
        aux_weight=AUX_LOSS_WEIGHT,
        max_length=MAX_LENGTH,
        min_length=MIN_LENGTH,
        eval_k=EVAL_K,
        pretrain_ds=pretrain_set,
        syn_batch_ratio=SYN_BATCH_RATIO,
    )
