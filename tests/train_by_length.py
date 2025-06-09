from __future__ import annotations
import os, sys, random
from types import SimpleNamespace  
# ------------------------------------------------------------------
# Hyper‑parameters controlling training and evaluation. The values
# below are largely defaults used when running this script directly.
#
#   gpu_id              – CUDA device identifier.
#   max_length/min_length – inclusive range of word lengths used for
#                            selecting training samples (spaces ignored).
#   eval_k              – length threshold used when reporting CER.
#   n_aligned           – maximum number of aligned ground‑truth words
#                          sampled for training.
#   k_external_words    – number of external words to align against.
#   num_epochs          – number of fine‑tuning epochs.
#   batch_size          – mini‑batch size during training.
#   learning_rate       – optimiser learning rate.
#   main/aux_loss_weight – weights for the main and auxiliary CTC losses.
#   dataset_fixed_size  – target (H, W) used when resizing dataset images.
#   architecture_config – parameters defining the HTRNet architecture.
#   dataset_base_folder_name – dataset folder containing processed words.
#   figure_output_dir/filename – where to write diagnostic figures.
# ------------------------------------------------------------------
HP = {
    "gpu_id": "0",
    "max_length": 4,
    "min_length": 0,
    "eval_k": 4,
    "n_aligned": 500,
    "k_external_words": 200,
    "num_epochs": 600,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "main_loss_weight": 1.0,
    "aux_loss_weight": 0.1,
    "dataset_fixed_size": (64, 256),
    "architecture_config": {
        "cnn_cfg": [[2, 64], "M", [3, 128], "M", [2, 256]],
        "head_type": "both",
        "rnn_type": "gru",
        "rnn_layers": 3,
        "rnn_hidden_size": 256,
        "flattening": "maxpool",
        "stn": False,
        "feat_dim": None,
    },
    "dataset_base_folder_name": "GW",
    "figure_output_dir": "tests/figures",
    "figure_filename": "long.png",
}
os.environ["CUDA_VISIBLE_DEVICES"] = HP['gpu_id']
from pathlib import Path
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
from htr_base.utils.htr_dataset import HTRDataset
from htr_base.models import HTRNet
from htr_base.utils.metrics import CER
from htr_base.utils.transforms import aug_transforms
from alignment.ctc_utils import encode_for_ctc, greedy_ctc_decode
# ---------------------------------------------------------------------
# Save a histogram of characters appearing in the provided strings to a
# PNG file.  Useful for quickly visualising the dataset distribution.
# ---------------------------------------------------------------------
def save_char_histogram_png(
    strings: List[str],
    output_dir: str = HP['figure_output_dir'],
    filename: str = HP['figure_filename']
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
def _build_vocab_dicts(ds: HTRDataset) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Return char→id / id→char dicts with index 0 reserved for the blank."""
    chars: List[str] = list(ds.character_classes)
    if " " not in chars:
        chars.append(" ")
    chars = sorted(set(chars))
    c2i = {c: i + 1 for i, c in enumerate(chars)}
    i2c = {i + 1: c for i, c in enumerate(chars)}
    return c2i, i2c
# ---------------------------------------------------------------------
# Wrapper around PyTorch's CTC loss that normalises by sequence length
# and handles infinite losses gracefully.
# ---------------------------------------------------------------------
def _ctc_loss_fn(logits: torch.Tensor,
                 targets: torch.IntTensor,
                 inp_lens: torch.IntTensor,
                 tgt_lens: torch.IntTensor) -> torch.Tensor:
    """Length-normalised CTC loss on raw logits."""
    return F.ctc_loss(F.log_softmax(logits, 2), targets, inp_lens, tgt_lens,
                      reduction="mean", zero_infinity=True)
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
                        eval_k: int | None = None) -> None:
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
    c2i, i2c = _build_vocab_dicts(dataset)
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
    unique_words = {transcrs[i].strip().lower() for i in subset_idx}
    print(
        f"[Refine] training on {len(subset_idx)} samples "
        f"containing {len(unique_words)} unique words"
    )
    train_loader = DataLoader(subset_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0,
                              pin_memory=(device.type == "cuda"))
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0; effective_batches = 0
        for imgs, trans, _ in train_loader:
            imgs = imgs.to(device)
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
            loss_m = _ctc_loss_fn(main_logits, targets, inp_lens, tgt_lens)
            loss_a = (
                _ctc_loss_fn(aux_logits, targets, inp_lens, tgt_lens)
                if aux_logits is not None
                else torch.tensor(0.0, device=main_logits.device)
            )
            loss = main_weight * loss_m + aux_weight * loss_a
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            epoch_loss += loss.item(); effective_batches += 1
        sched.step()
        avg_loss = epoch_loss / max(1, effective_batches)
        print(f"Epoch {epoch:03}/{num_epochs}  loss={avg_loss:.4f}  lr={sched.get_last_lr()[0]:.2e}")
        if (epoch + 1) % 20 == 0 or epoch == num_epochs:
            k_eval = eval_k if eval_k is not None else max_length
            cer = _evaluate_cer(backbone, test_loader, i2c, device, k=k_eval)
            print(f"[Eval] CER @ epoch {epoch}: {cer:.4f}")
    print("[Refine] finished.")
if __name__ == "__main__":
    proj_root = Path(__file__).resolve().parents[1]  # repository root
    gw_folder = proj_root / "htr_base" / "data" / HP['dataset_base_folder_name'] / "processed_words"
    if not gw_folder.exists():
        raise RuntimeError("GW processed dataset not found – generate it first!")
    class DummyCfg:
        def __init__(self, hp_config):
            self.k_external_words = hp_config['k_external_words']
            self.n_aligned = hp_config['n_aligned']
    train_set = HTRDataset(str(gw_folder), subset="train", fixed_size=HP['dataset_fixed_size'],
                            transforms=aug_transforms, config=DummyCfg(HP), concat_prob=0.8)
    c2i, _ = _build_vocab_dicts(train_set)
    arch_cfg_dict = HP['architecture_config']
    net = HTRNet(SimpleNamespace(**arch_cfg_dict), nclasses=len(c2i) + 1)
    net.to("cuda")
    refine_visual_model(
        train_set,
        net,
        num_epochs=HP['num_epochs'],
        batch_size=HP['batch_size'],
        lr=HP['learning_rate'],
        main_weight=HP['main_loss_weight'],
        aux_weight=HP['aux_loss_weight'],
        max_length=HP['max_length'],
        min_length=HP['min_length'],
        eval_k=HP['eval_k']
    )
