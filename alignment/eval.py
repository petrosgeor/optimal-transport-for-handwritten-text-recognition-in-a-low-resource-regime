"""Evaluation helpers for alignment module."""

import torch
from torch.utils.data import DataLoader

from htr_base.utils.metrics import CER
from htr_base.utils.vocab import load_vocab
from .ctc_utils import greedy_ctc_decode, beam_search_ctc_decode
from htr_base.utils.htr_dataset import HTRDataset


def _assert_finite(t: torch.Tensor, where: str) -> None:
    """Raise an error if t contains NaN or Inf values.

    Args:
        t (torch.Tensor): Tensor to check for numeric stability, with any shape.
        where (str): Description of the tensors origin used in the message.
    """
    assert torch.isfinite(t).all(), f"Non-finite values in {where}"


def compute_cer(
    dataset: HTRDataset,
    model: torch.nn.Module,
    *,
    batch_size: int = 64,
    device: str = "cpu",
    decode: str = "greedy",
    beam_width: int = 10,
    k: int = None,
) -> float:
    """Return character error rate on *dataset* using *model*.

    Args:
        dataset (HTRDataset): Dataset yielding `(img, transcription, aligned)`
            where each `img` has shape (1, H, W).
        model (torch.nn.Module): CTC model returning logits with shape
            (T, B, C), or a tuple whose first element has shape (T, B, C).
        batch_size (int): Evaluation mini-batch size B.
        device (str): Device identifier used for forward passes.
        decode (str): Decoder type, either "greedy" or "beam".
        beam_width (int): Beam width used when decode is "beam".
        k (int | None): Optional split threshold for reporting by word length.

    Returns:
        float: Dataset-level CER value.
    """
    device = torch.device(device)
    model = model.to(device).eval()

    orig_tf = getattr(dataset, "transforms", None)
    if hasattr(dataset, "transforms"):
        dataset.transforms = None

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    # Support torch.utils.data.Subset by unwrapping base dataset for vocab
    base_ds = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
    c2i, i2c = load_vocab(base_ds.get_dataset_name())

    total = CER()
    le = CER(); gt = CER()
    n_le = 0; n_gt = 0
    with torch.no_grad():
        for imgs, txts, _ in loader:
            imgs = imgs.to(device)
            logits = model(imgs, return_feats=False)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            _assert_finite(logits, "logits")
            assert logits.shape[-1] == len(c2i) + 1, "CTC class dimension mismatch"
            if decode == "beam":
                preds = beam_search_ctc_decode(logits, i2c, beam_width=beam_width)
            else:
                preds = greedy_ctc_decode(logits, i2c)
            for p, t in zip(preds, txts):
                gt_txt = t.strip(); pr_txt = p.strip()
                total.update(pr_txt, gt_txt)
                if k is not None:
                    L = len(gt_txt.replace(" ", ""))
                    if L <= k:
                        le.update(pr_txt, gt_txt); n_le += 1
                    else:
                        gt.update(pr_txt, gt_txt); n_gt += 1

    if hasattr(dataset, "transforms"):
        dataset.transforms = orig_tf
    model.train()

    msg = f"[Eval] CER: {total.score():.4f}"
    if k is not None:
        if n_le > 0:
            msg += f"  <={k}: {le.score():.4f} (n={n_le})"
        if n_gt > 0:
            msg += f"  >{k}: {gt.score():.4f} (n={n_gt})"
    print(msg)
    return total.score()



def compute_wer(
    dataset: HTRDataset,
    model: torch.nn.Module,
    *,
    batch_size: int = 64,
    device: str = "cpu",
    decode: str = "greedy",
) -> int:
    """Return integer word error rate (%) on *dataset* using *model*.

    Args:
        dataset (HTRDataset): Dataset yielding `(img, transcription, aligned)`
            where each `img` has shape (1, H, W).
        model (torch.nn.Module): CTC model returning logits with shape
            (T, B, C), or a tuple whose first element has shape (T, B, C).
        batch_size (int): Evaluation mini-batch size B.
        device (str): Device identifier used for forward passes.
        decode (str): Decoder type, either "greedy" or "beam".

    Returns:
        int: Dataset-level WER percentage.
    """
    import editdistance  # local import to avoid touching module-level deps

    device = torch.device(device)
    model = model.to(device).eval()

    # Temporarily disable dataset augmentations for stable decoding (same as compute_cer)
    orig_tf = getattr(dataset, "transforms", None)
    if hasattr(dataset, "transforms"):
        dataset.transforms = None

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    # Support torch.utils.data.Subset by unwrapping base dataset for vocab
    base_ds = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
    c2i, i2c = load_vocab(base_ds.get_dataset_name())

    total_dist = 0.0
    total_len = 0
    BEAM_WIDTH = 10  # fixed width since signature doesn't expose it

    with torch.no_grad():
        for imgs, txts, _ in loader:
            imgs = imgs.to(device)
            logits = model(imgs, return_feats=False)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]

            # numeric sanity + class-dim check, consistent with compute_cer
            assert torch.isfinite(logits).all(), "Non-finite values in logits"
            assert logits.shape[-1] == len(c2i) + 1, "CTC class dimension mismatch"

            if decode == "beam":
                preds = beam_search_ctc_decode(logits, i2c, beam_width=BEAM_WIDTH)
            else:
                preds = greedy_ctc_decode(logits, i2c)

            for p, t in zip(preds, txts):
                gt_words = [w for w in t.strip().split() if w]
                pr_words = [w for w in p.strip().split() if w]
                dist = float(editdistance.eval(pr_words, gt_words))
                total_dist += dist
                total_len += len(gt_words)

    # restore dataset state and model mode
    if hasattr(dataset, "transforms"):
        dataset.transforms = orig_tf
    model.train()

    wer_pct = 0 if total_len == 0 else (100.0 * total_dist / total_len)
    print(f"[Eval] WER: {wer_pct}%")
    return wer_pct
