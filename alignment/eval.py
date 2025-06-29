"""Evaluation helpers for alignment module."""

import torch
from torch.utils.data import DataLoader, Dataset

from htr_base.utils.metrics import CER
from htr_base.utils.vocab import load_vocab
from .ctc_utils import greedy_ctc_decode, beam_search_ctc_decode


def _assert_finite(t: torch.Tensor, where: str) -> None:
    assert torch.isfinite(t).all(), f"Non-finite values in {where}"


def compute_cer(
    dataset: Dataset,
    model: torch.nn.Module,
    *,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
    decode: str = "greedy",
    beam_width: int = 10,
    k: int | None = None,
) -> float:
    """Return character error rate on *dataset* using *model*.

    Parameters
    ----------
    dataset : Dataset
        Items yield ``(img, transcription, _)`` triples as in ``HTRDataset``.
    model : torch.nn.Module
        Network returning CTC logits (``(T,B,C)``).
    batch_size : int, optional
        Mini-batch size used during evaluation.
    device : str or torch.device, optional
        Compute device for the forward pass.
    decode : {'greedy', 'beam'}, optional
        Decoding strategy for CTC outputs.
    beam_width : int, optional
        Beam width when ``decode='beam'``.
    k : int, optional
        If given, also report CER for samples of length ``<= k`` and ``> k``.

    Returns
    -------
    float
        Overall CER over the dataset.
    """
    device = torch.device(device)
    model = model.to(device).eval()

    orig_tf = getattr(dataset, "transforms", None)
    if hasattr(dataset, "transforms"):
        dataset.transforms = None

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    c2i, i2c = load_vocab()

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
        msg += (
            f"  <={k}: {le.score():.4f} (n={n_le})  >{k}: {gt.score():.4f} (n={n_gt})"
        )
    print(msg)
    return total.score()
