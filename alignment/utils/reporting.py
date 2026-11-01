"""Reporting helpers for pseudo-labeling rounds."""

from __future__ import annotations

from pathlib import Path

import torch

from htr_base.utils.htr_dataset import HTRDataset


def log_pseudo_labels(
    new_indices: torch.Tensor,
    dataset: HTRDataset,
    round_idx: int,
    out_dir: str = "results",
) -> None:
    """Save pseudo-labeled examples for the current refinement round.

    The output file has three tab-separated columns:
    1) Dataset index of the sample (integer)
    2) Predicted transcription (pseudo-label)
    3) Ground-truth transcription (original label)

    Args:
        new_indices (torch.Tensor): Newly pseudo-labeled dataset indices with
            shape (K,).
        dataset (HTRDataset): Dataset exposing unique_words, transcriptions, and
            aligned ids with shape (N,).
        round_idx (int): Refinement cycle index used in output filename.
        out_dir (str): Directory where pseudo-label logs are saved.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    for idx in new_indices.tolist():
        word_id = int(dataset.aligned[idx])
        if word_id == -1:
            continue
        pred_word = dataset.unique_words[word_id]
        true_word = dataset.transcriptions[idx]
        rows.append(f"{idx}\t{pred_word}\t{true_word}")

    file_path = out_path / f"pseudo_labels_round_{round_idx}.txt"
    with open(file_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(rows))


def log_round_metrics(
    round_idx: int,
    correct_pseudo: int,
    cer_test: float,
    wer_test: float,
    out_file: str,
    *,
    basefolder: str,
    n_aligned: int,
    seed: int,
    rules_disagree: int,
    correct_argmax: int,
) -> None:
    """Append one round of metrics to a TSV file.

    Behavior:
      - If out_file is a concrete path (anything other than auto,
        case-insensitive), save exactly there.
      - If out_file == auto (case-insensitive), build:
        results/<DATASET>/<dataset_lower>_<n_aligned>samples.txt
        where <DATASET> is inferred from basefolder.

    On the first round (round_idx == 1), an existing file at the final path is
    deleted. Subsequent rounds append one TSV row:
        <round_idx>\t<correct_pseudo>\t<cer_test>\t<wer_test>\t<seed>\t<rules_disagree>\t<correct_argmax>\n

    Args:
        round_idx (int): One-based round index.
        correct_pseudo (int): Number of correct pseudo-labels this round.
        cer_test (float): Character error rate on test split.
        wer_test (float): Word error rate on test split.
        out_file (str): Explicit output path or "auto" for generated path.
        basefolder (str): Dataset base folder used for dataset-name inference.
        n_aligned (int): Initial number of aligned samples used in auto filename.
        seed (int): Seed that picked the initially aligned samples.
        rules_disagree (int): New pseudo-labels where the argmax rule picks a
            different word than the barycentre rule.
        correct_argmax (int): New pseudo-labels the argmax rule gets right.
    """
    if isinstance(out_file, str) and out_file.strip().lower() == "auto":
        base = str(basefolder)
        tokens = [p.lower() for p in Path(base).parts]
        ds_name = None
        for candidate in ("GW", "IAM", "CVL"):
            if candidate.lower() in tokens or candidate.lower() in base.lower():
                ds_name = candidate
                break
        if ds_name is None:
            ds_name = (Path(base).parts[-2] if len(Path(base).parts) >= 2 else "DATASET").upper()

        auto_dir = Path("results") / ds_name
        auto_name = f"{ds_name.lower()}_{int(n_aligned)}samples.txt"
        output_path = auto_dir / auto_name
    else:
        output_path = Path(out_file)

    if round_idx == 1 and output_path.is_file():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if output_path.is_file() else "w"

    with open(output_path, mode, encoding="utf-8") as fh:
        fh.write(
            f"{round_idx}\t{correct_pseudo}\t{cer_test:.4f}\t{wer_test:.4f}\t{seed}"
            f"\t{rules_disagree}\t{correct_argmax}\n"
        )
