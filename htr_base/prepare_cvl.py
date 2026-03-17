#!/usr/bin/env python3
"""Utility to convert the raw CVL dataset into word-level splits."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image

from tqdm import tqdm

SplitMap = Dict[str, str]
WordRecord = Tuple[Path, str]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for CVL preprocessing.

    Purpose:
        Collect CLI inputs describing source and destination paths.
    Returns:
        argparse.Namespace: Namespace containing cvl_root, output_root
            and max_words.
    """
    parser = argparse.ArgumentParser(
        description="Prepare the CVL dataset into processed word-level splits."
    )
    parser.add_argument(
        "--cvl-root",
        required=True,
        help="Path to the raw CVL dataset root (expects trainset/ and testset/).",
    )
    parser.add_argument(
        "--output-root",
        default=Path(__file__).resolve().parent / "data" / "CVL",
        help="Destination directory for processed_words/{train,test}.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=None,
        help="Optional limit on the number of words copied per split (useful for smoke tests).",
    )
    return parser.parse_args()


def transcription_from_filename(path: Path) -> str:
    """Extract the transcription encoded in a CVL filename.

    Purpose:
        Retrieve the human-readable word from the image filename tokens.
    Args:
        path (Path): File whose stem contains metadata and the transcription.
    Returns:
        str: Text that follows the position identifiers inside the filename.
    """
    parts = path.stem.split("-")
    if len(parts) < 5:
        raise ValueError(f"Unexpected CVL filename format: {path.name}")
    return normalize_cvl_transcription("-".join(parts[4:]))


def normalize_cvl_transcription(text: str) -> str:
    """Normalize a CVL transcription for this repository's vocabulary.

    Purpose:
        Remove unsupported hyphen characters while preserving CVL casing
        and supported symbols such as apostrophes and umlauts.
    Args:
        text (str): Raw transcription extracted from the filename.
    Returns:
        str: Normalized transcription with all "-" removed and whitespace trimmed.

    Raises:
        ValueError: If normalization produces an empty transcription.
    """
    normalized = text.replace("-", "").strip()
    if not normalized:
        raise ValueError("Empty transcription after normalization.")
    return normalized


def collect_word_records(words_root: Path) -> Tuple[List[WordRecord], Dict[str, int]]:
    """Collect ordered CVL word records from a split directory.

    Purpose:
        Traverse writer folders and gather paths alongside their transcriptions.
    Args:
        words_root (Path): Directory containing writer-id subfolders with word images.
    Returns:
        tuple[list[tuple[Path, str]], dict[str, int]]: Sorted list of word image
            paths and normalized text labels, plus split-level collection stats.
    """
    if not words_root.is_dir():
        raise FileNotFoundError(f"Missing CVL word directory: {words_root}")
    records: List[WordRecord] = []
    stats = {
        "files_found": 0,
        "skipped_bad_filename": 0,
        "skipped_empty_after_normalization": 0,
    }
    for writer_dir in sorted(p for p in words_root.iterdir() if p.is_dir()):
        for image_path in sorted(writer_dir.glob("*.tif")):
            stats["files_found"] += 1
            try:
                transcription = transcription_from_filename(image_path)
            except ValueError as error:
                err_msg = str(error)
                if "Unexpected CVL filename format" in err_msg:
                    stats["skipped_bad_filename"] += 1
                    continue
                if "Empty transcription after normalization" in err_msg:
                    stats["skipped_empty_after_normalization"] += 1
                    continue
                raise
            records.append((image_path, transcription))
    if not records:
        raise RuntimeError(f"No word images found under {words_root}")
    return records, stats


def write_split(records: Iterable[WordRecord], destination: Path, label: str) -> Tuple[int, int]:
    """Write CVL word records and the accompanying ground-truth file.

    Purpose:
        Copy images into the processed split folder and emit gt.txt.
    Args:
        records (Iterable[WordRecord]): Sequence of (path, transcription) pairs.
        destination (Path): Output directory for the processed split.
        label (str): Progress label displayed during copy.
    Returns:
        tuple[int, int]: Number of records written and count skipped due to PIL
            read/convert/save errors.
    """
    destination.mkdir(parents=True, exist_ok=True)
    gt_path = destination / "gt.txt"
    written = 0
    skipped_pil_errors = 0
    with gt_path.open("w", encoding="utf-8") as handle:
        for src, transcription in tqdm(records, desc=label, unit="word"):
            dst_name = f"{src.stem}.png"
            dst_path = destination / dst_name
            try:
                with Image.open(src) as tif_image:
                    tif_image.convert("L").save(dst_path, format="PNG")
            except Exception:
                skipped_pil_errors += 1
                continue
            handle.write(f"{dst_name} {transcription}\n")
            written += 1
    return written, skipped_pil_errors


def prepare_cvl_dataset(
    cvl_root: Path,
    output_root: Path,
    max_words: int | None = None,
) -> None:
    """Convert the raw CVL dataset into processed word-level splits.

    Purpose:
        Generate processed_words/train and processed_words/test directories with gt.txt files.
    Args:
        cvl_root (Path): Root directory containing trainset and testset folders.
        output_root (Path): Destination that will hold the processed data layout.
        max_words (int | None): Optional per-split limit for copied examples.
    """
    cvl_root = cvl_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    split_map: SplitMap = {"train": "trainset", "test": "testset"}
    processed_root = output_root / "processed_words"
    processed_root.mkdir(parents=True, exist_ok=True)
    for split_name, raw_folder in split_map.items():
        words_root = cvl_root / raw_folder / "words"
        records, collect_stats = collect_word_records(words_root)
        if max_words is not None:
            if max_words < 0:
                raise ValueError("max_words must be non-negative")
            records = records[:max_words]
        if not records:
            raise RuntimeError(f"No records available for split '{split_name}'")
        destination = processed_root / split_name
        written, skipped_pil = write_split(records, destination, label=f"Writing {split_name}")
        skipped_total = (
            collect_stats["skipped_bad_filename"]
            + collect_stats["skipped_empty_after_normalization"]
            + skipped_pil
        )
        print(
            f"[{split_name}] files_found={collect_stats['files_found']} "
            f"written={written} skipped_total={skipped_total} "
            f"(bad_filename={collect_stats['skipped_bad_filename']} "
            f"empty_after_normalization={collect_stats['skipped_empty_after_normalization']} "
            f"pil_errors={skipped_pil})"
        )


def main() -> None:
    """Entry-point to trigger CVL dataset preparation from the CLI.

    Purpose:
        Parse CLI arguments and launch the preprocessing routine.
    """
    args = parse_arguments()
    prepare_cvl_dataset(
        cvl_root=Path(args.cvl_root),
        output_root=Path(args.output_root),
        max_words=args.max_words,
    )


if __name__ == "__main__":
    main()
