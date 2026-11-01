#!/usr/bin/env python3
"""Build GW word-level train/val/test folders from WashingtonDB v1.0.

Usage:
python htr_base/prepare_gw.py \
    --dataset_folder /path/to/washingtondb-v1.0 \
    --output_path htr_base/data/GW/processed_words

Each split folder gets a copy of its word images and a `gt.txt` with one
`<word_id> <transcription>` line per word. Transcriptions keep the original
case and punctuation, matching the GW recognition protocol.
"""

import argparse
import re
import shutil
from collections import Counter
from pathlib import Path

from tqdm import tqdm

# Official cv1 split by page id: 2433/1293/1168 words.
SPLIT_PAGES = {"train": (270, 279), "val": (300, 304), "test": (305, 309)}

# WashingtonDB special tokens and the characters they stand for. Plain letters
# are written as single tokens with their original case; digits and ordinals
# are written as ``s_<n>`` / ``s_<n>st`` and kept as written.
SPECIAL_TOKENS = {
    "s_pt": ".",
    "s_cm": ",",
    "s_mi": "-",
    "s_sq": ";",
    "s_qo": ":",
    "s_qt": "'",
    "s_bl": "(",
    "s_br": ")",
    "s_sl": "/",
    "s_lb": "£",
    "s_et": "&",
    "s_s": "s",  # long s, written as a plain s
    "s_GW": "G.W.",
}


def split_of(word_id):
    """Return the split a GW word belongs to, from the page id in its name.

    Args:
    - word_id (str): WashingtonDB word id such as `270-01-01`.

    Returns:
    - (str): `train`, `val` or `test`.

    Raises:
    - ValueError: If the page id is outside every split.
    """

    page_id = int(word_id.split("-")[0])
    for split, (first, last) in SPLIT_PAGES.items():
        if first <= page_id <= last:
            return split
    raise ValueError(f"Invalid page id: {page_id}")


def decode_word(tokens, unknown):
    """Decode a WashingtonDB hyphen-tokenized word label into text.

    Args:
    - tokens (str): Tokenized label, such as `s_2-s_7-s_0-s_pt` ("270.") or
      `L-e-t-t-e-r-s-s_cm` ("Letters,").
    - unknown (Counter): Counts tokens that no rule decodes; they are dropped.

    Returns:
    - (str): Transcription with the original case and punctuation.
    """

    parts = []
    for token in tokens.split("-"):
        if not token:
            continue
        if len(token) == 1 and token.isalpha():
            parts.append(token)
        elif token in SPECIAL_TOKENS:
            parts.append(SPECIAL_TOKENS[token])
        elif token.startswith("s_") and re.fullmatch(r"[0-9]+(?:st|nd|rd|th)?", token[2:]):
            parts.append(token[2:])
        else:
            unknown[token] += 1
    return "".join(parts).strip()


def read_word_labels(root, unknown):
    """Read and decode every word label, grouped by split in file order.

    Args:
    - root (Path): WashingtonDB root containing `ground_truth/word_labels.txt`.
    - unknown (Counter): Passed to `decode_word` to count undecoded tokens.

    Returns:
    - (dict[str, list[tuple[str, str]]]): `(word_id, text)` pairs per split.
    """

    labels_file = root / "ground_truth" / "word_labels.txt"
    if not labels_file.is_file():
        raise FileNotFoundError(f"Word labels file not found: {labels_file}")

    by_split = {split: [] for split in SPLIT_PAGES}
    with open(labels_file, encoding="utf-8-sig") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            word_id, tokens = parts
            by_split[split_of(word_id)].append((word_id, decode_word(tokens, unknown)))
    return by_split


def write_split(image_dir, items, out_dir):
    """Copy one split's word images and write its `gt.txt`.

    Words with an empty transcription are skipped, since CTC cannot encode them.

    Args:
    - image_dir (Path): Folder with the `<word_id>.png` word images.
    - items (list[tuple[str, str]]): `(word_id, text)` pairs for the split.
    - out_dir (Path): Output folder for this split.

    Returns:
    - (tuple[int, int]): Number of words written and number skipped.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    gt_lines = []
    for word_id, text in tqdm(items, desc=f"Writing {out_dir.name}"):
        if not text:
            continue
        src = image_dir / f"{word_id}.png"
        if not src.is_file():
            raise FileNotFoundError(f"Expected word image missing: {src}")
        shutil.copy2(src, out_dir / src.name)
        gt_lines.append(f"{word_id} {text}\n")
    with open(out_dir / "gt.txt", "w", encoding="utf-8") as f:
        f.writelines(gt_lines)
    return len(gt_lines), len(items) - len(gt_lines)


def parse_arguments():
    """Parse the command-line arguments.

    Returns:
    - (argparse.Namespace): `dataset_folder` and `output_path`.
    """

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--dataset_folder",
        required=True,
        help="WashingtonDB root (contains data/ and ground_truth/).",
    )
    p.add_argument("--output_path", required=True, help="Where to write {train,val,test}/.")
    return p.parse_args()


def main():
    """Build the GW train/val/test folders and report undecoded tokens."""

    args = parse_arguments()
    root = Path(args.dataset_folder)
    image_dir = root / "data" / "word_images_normalized"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Word image folder not found: {image_dir}")

    unknown = Counter()
    for split, items in read_word_labels(root, unknown).items():
        written, skipped = write_split(image_dir, items, Path(args.output_path) / split)
        print(f"[{split}] wrote {written} samples (skipped {skipped} empty labels)")

    if unknown:
        print("Unknown decode tokens (top 15):")
        for token, count in unknown.most_common(15):
            print(f"  {token}: {count}")
    else:
        print("Unknown decode tokens: none")


if __name__ == "__main__":
    main()
