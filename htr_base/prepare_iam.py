#!/usr/bin/env python3
"""Utility to convert the raw IAM dataset into word-level splits."""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, Iterator, List, Tuple
from xml.etree import ElementTree as ET

from PIL import Image
from tqdm import tqdm

SplitMap = Dict[str, List[str]]
WordIterator = Iterator[Tuple[Image.Image, str]]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for IAM preprocessing.

    Purpose:
        Collect the user-provided configuration for preparing the IAM dataset.
    Args:
        None: All inputs are gathered from ``sys.argv``.
    Returns:
        argparse.Namespace: Namespace including ``iam_root``, ``output_root``,
            ``max_words`` and ``padding``.
    """
    parser = argparse.ArgumentParser(
        description="Prepare the IAM dataset into processed word-level splits."
    )
    parser.add_argument(
        "--iam-root",
        required=True,
        help="Path to the raw IAM dataset root (expects IAM_images/, xml/ and splits/).",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(Path(__file__).resolve().parent, "data", "IAM"),
        help="Destination directory for processed_words/ and splits/.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=None,
        help="Optional limit on the number of words to export per split (useful for smoke tests).",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=0,
        help="Additional pixels to pad around each word crop on every side.",
    )
    return parser.parse_args()


def load_split_definitions(split_dir: Path) -> SplitMap:
    """Load IAM split definitions from ``.uttlist`` files.

    Purpose:
        Read the official IAM train/val/test form lists and return them by split.
    Args:
        split_dir (Path): Directory containing ``train.uttlist``, ``validation.uttlist``
            and ``test.uttlist`` files provided with the IAM dataset.
    Returns:
        dict[str, list[str]]: Mapping of split names to ordered form identifiers.
    """
    mapping = {
        "train": "train.uttlist",
        "val": "validation.uttlist",
        "test": "test.uttlist",
    }
    splits: SplitMap = {}
    for name, filename in mapping.items():
        path = split_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing split file: {path}")
        with path.open("r", encoding="utf-8") as handle:
            splits[name] = [line.strip() for line in handle if line.strip()]
    return splits


def extract_word_images(
    xml_path: Path,
    page_image: Image.Image,
    padding: int = 0,
) -> WordIterator:
    """Yield cropped word images described in an IAM XML annotation.

    Purpose:
        Parse the per-form XML and extract bounding boxes for each word.
    Args:
        xml_path (Path): Path to the IAM XML file describing word components.
        page_image (Image.Image): Grayscale page image associated with the XML.
        padding (int): Additional pixels to add around each cropped word region.
    Returns:
        Iterator[tuple[Image.Image, str]]: Iterator of ``(crop, transcription)`` tuples.
    """
    if padding < 0:
        raise ValueError("padding must be non-negative")
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    width, height = page_image.size
    for word in root.findall(".//word"):
        text = word.attrib.get("text", "").strip()
        if not text:
            continue
        components = word.findall("cmp")
        if not components:
            continue
        xs: List[int] = []
        ys: List[int] = []
        xe: List[int] = []
        ye: List[int] = []
        for cmp_elem in components:
            x = int(cmp_elem.attrib.get("x", 0))
            y = int(cmp_elem.attrib.get("y", 0))
            w = int(cmp_elem.attrib.get("width", 0))
            h = int(cmp_elem.attrib.get("height", 0))
            xs.append(x)
            ys.append(y)
            xe.append(x + w)
            ye.append(y + h)
        if not xs or not ys:
            continue
        xmin = max(min(xs) - padding, 0)
        ymin = max(min(ys) - padding, 0)
        xmax = min(max(xe) + padding, width)
        ymax = min(max(ye) + padding, height)
        if xmax <= xmin or ymax <= ymin:
            continue
        crop = page_image.crop((xmin, ymin, xmax, ymax))
        yield crop, text


def copy_split_files(raw_split_dir: Path, output_split_dir: Path) -> None:
    """Copy official IAM split files into the output directory.

    Purpose:
        Ensure downstream consumers can reuse the original ``.uttlist`` files.
    Args:
        raw_split_dir (Path): Source directory containing the canonical splits.
        output_split_dir (Path): Destination directory where the files are written.
    Returns:
        None
    """
    output_split_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("train.uttlist", "validation.uttlist", "test.uttlist"):
        src = raw_split_dir / filename
        dst = output_split_dir / filename
        shutil.copyfile(src, dst)


def prepare_iam_dataset(
    iam_root: Path,
    output_root: Path,
    max_words: int | None = None,
    padding: int = 0,
) -> None:
    """Convert the raw IAM dataset into word-level splits.

    Purpose:
        Generate ``processed_words/{train,val,test}`` folders with PNG crops and ``gt.txt`` files.
    Args:
        iam_root (Path): Root of the raw IAM dataset.
        output_root (Path): Destination directory that will contain ``processed_words`` and ``splits``.
        max_words (int | None): Optional maximum number of words to extract per split.
        padding (int): Extra pixels to pad around each cropped word image.
    Returns:
        None
    """
    iam_root = iam_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    image_dir = iam_root / "IAM_images"
    xml_dir = iam_root / "xml"
    split_dir = iam_root / "splits"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing IAM image directory: {image_dir}")
    if not xml_dir.is_dir():
        raise FileNotFoundError(f"Missing IAM XML directory: {xml_dir}")
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing IAM split directory: {split_dir}")
    splits = load_split_definitions(split_dir)
    processed_root = output_root / "processed_words"
    processed_root.mkdir(parents=True, exist_ok=True)
    copy_split_files(split_dir, output_root / "splits")
    for split_name, form_ids in splits.items():
        split_dir_out = processed_root / split_name
        split_dir_out.mkdir(parents=True, exist_ok=True)
        words_written = 0
        gt_path = split_dir_out / "gt.txt"
        with gt_path.open("w", encoding="utf-8") as gt_file:
            for form_id in tqdm(form_ids, desc=f"Preparing {split_name}"):
                image_path = image_dir / f"{form_id}.png"
                xml_path = xml_dir / f"{form_id}.xml"
                if not image_path.is_file():
                    raise FileNotFoundError(f"Missing page image: {image_path}")
                if not xml_path.is_file():
                    raise FileNotFoundError(f"Missing annotation XML: {xml_path}")
                with Image.open(image_path) as page_image:
                    grayscale = page_image.convert("L")
                    for crop, transcription in extract_word_images(xml_path, grayscale, padding=padding):
                        sample_id = f"{split_name}_{words_written:06d}"
                        crop.save(split_dir_out / f"{sample_id}.png")
                        gt_file.write(f"{sample_id} {transcription}\n")
                        words_written += 1
                        if max_words is not None and words_written >= max_words:
                            break
                if max_words is not None and words_written >= max_words:
                    break
        if words_written == 0:
            raise RuntimeError(f"No words were written for split '{split_name}'.")


def main() -> None:
    """Entry-point that wires CLI arguments to dataset preparation.

    Purpose:
        Allow the script to be invoked from the command line.
    Args:
        None
    Returns:
        None
    """
    args = parse_arguments()
    prepare_iam_dataset(
        iam_root=Path(args.iam_root),
        output_root=Path(args.output_root),
        max_words=args.max_words,
        padding=args.padding,
    )


if __name__ == "__main__":
    main()
