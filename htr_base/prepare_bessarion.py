#!/usr/bin/env python3
"""Prepare the Bessarion dataset into deterministic word-level splits.

This script scans PAGE XML annotations under a Bessarion root, extracts word-level
bounding boxes and transcriptions, crops word images, and writes a processed layout
compatible with the repository's `HTRDataset` loader.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterator, List, Tuple
from xml.etree import ElementTree as ET

from PIL import Image
from tqdm import tqdm

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
NS = {"pg": PAGE_NS}
IMAGE_EXT_CANDIDATES = [".jpg", ".JPG", ".jpeg", ".JPEG", ".tiff", ".tif", ".png", ".PNG"]
SPLITS = ("train", "val", "test")

WordEntry = Tuple[str, str, str]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for Bessarion preprocessing.

    Args:
    - None: Inputs are collected from ``sys.argv``.

    Returns:
    - (argparse.Namespace): Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Prepare the Bessarion dataset into word-level processed splits."
    )
    parser.add_argument(
        "--bessarion-root",
        required=True,
        help="Path to DATASETS/Bessarion-master.",
    )
    parser.add_argument(
        "--output-root",
        default=Path(__file__).resolve().parent / "data" / "BESSARION",
        help="Destination directory that will contain processed_words/{train,val,test}.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional limit on how many PAGE XML files to process (for smoke runs).",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=None,
        help="Optional global limit on saved word crops across all splits.",
    )
    parser.add_argument(
        "--skip-backup-xml",
        dest="skip_backup_xml",
        action="store_true",
        default=True,
        help="Skip XML files that include '~' in the filename (default: true).",
    )
    parser.add_argument(
        "--include-backup-xml",
        dest="skip_backup_xml",
        action="store_false",
        help="Include XML files that include '~' in the filename.",
    )
    return parser.parse_args()


def decode_xml_bytes(xml_bytes: bytes, xml_path: Path) -> str:
    """Decode XML bytes using a deterministic fallback chain.

    Args:
    - xml_bytes (bytes): Raw bytes read from an XML file.
    - xml_path (Path): Path used for descriptive error reporting.

    Returns:
    - (str): Decoded XML text.

    Raises:
    - UnicodeDecodeError: If all candidate decoders fail.
    """
    decoders = ("utf-8", "cp1253", "iso-8859-7")
    last_error: UnicodeDecodeError | None = None
    for encoding in decoders:
        try:
            return xml_bytes.decode(encoding)
        except UnicodeDecodeError as error:
            last_error = error
    raise UnicodeDecodeError(
        last_error.encoding if last_error is not None else "unknown",
        xml_bytes,
        last_error.start if last_error is not None else 0,
        last_error.end if last_error is not None else 1,
        f"Could not decode XML file: {xml_path}",
    )


def parse_page_xml(xml_path: Path) -> ET.Element:
    """Parse a PAGE XML file into an ElementTree root node.

    Args:
    - xml_path (Path): XML file to parse.

    Returns:
    - (ET.Element): Root element for the parsed XML.
    """
    xml_text = decode_xml_bytes(xml_path.read_bytes(), xml_path)
    return ET.fromstring(xml_text)


def normalize_transcription(text: str) -> str:
    """Normalize a transcription while preserving Greek characters.

    Args:
    - text (str): Raw text from ``Word/TextEquiv/Unicode``.

    Returns:
    - (str): Canonical text with NFC normalization and collapsed whitespace.
    """
    cleaned = text.replace("\ufeff", "")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = unicodedata.normalize("NFC", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def parse_points(points: str) -> List[Tuple[int, int]]:
    """Parse PAGE polygon points formatted as ``x,y x,y ...``.

    Args:
    - points (str): PAGE coordinates string.

    Returns:
    - (list[tuple[int, int]]): Parsed integer coordinates.
    """
    parsed: List[Tuple[int, int]] = []
    for token in points.strip().split():
        if "," not in token:
            continue
        x_str, y_str = token.split(",", 1)
        try:
            x = int(round(float(x_str)))
            y = int(round(float(y_str)))
        except ValueError:
            continue
        parsed.append((x, y))
    return parsed


def has_non_empty_word_unicode(root: ET.Element) -> bool:
    """Return whether a PAGE XML file contains at least one labeled word.

    Args:
    - root (ET.Element): Parsed PAGE XML root.

    Returns:
    - (bool): ``True`` when at least one ``Word`` has non-empty ``Unicode`` text.
    """
    for word in root.findall(".//pg:Word", NS):
        unicode_elem = word.find(".//pg:Unicode", NS)
        if unicode_elem is None:
            continue
        if normalize_transcription(unicode_elem.text or ""):
            return True
    return False


def find_canonical_xml_files(
    image_root: Path,
    skip_backup_xml: bool,
) -> Tuple[List[Path], Dict[str, int]]:
    """Find and filter PAGE XML files that contain word-level text labels.

    Args:
    - image_root (Path): Root directory containing images and PAGE XML files.
    - skip_backup_xml (bool): Whether filenames containing ``~`` should be excluded.

    Returns:
    - (tuple[list[Path], dict[str, int]]): Canonical XML paths and discovery stats.
    """
    stats = {
        "xml_scanned": 0,
        "xml_skipped_backup": 0,
        "xml_parse_failed": 0,
        "xml_with_text": 0,
        "xml_without_text": 0,
    }

    canonical: List[Path] = []
    for xml_path in sorted(image_root.rglob("*.xml")):
        stats["xml_scanned"] += 1
        if skip_backup_xml and "~" in xml_path.name:
            stats["xml_skipped_backup"] += 1
            continue
        try:
            root = parse_page_xml(xml_path)
        except Exception:
            stats["xml_parse_failed"] += 1
            continue
        if has_non_empty_word_unicode(root):
            canonical.append(xml_path)
            stats["xml_with_text"] += 1
        else:
            stats["xml_without_text"] += 1

    return canonical, stats


def resolve_image_path(xml_path: Path, page_elem: ET.Element | None) -> Path | None:
    """Resolve the source image path for an XML page.

    Args:
    - xml_path (Path): PAGE XML file path.
    - page_elem (ET.Element | None): ``Page`` element from the XML, if present.

    Returns:
    - (Path | None): Existing image path, or ``None`` when unresolved.
    """
    xml_dir = xml_path.parent

    if page_elem is not None:
        image_filename = (page_elem.attrib.get("imageFilename") or "").strip()
        if image_filename:
            candidate = Path(image_filename)
            if not candidate.is_absolute():
                candidate = xml_dir / candidate
            if candidate.is_file():
                return candidate

    for ext in IMAGE_EXT_CANDIDATES:
        candidate = xml_dir / f"{xml_path.stem}{ext}"
        if candidate.is_file():
            return candidate

    return None


def iter_word_entries(root: ET.Element) -> Iterator[WordEntry]:
    """Yield word ids, coordinates and normalized transcriptions from PAGE XML.

    Args:
    - root (ET.Element): Parsed PAGE XML root.

    Returns:
    - (Iterator[tuple[str, str, str]]): Iterator yielding ``(word_id, points, text)``.
    """
    for idx, word_elem in enumerate(root.findall(".//pg:Word", NS)):
        unicode_elem = word_elem.find(".//pg:Unicode", NS)
        coords_elem = word_elem.find("./pg:Coords", NS)
        if unicode_elem is None or coords_elem is None:
            continue
        text = normalize_transcription(unicode_elem.text or "")
        points = coords_elem.attrib.get("points", "")
        word_id = word_elem.attrib.get("id", f"word_{idx:05d}")
        yield word_id, points, text


def assign_split(page_key: str) -> str:
    """Assign train/val/test split deterministically from a page key.

    Args:
    - page_key (str): Relative page identifier used as split seed.

    Returns:
    - (str): Split label in ``{"train", "val", "test"}``.
    """
    bucket = int(hashlib.sha1(page_key.encode("utf-8")).hexdigest(), 16) % 100
    if bucket <= 79:
        return "train"
    if bucket <= 89:
        return "val"
    return "test"


def prepare_bessarion_dataset(
    bessarion_root: Path,
    output_root: Path,
    max_pages: int | None = None,
    max_words: int | None = None,
    skip_backup_xml: bool = True,
) -> None:
    """Convert raw Bessarion PAGE annotations into word-level split folders.

    Args:
    - bessarion_root (Path): Path to ``DATASETS/Bessarion-master``.
    - output_root (Path): Output path for the ``processed_words`` structure.
    - max_pages (int | None): Optional limit on processed PAGE XML pages.
    - max_words (int | None): Optional global limit on saved word samples.
    - skip_backup_xml (bool): Whether to skip XML files containing ``~``.
    """
    if max_pages is not None and max_pages < 0:
        raise ValueError("max_pages must be non-negative")
    if max_words is not None and max_words < 0:
        raise ValueError("max_words must be non-negative")

    bessarion_root = bessarion_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    image_root = bessarion_root / "dataset" / "image_resources_raw"
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing image_resources_raw directory: {image_root}")

    processed_root = output_root / "processed_words"
    split_dirs = {split: processed_root / split for split in SPLITS}
    for path in split_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    gt_handles = {
        split: (split_dirs[split] / "gt.txt").open("w", encoding="utf-8") for split in SPLITS
    }
    manifest_path = processed_root / "manifest.tsv"
    manifest_handle = manifest_path.open("w", encoding="utf-8", newline="")
    manifest_writer = csv.writer(manifest_handle, delimiter="\t")
    manifest_writer.writerow(
        [
            "sample_id",
            "split",
            "crop_path",
            "source_image",
            "source_xml",
            "word_id",
            "transcription",
        ]
    )

    xml_paths, xml_stats = find_canonical_xml_files(image_root, skip_backup_xml=skip_backup_xml)
    if max_pages is not None:
        xml_paths = xml_paths[:max_pages]

    counters = {
        "xml_missing_image": 0,
        "words_extracted": 0,
        "words_skipped_empty_text": 0,
        "words_skipped_bad_coords": 0,
        "words_skipped_tiny_crop": 0,
    }
    split_counts = {split: 0 for split in SPLITS}

    try:
        for xml_path in tqdm(xml_paths, desc="Processing PAGE XML", unit="xml"):
            root = parse_page_xml(xml_path)
            page_elem = root.find(".//pg:Page", NS)
            image_path = resolve_image_path(xml_path, page_elem)
            if image_path is None:
                counters["xml_missing_image"] += 1
                continue

            page_key = xml_path.relative_to(image_root).as_posix()
            split = assign_split(page_key)

            with Image.open(image_path) as image:
                grayscale = image.convert("L")
                width, height = grayscale.size

                for word_id, points_str, text in iter_word_entries(root):
                    if not text:
                        counters["words_skipped_empty_text"] += 1
                        continue
                    points = parse_points(points_str)
                    if not points:
                        counters["words_skipped_bad_coords"] += 1
                        continue

                    x_values = [point[0] for point in points]
                    y_values = [point[1] for point in points]
                    xmin = max(min(x_values), 0)
                    ymin = max(min(y_values), 0)
                    xmax = min(max(x_values), width)
                    ymax = min(max(y_values), height)
                    if xmax <= xmin or ymax <= ymin:
                        counters["words_skipped_tiny_crop"] += 1
                        continue
                    if (xmax - xmin) < 2 or (ymax - ymin) < 2:
                        counters["words_skipped_tiny_crop"] += 1
                        continue

                    sample_idx = split_counts[split]
                    sample_id = f"bessarion_{split}_{sample_idx:07d}"
                    crop_filename = f"{sample_id}.png"
                    crop_path = split_dirs[split] / crop_filename

                    crop = grayscale.crop((xmin, ymin, xmax, ymax))
                    crop.save(crop_path, format="PNG")

                    gt_handles[split].write(f"{sample_id} {text}\n")
                    manifest_writer.writerow(
                        [
                            sample_id,
                            split,
                            str(crop_path.relative_to(output_root)),
                            str(image_path.relative_to(bessarion_root)),
                            str(xml_path.relative_to(bessarion_root)),
                            word_id,
                            text,
                        ]
                    )

                    split_counts[split] += 1
                    counters["words_extracted"] += 1
                    if max_words is not None and counters["words_extracted"] >= max_words:
                        break

            if max_words is not None and counters["words_extracted"] >= max_words:
                break
    finally:
        for handle in gt_handles.values():
            handle.close()
        manifest_handle.close()

    print(f"Output root: {output_root}")
    print(f"Processed root: {processed_root}")
    print(f"XML scanned: {xml_stats['xml_scanned']}")
    print(f"XML skipped as backup: {xml_stats['xml_skipped_backup']}")
    print(f"XML parse failed: {xml_stats['xml_parse_failed']}")
    print(f"XML with text: {xml_stats['xml_with_text']}")
    print(f"XML without text: {xml_stats['xml_without_text']}")
    print(f"XML missing image: {counters['xml_missing_image']}")
    print(f"Words extracted: {counters['words_extracted']}")
    print(f"Words skipped (empty text): {counters['words_skipped_empty_text']}")
    print(f"Words skipped (bad coords): {counters['words_skipped_bad_coords']}")
    print(f"Words skipped (tiny/invalid crop): {counters['words_skipped_tiny_crop']}")
    for split in SPLITS:
        print(f"Split {split}: {split_counts[split]} words -> {split_dirs[split]}")
    print(f"Manifest: {manifest_path}")


def main() -> None:
    """Run Bessarion preprocessing using CLI arguments."""
    args = parse_arguments()
    prepare_bessarion_dataset(
        bessarion_root=Path(args.bessarion_root),
        output_root=Path(args.output_root),
        max_pages=args.max_pages,
        max_words=args.max_words,
        skip_backup_xml=args.skip_backup_xml,
    )


if __name__ == "__main__":
    main()
