#!/usr/bin/env python3
"""
Usage example:
python prepare_gw.py \
    --dataset_folder /path/to/washingtondb-v1.0 \
    --output_path ./data/GW/processed_word \
    --level word

Or for line-level processing:
python prepare_gw.py \
    --dataset_folder /path/to/washingtondb-v1.0 \
    --output_path ./data/GW/processed_line \
    --level line

`--dataset_folder` accepts either:
1) WashingtonDB root containing `data/` and `ground_truth/`, or
2) Legacy parent containing `GeorgeWashington20/washingtondb-v1.0/`.
"""
import argparse
import io
import os
import re
import shutil
from collections import Counter

import numpy as np
from skimage import io as skio
from PIL import Image
from tqdm import tqdm

def decide_split(page_id):
    """Map a GW page id to the official train/val/test split index.

    Args:
    - page_id (int): The numeric page id parsed from a GW sample id.

    Returns:
    - (int): Fold id `1` (train), `2` (validation), or `3` (test).
    """

    # GW fold mapping: 1=train, 2=val, 3=test
    if 270 <= page_id <= 279:
        return 1
    if 300 <= page_id <= 304:
        return 2
    if 305 <= page_id <= 309:
        return 3
    raise ValueError(f'Invalid page id: {page_id}')


def resolve_washingtondb_root(dataset_folder):
    """Resolve the WashingtonDB root for supported on-disk layouts.

    Args:
    - dataset_folder (str): User-provided dataset folder argument.

    Returns:
    - (str): Absolute path to the resolved `washingtondb-v1.0` root.

    Raises:
    - FileNotFoundError: If no supported WashingtonDB layout is found.
    """

    direct_root = os.path.abspath(dataset_folder)
    direct_word_dir = os.path.join(direct_root, 'data', 'word_images_normalized')
    legacy_root = os.path.join(direct_root, 'GeorgeWashington20', 'washingtondb-v1.0')
    legacy_word_dir = os.path.join(legacy_root, 'data', 'word_images_normalized')

    if os.path.isdir(direct_word_dir):
        return direct_root
    if os.path.isdir(legacy_word_dir):
        return legacy_root

    raise FileNotFoundError(
        "Could not locate WashingtonDB word images. Expected one of:\n"
        f"- {direct_word_dir}\n"
        f"- {legacy_word_dir}"
    )


def decode_washington_word_label(tokenized_label, unknown_token_counter=None):
    """Decode a WashingtonDB hyphen-tokenized word label into GW charset text.

    Args:
    - tokenized_label (str): Tokenized WashingtonDB label, such as
      `s_2-s_7-s_0-s_pt` or `L-e-t-t-e-r-s-s_cm`.
    - unknown_token_counter (Counter | None): Optional counter that tracks
      tokens dropped due to unknown decode rules.

    Returns:
    - (str): Normalized transcription restricted to `[a-z0-9 ]`.
    """

    punctuation_tokens = {
        's_pt', 's_cm', 's_qo', 's_qt', 's_sq', 's_mi',
        's_sl', 's_bl', 's_br', 's_lb',
    }
    decoded_parts = []

    for token in tokenized_label.split('-'):
        if not token:
            continue
        if len(token) == 1 and token.isalpha():
            decoded_parts.append(token.lower())
            continue
        if token in punctuation_tokens:
            continue
        if token == 's_s':
            decoded_parts.append('s')
            continue
        if token == 's_et':
            decoded_parts.append('et')
            continue
        if token in {'s_GW', 's_gw'}:
            decoded_parts.append('gw')
            continue
        if token.startswith('s_'):
            suffix = token[2:].lower()
            if re.fullmatch(r'[0-9]+(?:st|nd|rd|th)?', suffix):
                decoded_parts.append(suffix)
            elif unknown_token_counter is not None:
                unknown_token_counter[token] += 1
            continue
        if unknown_token_counter is not None:
            unknown_token_counter[token] += 1

    decoded = ''.join(decoded_parts).lower()
    decoded = re.sub(r'[^a-z0-9 ]+', ' ', decoded)
    decoded = re.sub(r'\s+', ' ', decoded).strip()
    return decoded


def load_washington_word_data(washington_root, allowed_folds, unknown_token_counter=None):
    """Load normalized WashingtonDB word images and decoded word labels.

    Args:
    - washington_root (str): Resolved path to the WashingtonDB root folder.
    - allowed_folds (list[int]): Fold ids to keep (1=train, 2=val, 3=test).
    - unknown_token_counter (Counter | None): Optional token histogram for
      unknown decode tokens.

    Returns:
    - (list[tuple[str, str, str]]): `(src_img_path, decoded_text, word_id)`
      entries for samples that belong to the requested folds.
    """

    img_dir = os.path.join(washington_root, 'data', 'word_images_normalized')
    ann_f = os.path.join(washington_root, 'ground_truth', 'word_labels.txt')
    if not os.path.isfile(ann_f):
        raise FileNotFoundError(f'Word labels file not found: {ann_f}')

    data = []
    for ln in io.open(ann_f, encoding='utf-8-sig'):
        parts = ln.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue

        word_id, tokenized_label = parts
        page_id = int(word_id.split('-')[0])
        split = decide_split(page_id)
        if split not in allowed_folds:
            continue

        img_path = os.path.join(img_dir, f'{word_id}.png')
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f'Expected word image missing: {img_path}')

        decoded_label = decode_washington_word_label(tokenized_label, unknown_token_counter)
        data.append((img_path, decoded_label, word_id))
    return data


def load_legacy_word_data(root, allowed_folds):
    """Load legacy GW word crops from the `gw4860` page/gtp layout.

    Args:
    - root (str): Root folder that contains `GeorgeWashington20/gw4860`.
    - allowed_folds (list[int]): Fold ids to keep (1=train, 2=val, 3=test).

    Returns:
    - (list[tuple[np.ndarray, str]]): `(crop_array, raw_text)` entries.
    """

    gw_root = os.path.join(root, 'GeorgeWashington20')
    pages_dir = os.path.join(gw_root, 'gw4860', 'pages')
    gt_dir = os.path.join(gw_root, 'gw4860', 'ground_truth')
    data = []
    for fn in sorted(os.listdir(pages_dir)):
        if not fn.endswith('.tif'):
            continue
        pid = int(fn.split('.')[0][:3])
        split = decide_split(pid)
        if split not in allowed_folds:
            continue
        img = skio.imread(os.path.join(pages_dir, fn))
        # img = (255 - img).astype(np.uint8)  # invert to black text on white
        ann = fn.replace('.tif', '.gtp')
        for line in io.open(os.path.join(gt_dir, ann), 'rb'):
            ulx, uly, lrx, lry, txt = line.split(b' ', 4)
            ulx, uly, lrx, lry = map(int, (ulx, uly, lrx, lry))
            crop = img[uly:lry, ulx:lrx]
            if crop.size == 0:
                continue
            data.append((crop, txt.decode('utf-8').strip()))
    return data


def load_word_data(root, allowed_folds, unknown_token_counter=None):
    """Load word-level GW data with WashingtonDB-first layout detection.

    Args:
    - root (str): Dataset folder passed by the CLI.
    - allowed_folds (list[int]): Fold ids to keep (1=train, 2=val, 3=test).
    - unknown_token_counter (Counter | None): Optional token histogram for
      unknown WashingtonDB word tokens.

    Returns:
    - (list[tuple]): Either Washington tuples
      `(src_img_path, decoded_text, word_id)` or legacy tuples
      `(crop_array, raw_text)`.
    """

    try:
        washington_root = resolve_washingtondb_root(root)
    except FileNotFoundError:
        return load_legacy_word_data(root, allowed_folds)
    return load_washington_word_data(
        washington_root=washington_root,
        allowed_folds=allowed_folds,
        unknown_token_counter=unknown_token_counter,
    )


def load_line_data(root, allowed_folds):
    """Load line-level images and transcriptions from WashingtonDB.

    Args:
    - root (str): Dataset folder passed by the CLI.
    - allowed_folds (list[int]): Fold ids to keep (1=train, 2=val, 3=test).

    Returns:
    - (list[tuple[np.ndarray, str]]): `(line_image_array, raw_transcription)`
      entries for the requested folds.
    """

    try:
        washington_root = resolve_washingtondb_root(root)
    except FileNotFoundError:
        washington_root = os.path.join(root, 'GeorgeWashington20', 'washingtondb-v1.0')

    img_dir = os.path.join(washington_root, 'data', 'line_images_normalized')
    ann_f = os.path.join(washington_root, 'ground_truth', 'transcription.txt')
    if not os.path.isfile(ann_f):
        raise FileNotFoundError(f'Line transcription file not found: {ann_f}')

    data = []
    for ln in io.open(ann_f, encoding='utf-8-sig'):
        fn, raw = ln.split(' ', 1)
        pid = int(fn[:3])
        split = decide_split(pid)
        if split not in allowed_folds:
            continue
        imgfn = f"{pid:03d}-{int(fn[4:6]):02d}.png"
        path = os.path.join(img_dir, imgfn)
        if not os.path.isfile(path):
            continue
        img = skio.imread(path)
        arr = (img * 255).astype(np.uint8) if img.dtype == np.float32 else img
        data.append((arr, raw.strip()))
    return data


def prepare(args):
    """Preprocess GW data into train/val/test folders with `gt.txt` labels.

    Args:
    - args (argparse.Namespace): Parsed CLI args with `dataset_folder`,
      `output_path`, and `level`.

    Notes:
    - Samples whose decoded transcription is empty are skipped entirely (no image
      file is written and no `gt.txt` line is created). Empty labels can occur
      when the WashingtonDB tokenized transcription contains only punctuation or
      unsupported tokens, which would otherwise cause downstream CTC target
      encoding to raise on zero-length strings.

    """

    # use official folds
    folds = {'train': [1], 'val': [2], 'test': [3]}
    unknown_token_counter = Counter()
    empty_decoded_labels = 0
    is_washington_word_mode = False
    skipped_empty_by_split = {name: 0 for name in folds}

    if args.level == 'word':
        loader = lambda root, allowed: load_word_data(
            root=root,
            allowed_folds=allowed,
            unknown_token_counter=unknown_token_counter,
        )
    else:
        loader = load_line_data

    splits = {}
    for name, allowed in folds.items():
        splits[name] = loader(args.dataset_folder, allowed)
        if args.level == 'word' and splits[name]:
            is_washington_word_mode = is_washington_word_mode or len(splits[name][0]) == 3

    os.makedirs(args.output_path, exist_ok=True)
    for name, dataset in splits.items():
        outd = os.path.join(args.output_path, name)
        os.makedirs(outd, exist_ok=True)
        gt_lines = []
        written_count = 0
        for i, sample in enumerate(tqdm(dataset, desc=f'Writing {name}')):
            if args.level == 'word' and len(sample) == 3:
                src_img_path, txt, word_id = sample
                if not txt:
                    empty_decoded_labels += 1
                    skipped_empty_by_split[name] += 1
                    continue
                shutil.copy2(src_img_path, os.path.join(outd, f'{word_id}.png'))
                gt_lines.append(f'{word_id} {txt}\n')
            else:
                img, txt = sample
                txt = str(txt).strip()
                if not txt:
                    skipped_empty_by_split[name] += 1
                    continue
                fn_base = f'{name}_{i:06d}'
                fn = f'{fn_base}.png'
                Image.fromarray(img).save(os.path.join(outd, fn))
                gt_lines.append(f'{fn_base} {txt}\n')
            written_count += 1
        with open(os.path.join(outd, 'gt.txt'), 'w', encoding='utf-8') as f:
            f.writelines(gt_lines)
        print(
            f'[{name}] wrote {written_count} samples '
            f'(skipped {skipped_empty_by_split[name]} empty labels)'
        )

    if args.level == 'word' and is_washington_word_mode:
        print(f'Empty decoded labels: {empty_decoded_labels}')
        if unknown_token_counter:
            print('Unknown decode tokens (top 15):')
            for token, count in unknown_token_counter.most_common(15):
                print(f'  {token}: {count}')
        else:
            print('Unknown decode tokens: none')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_folder', required=True,
                   help=(
                       'GW dataset root. Supports either washingtondb-v1.0 '
                       '(contains data/ and ground_truth/) or a parent '
                       'containing GeorgeWashington20/washingtondb-v1.0/.'
                   ))
    p.add_argument('--output_path', required=True,
                   help='Where to write processed/{train,val,test}/')
    p.add_argument('--level', choices=['word', 'line'], default='word',
                   help='Crop level: word or line')
    args = p.parse_args()
    prepare(args)
