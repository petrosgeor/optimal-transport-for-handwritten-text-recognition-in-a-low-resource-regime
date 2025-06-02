#!/usr/bin/env python3
"""
Usage example:
python prepare_gw.py \
    --dataset_folder /path/to/GW \
    --output_path ./data/GW/processed_word \
    --level word

Or for line-level processing:
python prepare_gw.py \
    --dataset_folder /path/to/GW \
    --output_path ./data/GW/processed_line \
    --level line
"""
import os, io, argparse
import numpy as np
from skimage import io as skio
from PIL import Image
from tqdm import tqdm

def decide_split(page_id):
    # GW fold mapping: 1=train, 2=val, 3=test
    if 270 <= page_id <= 279:
        return 1
    if 300 <= page_id <= 304:
        return 2
    if 305 <= page_id <= 309:
        return 3
    raise ValueError(f'Invalid page id: {page_id}')

def load_word_data(root, allowed_folds):
    gw_root = os.path.join(root, 'GeorgeWashington20')
    pages_dir = os.path.join(gw_root, 'gw4860', 'pages')
    gt_dir    = os.path.join(gw_root, 'gw4860', 'ground_truth')
    data = []
    for fn in sorted(os.listdir(pages_dir)):
        if not fn.endswith('.tif'): continue
        pid = int(fn.split('.')[0][:3])
        split = decide_split(pid)
        if split not in allowed_folds: continue
        img = skio.imread(os.path.join(pages_dir, fn))
        #img = (255 - img).astype(np.uint8)  # invert to black text on white
        ann = fn.replace('.tif', '.gtp')
        for line in io.open(os.path.join(gt_dir, ann), 'rb'):
            ulx, uly, lrx, lry, txt = line.split(b' ', 4)
            ulx,uly,lrx,lry = map(int, (ulx,uly,lrx,lry))
            crop = img[uly:lry, ulx:lrx]
            if crop.size == 0:
                continue
            data.append((crop, txt.decode('utf-8').strip()))
    return data


def load_line_data(root, allowed_folds):
    gw_root = os.path.join(root, 'GeorgeWashington20')
    img_dir = os.path.join(gw_root, 'washingtondb-v1.0', 'data', 'line_images_normalized')
    ann_f   = os.path.join(gw_root, 'washingtondb-v1.0', 'ground_truth', 'transcription.txt')
    data = []
    for ln in io.open(ann_f, encoding='utf-8-sig'):
        fn, raw = ln.split(' ', 1)
        pid = int(fn[:3])
        split = decide_split(pid)
        if split not in allowed_folds: continue
        imgfn = f"{pid:03d}-{int(fn[4:6]):02d}.png"
        path = os.path.join(img_dir, imgfn)
        if not os.path.isfile(path): continue
        img = skio.imread(path)
        arr = (img * 255).astype(np.uint8) if img.dtype == np.float32 else img
        data.append((arr, raw.strip()))
    return data


def prepare(args):
    # use official folds
    folds = { 'train': [1], 'val': [2], 'test': [3] }
    loader = load_word_data if args.level == 'word' else load_line_data

    splits = {}
    for name, allowed in folds.items():
        splits[name] = loader(args.dataset_folder, allowed)

    os.makedirs(args.output_path, exist_ok=True)
    for name, dataset in splits.items():
        outd = os.path.join(args.output_path, name)
        os.makedirs(outd, exist_ok=True)
        gt_lines = []
        for i, (img, txt) in enumerate(tqdm(dataset, desc=f"Writing {name}")):
            fn_base = f"{name}_{i:06d}"
            fn = f"{fn_base}.png"
            Image.fromarray(img).save(os.path.join(outd, fn))
            gt_lines.append(f"{fn_base} {txt}\n")
        with open(os.path.join(outd, 'gt.txt'), 'w', encoding='utf-8') as f:
            f.writelines(gt_lines)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_folder', required=True,
                   help='Root GW folder (contains GeorgeWashington20/...)')
    p.add_argument('--output_path',    required=True,
                   help='Where to write processed/{train,val,test}/')
    p.add_argument('--level', choices=['word','line'], default='word',
                   help='Crop level: word or line')
    args = p.parse_args()
    prepare(args)