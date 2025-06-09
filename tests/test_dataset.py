import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'htr_base'))


from htr_base.utils.htr_dataset import HTRDataset
from skimage.io import imsave

def test_plot_image_saves(tmp_path):
    base = tmp_path / 'data'
    train_dir = base / 'train'
    train_dir.mkdir(parents=True)
    img = np.zeros((10, 10), dtype=np.uint8)
    imsave(train_dir / 'sample.png', img)
    with open(train_dir / 'gt.txt', 'w') as f:
        f.write('sample hello')

    ds = HTRDataset(str(base), subset='train')
    out_dir = tmp_path / 'out'
    ds.plot_image(0, str(out_dir))
    assert os.path.exists(out_dir / 'sample.png')
