import sys
from pathlib import Path
import torch
import pytest

# Add project root to path for imports
proj_root = Path(__file__).resolve().parents[1]
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from htr_base.utils.htr_dataset import HTRDataset

class DummyCfg:
    def __init__(self):
        self.k_external_words = 0
        self.n_aligned = 0
        self.word_emb_dim = 32

def _dataset(two_views=False, concat_prob=0.0):
    data_path = proj_root / "htr_base" / "data" / "GW" / "processed_words"
    return HTRDataset(str(data_path), subset="train", fixed_size=(64, 256),
                      transforms=None, config=DummyCfg(),
                      two_views=two_views, concat_prob=concat_prob)

@pytest.mark.parametrize("two_views,concat", [(False, 0.0), (False, 1.0), (True, 0.0), (True, 1.0)])
def test_getitem_fixed_size(two_views, concat, capsys):
    ds = _dataset(two_views=two_views, concat_prob=concat)
    item = ds[0]
    base_trans = ds.data[0][1]

    if two_views:
        imgs, transcr, _ = item
        print(imgs[0].shape, imgs[1].shape)
        assert imgs[0].shape[-2:] == ds.fixed_size
        assert imgs[1].shape[-2:] == ds.fixed_size
    else:
        img, transcr, _ = item
        print(img.shape)
        assert img.shape[-2:] == ds.fixed_size

    if concat > 0:
        expected = f" {base_trans}   {base_trans} "
    else:
        expected = f" {base_trans} "
    print(transcr)
    assert transcr == expected
