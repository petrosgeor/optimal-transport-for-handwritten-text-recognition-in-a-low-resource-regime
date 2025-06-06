import pathlib
import os, sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from htr_base.utils.htr_dataset import HTRDataset
from alignment.alignment_utilities import print_dataset_stats

def test_print_dataset_stats(capsys):
    data_dir = pathlib.Path('htr_base/data/GW/processed_words')
    ds = HTRDataset(str(data_dir), subset='train', fixed_size=(32, 128), transforms=None)
    capsys.readouterr()  # discard dataset init prints
    print_dataset_stats(ds)
    out, _ = capsys.readouterr()
    assert 'samples:' in out
    assert str(len(ds)) in out
