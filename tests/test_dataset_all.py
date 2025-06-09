import sys
import types
from pathlib import Path
import torch

# ensure project root is on the path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# stub out heavy matplotlib dependency
sys.modules.setdefault('matplotlib', types.ModuleType('matplotlib'))
sys.modules.setdefault('matplotlib.pyplot', types.ModuleType('matplotlib.pyplot'))

from htr_base.utils.htr_dataset import HTRDataset

def test_all_split_loads_everything():
    proj_root = Path(__file__).resolve().parents[1]
    ds_path = proj_root / "htr_base" / "data" / "GW" / "processed_words"
    train_ds = HTRDataset(str(ds_path), subset="train", fixed_size=(64, 256))
    val_ds = HTRDataset(str(ds_path), subset="val", fixed_size=(64, 256))
    test_ds = HTRDataset(str(ds_path), subset="test", fixed_size=(64, 256))
    all_ds = HTRDataset(str(ds_path), subset="all", fixed_size=(64, 256))

    assert len(all_ds) == len(train_ds) + len(val_ds) + len(test_ds)

    sample = all_ds[0]
    assert isinstance(sample[0], torch.Tensor)
    assert isinstance(sample[1], str)
