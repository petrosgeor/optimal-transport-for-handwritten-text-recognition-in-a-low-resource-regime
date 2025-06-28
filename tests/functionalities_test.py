from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from htr_base.utils.htr_dataset import HTRDataset


def test_train_val_subset():
    base = Path("htr_base/data/GW/processed_words")
    train = HTRDataset(basefolder=str(base), subset="train", fixed_size=(64, 256))
    val = HTRDataset(basefolder=str(base), subset="val", fixed_size=(64, 256))
    train_val = HTRDataset(basefolder=str(base), subset="train_val", fixed_size=(64, 256))

    assert len(train_val) == len(train) + len(val)

    img_train, _, _ = train[0]
    img_tv, _, _ = train_val[0]
    assert img_train.shape == img_tv.shape

