from pathlib import Path
import sys
import shutil

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from htr_base.utils.htr_dataset import HTRDataset
from htr_base.utils.htr_dataset import PretrainingHTRDataset
from htr_base.utils.vocab import load_vocab
from htr_base.models import HTRNet
from alignment.alignment_trainer import refine_visual_backbone, cfg
from types import SimpleNamespace


def test_train_val_subset():
    base = Path("htr_base/data/GW/processed_words")
    train = HTRDataset(basefolder=str(base), subset="train", fixed_size=(64, 256))
    val = HTRDataset(basefolder=str(base), subset="val", fixed_size=(64, 256))
    train_val = HTRDataset(basefolder=str(base), subset="train_val", fixed_size=(64, 256))

    assert len(train_val) == len(train) + len(val)

    img_train, _, _ = train[0]
    img_tv, _, _ = train_val[0]
    assert img_train.shape == img_tv.shape


def test_refine_backbone_with_pretraining(tmp_path):
    base = Path("htr_base/data/GW/processed_words")
    ds = HTRDataset(basefolder=str(base), subset="train", fixed_size=(64, 256))
    ds.external_words = [ds.transcriptions[0].strip()]
    ds.aligned[0] = 0

    pre_dir = tmp_path / "pretrain"
    pre_dir.mkdir()
    import shutil
    img_src = ds.data[1][0]
    img_path = pre_dir / "000_word_0.png"
    shutil.copy(img_src, img_path)
    list_file = pre_dir / "imlist.txt"
    with open(list_file, "w") as f:
        f.write(Path(img_path).name + "\n")
    pre_ds = PretrainingHTRDataset(str(list_file), fixed_size=(64, 256), base_path=str(pre_dir))

    c2i, _ = load_vocab()
    arch = SimpleNamespace(**cfg.architecture)
    net = HTRNet(arch, nclasses=len(c2i) + 1)

    refine_visual_backbone(ds, net, num_epochs=1, batch_size=2, lr=1e-4,
                           pretrain_ds=pre_ds, syn_batch_ratio=0.5)

    assert not net.training

