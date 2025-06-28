from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from htr_base.utils.htr_dataset import HTRDataset
import torch


def test_train_val_subset():
    base = Path("htr_base/data/GW/processed_words")
    train = HTRDataset(basefolder=str(base), subset="train", fixed_size=(64, 256))
    val = HTRDataset(basefolder=str(base), subset="val", fixed_size=(64, 256))
    train_val = HTRDataset(basefolder=str(base), subset="train_val", fixed_size=(64, 256))

    assert len(train_val) == len(train) + len(val)

    img_train, _, _ = train[0]
    img_tv, _, _ = train_val[0]
    assert img_train.shape == img_tv.shape


def test_maybe_load_backbone():
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    from alignment.alignment_trainer import maybe_load_backbone
    from alignment.alignment_trainer import cfg as base_cfg
    from htr_base.models import HTRNet
    from htr_base.utils.vocab import load_vocab

    local_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    local_cfg.load_pretrained_backbone = True
    local_cfg.pretrained_backbone_path = "htr_base/saved_models/pretrained_backbone.pt"
    local_cfg.device = "cpu"

    c2i, _ = load_vocab()
    arch = SimpleNamespace(**local_cfg["architecture"])
    backbone = HTRNet(arch, nclasses=len(c2i) + 1)

    before = {k: p.clone() for k, p in backbone.state_dict().items()}
    maybe_load_backbone(backbone, local_cfg)
    after = backbone.state_dict()

    changed = any(not torch.allclose(before[k], after[k]) for k in before)
    assert changed

