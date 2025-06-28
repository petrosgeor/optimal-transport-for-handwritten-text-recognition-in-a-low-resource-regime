from pathlib import Path
import sys
import shutil

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from htr_base.utils.htr_dataset import HTRDataset
import torch

from htr_base.utils.htr_dataset import PretrainingHTRDataset
from htr_base.utils.vocab import load_vocab
from htr_base.models import HTRNet, Projector
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


def test_otaligner_shapes():
    base = Path("htr_base/data/GW/processed_words")
    ds = HTRDataset(basefolder=str(base), subset="train", fixed_size=(64, 256))

    # minimal external vocabulary
    ds.external_words = [ds.transcriptions[0].strip(), ds.transcriptions[1].strip()]
    ds.word_emb_dim = 8
    ds.external_word_embeddings = ds.find_word_embeddings(ds.external_words, n_components=8)
    ds.aligned[:] = -1
    ds.aligned[0] = 0

    c2i, _ = load_vocab()
    arch = SimpleNamespace(**cfg.architecture)
    arch.feat_dim = 32
    arch.phoc_levels = None
    backbone = HTRNet(arch, nclasses=len(c2i) + 1)
    projector = Projector(arch.feat_dim, ds.word_emb_dim)

    from alignment.alignment_utilities import OTAligner

    aligner = OTAligner(ds, backbone, [projector], batch_size=2, device="cpu", k=1)
    plan, proj, moved = aligner.align()

    assert plan.shape == (len(ds), len(ds.external_words))
    assert proj.shape == (len(ds), ds.word_emb_dim)
    assert moved.shape[0] == len(ds)

