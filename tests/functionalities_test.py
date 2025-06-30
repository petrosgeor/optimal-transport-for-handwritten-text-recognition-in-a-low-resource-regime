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
from alignment.trainer import refine_visual_backbone, cfg
from alignment.losses import ProjectionLoss
from htr_base.utils import build_phoc_description
from omegaconf import OmegaConf
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


def test_pretraining_yaml_loaded():
    yaml_cfg = OmegaConf.load("alignment/alignment_configs/pretraining_config.yaml")
    from alignment import pretraining
    assert pretraining.PRETRAINING_CONFIG["batch_size"] == yaml_cfg["batch_size"]
    assert pretraining.PRETRAINING_CONFIG["train_set_size"] == yaml_cfg["train_set_size"]
    assert pretraining.PRETRAINING_CONFIG["use_augmentations"] == yaml_cfg["use_augmentations"]


def test_maybe_load_backbone():
    from types import SimpleNamespace
    from alignment.trainer import maybe_load_backbone
    from alignment.trainer import cfg as base_cfg
    from htr_base.models import HTRNet
    from htr_base.utils.vocab import load_vocab

    local_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    local_cfg.load_pretrained_backbone = True
    local_cfg.pretrained_backbone_path = "htr_base/saved_models/pretrained_backbone.pt"
    local_cfg.device = "cpu"

    pre_cfg = OmegaConf.load("alignment/alignment_configs/pretraining_config.yaml")
    c2i, _ = load_vocab()
    arch = SimpleNamespace(**pre_cfg["architecture"])
    backbone = HTRNet(arch, nclasses=len(c2i) + 1)

    before = {k: p.clone() for k, p in backbone.state_dict().items()}
    maybe_load_backbone(backbone, local_cfg)
    after = backbone.state_dict()

    changed = any(not torch.allclose(before[k], after[k]) for k in before)
    assert changed
def test_refine_backbone_with_pretraining(tmp_path):
    base = Path("htr_base/data/GW/processed_words")
    class DummyCfg:
        k_external_words = 0
        n_aligned = 0

    ds = HTRDataset(basefolder=str(base), subset="train", fixed_size=(64, 256), config=DummyCfg())
    ds.external_words = [ds.transcriptions[0].strip()]
    ds.aligned[0] = 0
    ds.word_emb_dim = 8
    ds.external_word_embeddings = torch.zeros(len(ds.external_words), ds.word_emb_dim)

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
    pre_cfg = OmegaConf.load("alignment/alignment_configs/pretraining_config.yaml")
    arch = SimpleNamespace(**pre_cfg["architecture"])
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
    pre_cfg = OmegaConf.load("alignment/alignment_configs/pretraining_config.yaml")
    arch = SimpleNamespace(**pre_cfg["architecture"])
    arch.feat_dim = 32
    arch.phoc_levels = None
    backbone = HTRNet(arch, nclasses=len(c2i) + 1)
    projector = Projector(arch.feat_dim, ds.word_emb_dim, dropout=0.2)

    from alignment.alignment_utilities import OTAligner

    aligner = OTAligner(ds, backbone, [projector], batch_size=2, device="cpu", k=1)
    plan, proj, moved = aligner.align()

    assert plan.shape == (len(ds), len(ds.external_words))
    assert proj.shape == (len(ds), ds.word_emb_dim)
    assert moved.shape[0] == len(ds)


def test_alternating_refinement_uses_pretrain_ds(tmp_path, monkeypatch):
    base = Path("htr_base/data/GW/processed_words")
    ds = HTRDataset(basefolder=str(base), subset="train", fixed_size=(64, 256))
    ds.external_words = [ds.transcriptions[0].strip()]
    ds.aligned[0] = 0

    # create tiny synthetic dataset
    syn_dir = tmp_path / "syn"
    syn_dir.mkdir()
    img_src = ds.data[1][0]
    syn_img = syn_dir / "synth.png"
    shutil.copy(img_src, syn_img)
    list_file = syn_dir / "list.txt"
    with open(list_file, "w") as f:
        f.write("synth.png\n")

    local_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    local_cfg.syn_batch_ratio = 0.5
    local_cfg.synthetic_dataset = {
        "list_file": str(list_file),
        "base_path": str(syn_dir),
        "n_random": 1,
        "fixed_size": [64, 256],
    }
    local_cfg.device = "cpu"
    local_cfg.load_pretrained_backbone = False

    used = {}

    def fake_refine(*args, pretrain_ds=None, **kwargs):
        used["length"] = len(pretrain_ds) if pretrain_ds is not None else 0

    monkeypatch.setattr("alignment.trainer.refine_visual_backbone", fake_refine)
    monkeypatch.setattr("alignment.trainer.align_more_instances", lambda *a, **k: None)
    monkeypatch.setattr("alignment.trainer.cfg", local_cfg)

    arch = SimpleNamespace(**local_cfg["architecture"])
    net = HTRNet(arch, nclasses=len(ds.character_classes) + 1)
    proj = [Projector(arch.feat_dim, ds.word_emb_dim, dropout=0.2)]

    from alignment.trainer import alternating_refinement

    alternating_refinement(
        ds,
        net,
        proj,
        rounds=1,
        backbone_epochs=0,
        projector_epochs=0,
        align_kwargs={"k": 0},
    )

    assert used.get("length", 0) == 1


def test_compute_cer_prints(capsys):
    base = Path("htr_base/data/GW/processed_words")
    ds = HTRDataset(basefolder=str(base), subset="train", fixed_size=(64, 256))
    subset = torch.utils.data.Subset(ds, list(range(2)))

    c2i, _ = load_vocab()
    pre_cfg = OmegaConf.load("alignment/alignment_configs/pretraining_config.yaml")
    arch = SimpleNamespace(**pre_cfg["architecture"])
    net = HTRNet(arch, nclasses=len(c2i) + 1)

    from alignment.eval import compute_cer

    score = compute_cer(subset, net, batch_size=2, device="cpu")
    captured = capsys.readouterr()
    assert "CER:" in captured.out
    assert isinstance(score, float)


def test_refine_visual_model_uses_test_split(monkeypatch):
    base = Path("htr_base/data/GW/processed_words")
    class DummyCfg:
        k_external_words = 0
        n_aligned = 1

    ds = HTRDataset(basefolder=str(base), subset="train", fixed_size=(64, 256), config=DummyCfg())
    c2i, _ = load_vocab()
    pre_cfg = OmegaConf.load("alignment/alignment_configs/pretraining_config.yaml")
    arch = SimpleNamespace(**pre_cfg["architecture"])
    net = HTRNet(arch, nclasses=len(c2i) + 1)

    called = {}

    def fake_compute_cer(dataset, *args, **kwargs):
        called["subset"] = getattr(dataset, "subset", None)
        return 0.0

    monkeypatch.setattr("tests.train_by_length.compute_cer", fake_compute_cer)

    from tests.train_by_length import refine_visual_model

    refine_visual_model(ds, net, num_epochs=1, batch_size=2, lr=1e-4, syn_batch_ratio=None, pretrain_ds=None)

    assert called.get("subset") == "test"
def test_projector_dropout_behavior():
    proj = Projector(8, 4, dropout=0.5)
    x = torch.ones(2, 8)

    proj.train()
    out1 = proj(x)
    out2 = proj(x)
    assert not torch.allclose(out1, out2)

    proj.eval()
    out3 = proj(x)
    out4 = proj(x)
    assert torch.allclose(out3, out4)


def test_build_phoc_description_basic():
    c2i, _ = load_vocab()
    words = ["hello", "world"]
    phoc = build_phoc_description(words, c2i, levels=[1, 2])
    assert phoc.shape == (2, len(c2i) * 3)
    assert phoc.dtype == torch.bool


def test_refine_backbone_with_phoc(tmp_path):
    base = Path("htr_base/data/GW/processed_words")

    class DummyCfg:
        k_external_words = 0
        n_aligned = 0

    ds = HTRDataset(basefolder=str(base), subset="train", fixed_size=(64, 256), config=DummyCfg())
    ds.external_words = [ds.transcriptions[0].strip()]
    ds.aligned[0] = 0
    ds.word_emb_dim = 8
    ds.external_word_embeddings = torch.zeros(len(ds.external_words), ds.word_emb_dim)

    c2i, _ = load_vocab()
    arch = SimpleNamespace(**cfg["architecture"])
    net = HTRNet(arch, nclasses=len(c2i) + 1)

    refine_visual_backbone(
        ds,
        net,
        num_epochs=1,
        batch_size=2,
        lr=1e-4,
        enable_phoc=True,
        phoc_levels=tuple(arch.phoc_levels),
        phoc_weight=0.1,
    )

    assert not net.training


def test_projection_loss_weight():
    d = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    e = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
    a = torch.tensor([-1, 1])
    p = torch.tensor([0.5, 0.5])

    loss_default = ProjectionLoss()(d, e, a, p)
    loss_weighted = ProjectionLoss(supervised_weight=2.0)(d, e, a, p)
    assert loss_weighted > loss_default



def test_generate_knowledge_graph(tmp_path):
    from scripts.generate_knowledge_graph import build_repo_graph
    graphml, json_path = build_repo_graph(["alignment", "htr_base"],
                                           graphml_path=tmp_path/"kg.graphml",
                                           json_path=tmp_path/"kg.json")
    assert (tmp_path/"kg.graphml").exists()
    assert (tmp_path/"kg.json").exists()

