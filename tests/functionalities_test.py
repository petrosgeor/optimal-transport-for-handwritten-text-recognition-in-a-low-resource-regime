from pathlib import Path
import sys
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import os
import pickle

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from htr_base.utils.htr_dataset import HTRDataset, PretrainingHTRDataset
from htr_base.models import HTRNet, Projector, AttentivePool
from alignment.alignment_utilities import (
    align_more_instances,
    print_dataset_stats,
    plot_projector_tsne,
)
from alignment.alignment_trainer import tee_output
from alignment.alignment_trainer import (
    refine_visual_backbone,
    train_projector,
)
from htr_base.utils.vocab import load_vocab
from alignment.ctc_utils import encode_for_ctc
from alignment.losses import _ctc_loss_fn
import shutil
from htr_base.utils.transforms import aug_transforms
from omegaconf import OmegaConf


def test_align_logging(capsys):
    cfg = SimpleNamespace(k_external_words=5, n_aligned=0, word_emb_dim=8)
    base = 'htr_base/data/GW/processed_words'
    dataset = HTRDataset(base, subset='train', fixed_size=(32, 128), transforms=None, config=cfg)
    dataset.data = dataset.data[:3]
    dataset.transcriptions = dataset.transcriptions[:3]
    dataset.aligned = dataset.aligned[:3]
    dataset.is_in_dict = dataset.is_in_dict[:3]
    dataset.external_word_embeddings = dataset.find_word_embeddings(dataset.external_words, n_components=8)

    arch_cfg = SimpleNamespace(
        cnn_cfg=[[1, 16], 'M', [1, 32]],
        head_type='cnn',
        rnn_type='gru',
        rnn_layers=1,
        rnn_hidden_size=32,
        flattening='maxpool',
        stn=False,
        feat_dim=8,
    )
    backbone = HTRNet(arch_cfg, nclasses=len(dataset.character_classes) + 1)
    projector = Projector(arch_cfg.feat_dim, dataset.word_emb_dim)
    align_more_instances(dataset, backbone, projector, batch_size=1, device='cpu', k=1)
    out = capsys.readouterr().out
    assert '[Align] round accuracy' in out
    assert '[Align] cumulative accuracy' in out
    assert 'sample:' in out
    assert 'mean moved distance' in out


def test_letter_priors():
    ds = HTRDataset('htr_base/data/GW/processed_words', subset='train', fixed_size=(32, 128))
    # priors computed from wordfreq by default
    total = sum(ds.prior_char_probs.values())
    assert abs(total - 1.0) < 1e-4
    assert ('a' in ds.prior_char_probs) or ('0' in ds.prior_char_probs)

    wf_priors = HTRDataset.letter_priors()
    assert ds.prior_char_probs == wf_priors

    tr_priors = HTRDataset.letter_priors(ds.transcriptions)
    total_tr = sum(tr_priors.values())
    assert abs(total_tr - 1.0) < 1e-4


def test_external_words_lowercase():
    cfg = SimpleNamespace(k_external_words=5, n_aligned=0, word_emb_dim=8)
    dataset = HTRDataset(
        'htr_base/data/GW/processed_words',
        subset='train',
        fixed_size=(32, 128),
        transforms=None,
        config=cfg,
    )
    assert all(w.islower() for w in dataset.external_words)


def test_dataset_default_vocab(monkeypatch):
    def fake_load_vocab():
        return {"@": 1, "#": 2}, {1: "@", 2: "#"}

    monkeypatch.setattr("htr_base.utils.htr_dataset.load_vocab", fake_load_vocab)
    ds = HTRDataset("htr_base/data/GW/processed_words", subset="train", fixed_size=(32, 128))
    assert ds.character_classes == ["@", "#"]


def test_tee_output(tmp_path, capsys):
    out_file = tmp_path / "log.txt"
    with tee_output(out_file):
        print("hello")
    captured = capsys.readouterr().out
    assert captured.strip() == "hello"
    assert out_file.read_text() == "hello\n"

    with tee_output(out_file):
        print("bye")
    assert out_file.read_text() == "bye\n"


def test_pretraining_tee_output(tmp_path, capsys):
    out_file = tmp_path / "log.txt"
    from alignment.pretraining import tee_output as pt_tee_output

    with pt_tee_output(out_file):
        print("hello")
    captured = capsys.readouterr().out
    assert captured.strip() == "hello"
    assert out_file.read_text() == "hello\n"



def test_dataset_prealignment():
    class DummyCfg:
        k_external_words = 200
        n_aligned = 300

    base = 'htr_base/data/GW/processed_words'
    dataset = HTRDataset(
        base,
        subset='train',
        fixed_size=(64, 256),
        transforms=aug_transforms,
        config=DummyCfg(),
    )
    assert (dataset.aligned != -1).sum().item() > 0

def test_align_zero_row(tmp_path):
    cfg = SimpleNamespace(k_external_words=5, n_aligned=0, word_emb_dim=8)
    base = 'htr_base/data/GW/processed_words'
    ds = HTRDataset(base, subset='train', fixed_size=(32, 128), transforms=None, config=cfg)
    ds.data = ds.data[:3]
    ds.transcriptions = ds.transcriptions[:3]
    ds.aligned = ds.aligned[:3]
    ds.is_in_dict = ds.is_in_dict[:3]
    emb = ds.find_word_embeddings(ds.external_words, n_components=8)
    ds.external_word_embeddings = emb * 1e6

    arch = SimpleNamespace(
        cnn_cfg=[[1, 16], 'M', [1, 32]],
        head_type='both',
        rnn_type='gru',
        rnn_layers=1,
        rnn_hidden_size=32,
        flattening='maxpool',
        stn=False,
        feat_dim=8,
    )
    backbone = HTRNet(arch, nclasses=len(ds.character_classes) + 1)
    projector = Projector(arch.feat_dim, ds.word_emb_dim)

    align_more_instances(ds, backbone, projector, batch_size=1, device='cpu', unbalanced=True, reg_m=0.1, k=1)
    assert (ds.aligned != -1).any(), "No samples were pseudo-labelled"


def test_print_dataset_stats(capsys):
    cfg = SimpleNamespace(k_external_words=5, n_aligned=0, word_emb_dim=8)
    base = 'htr_base/data/GW/processed_words'
    ds = HTRDataset(base, subset='train', fixed_size=(32, 128), transforms=None, config=cfg)
    ds.data = ds.data[:3]
    ds.transcriptions = ds.transcriptions[:3]
    ds.aligned = ds.aligned[:3]
    ds.is_in_dict = ds.is_in_dict[:3]
    print_dataset_stats(ds)
    out = capsys.readouterr().out
    assert 'external vocab size' in out
    assert 'in-dictionary samples' in out
    assert 'transcriptions lowercase' in out


def test_plot_projector_tsne(tmp_path):
    cfg = SimpleNamespace(k_external_words=5, n_aligned=0, word_emb_dim=8)
    base = 'htr_base/data/GW/processed_words'
    ds = HTRDataset(base, subset='train', fixed_size=(32, 128), transforms=None, config=cfg)
    ds.data = ds.data[:3]
    ds.transcriptions = ds.transcriptions[:3]
    ds.aligned = ds.aligned[:3]
    ds.is_in_dict = ds.is_in_dict[:3]
    ds.external_word_embeddings = ds.find_word_embeddings(ds.external_words, n_components=8)

    projections = torch.randn(len(ds), ds.word_emb_dim)
    out_file = tmp_path / 'proj_tsne.png'
    plot_projector_tsne(projections, ds, out_file)
    assert out_file.exists()


def test_refine_epochs_hparam():
    from alignment.alignment_trainer import cfg as loaded
    cfg = OmegaConf.load('alignment/config.yaml')
    assert loaded.refine_epochs == cfg['refine_epochs']


def test_projector_epochs_hparam():
    from alignment.alignment_trainer import cfg as loaded
    cfg = OmegaConf.load('alignment/config.yaml')
    assert loaded.projector_epochs == cfg['projector_epochs']


def test_no_alt_backbone_hparam():
    from alignment.alignment_trainer import cfg as loaded
    assert not hasattr(loaded, 'alt_backbone_epochs')


def test_majority_vote_alignment():
    cfg = SimpleNamespace(k_external_words=5, n_aligned=0, word_emb_dim=8)
    base = 'htr_base/data/GW/processed_words'
    ds = HTRDataset(base, subset='train', fixed_size=(32, 128), transforms=None, config=cfg)
    ds.data = ds.data[:2]
    ds.transcriptions = ds.transcriptions[:2]
    ds.aligned = torch.full((2,), -1, dtype=torch.int32)
    ds.is_in_dict = ds.is_in_dict[:2]
    ds.external_word_embeddings = ds.find_word_embeddings(ds.external_words, n_components=8)

    arch = SimpleNamespace(
        cnn_cfg=[[1, 16], 'M', [1, 32]],
        head_type='both',
        rnn_type='gru',
        rnn_layers=1,
        rnn_hidden_size=32,
        flattening='maxpool',
        stn=False,
        feat_dim=8,
    )
    backbone = HTRNet(arch, nclasses=len(ds.character_classes) + 1)

    class ConstProj(nn.Module):
        def __init__(self, vec):
            super().__init__()
            self.vec = nn.Parameter(vec, requires_grad=False)

        def forward(self, x):
            return self.vec.expand(x.size(0), -1)

    vec0 = ds.external_word_embeddings[0]
    vec1 = ds.external_word_embeddings[1]
    projectors = [ConstProj(vec0), ConstProj(vec0), ConstProj(vec1)]

    align_more_instances(ds, backbone, projectors, batch_size=1, device='cpu', k=2, agree_threshold=4)
    assert (ds.aligned == -1).all()
    align_more_instances(ds, backbone, projectors, batch_size=1, device='cpu', k=2, agree_threshold=1)
    assert (ds.aligned != -1).all()

def test_word_silhouette_score():
    feats = torch.tensor([[0.0, 0.0], [0.0, 1.0], [3.0, 0.0], [3.0, 1.0]])
    words = ["a", "a", "b", "b"]
    from htr_base.utils.metrics import word_silhouette_score

    score = word_silhouette_score(feats, words)
    assert 0.6 < score <= 1.0


def test_refine_prints_silhouette(capsys):
    cfg = SimpleNamespace(k_external_words=5, n_aligned=0, word_emb_dim=8)
    base = 'htr_base/data/GW/processed_words'
    ds = HTRDataset(base, subset='train', fixed_size=(32, 128), transforms=None, config=cfg)
    ds.data = ds.data[:3]
    ds.transcriptions = ds.transcriptions[:3]
    ds.aligned = torch.tensor([0, 0, 1], dtype=torch.int32)
    ds.is_in_dict = ds.is_in_dict[:3]
    ds.external_word_embeddings = ds.find_word_embeddings(ds.external_words, n_components=8)

    arch = SimpleNamespace(
        cnn_cfg=[[1, 16], 'M', [1, 32]],
        head_type='both',
        rnn_type='gru',
        rnn_layers=1,
        rnn_hidden_size=32,
        flattening='maxpool',
        stn=False,
        feat_dim=8,
    )
    backbone = HTRNet(arch, nclasses=len(ds.character_classes) + 1)

    refine_visual_backbone(ds, backbone, num_epochs=1, batch_size=1, lr=1e-3)
    out = capsys.readouterr().out
    assert 'silhouette score' in out.lower()


def _tiny_dataset():
    cfg = SimpleNamespace(k_external_words=5, n_aligned=0, word_emb_dim=8)
    base = 'htr_base/data/GW/processed_words'
    ds = HTRDataset(base, subset='train', fixed_size=(32, 128), transforms=None, config=cfg)
    ds.data = ds.data[:35]
    ds.transcriptions = ds.transcriptions[:35]
    ds.aligned = ds.aligned[:35]
    ds.is_in_dict = ds.is_in_dict[:35]
    ds.external_word_embeddings = ds.find_word_embeddings(ds.external_words, n_components=8)
    return ds


def test_train_projector_no_tsne(tmp_path):
    ds = _tiny_dataset()

    arch = SimpleNamespace(
        cnn_cfg=[[1, 16], 'M', [1, 32]],
        head_type='cnn',
        rnn_type='gru',
        rnn_layers=1,
        rnn_hidden_size=32,
        flattening='maxpool',
        stn=False,
        feat_dim=8,
    )
    backbone = HTRNet(arch, nclasses=len(ds.character_classes) + 1)
    projector = Projector(arch.feat_dim, ds.word_emb_dim)

    figs = Path('tests/figures')
    if figs.exists():
        shutil.rmtree(figs)

    train_projector(
        ds,
        backbone,
        projector,
        num_epochs=1,
        batch_size=1,
        lr=1e-3,
        num_workers=0,
        device='cpu',
        plot_tsne=False,
    )

    assert not figs.exists() or not any(figs.iterdir())


def test_train_projector_with_tsne(tmp_path):
    ds = _tiny_dataset()

    arch = SimpleNamespace(
        cnn_cfg=[[1, 16], 'M', [1, 32]],
        head_type='cnn',
        rnn_type='gru',
        rnn_layers=1,
        rnn_hidden_size=32,
        flattening='maxpool',
        stn=False,
        feat_dim=8,
    )
    backbone = HTRNet(arch, nclasses=len(ds.character_classes) + 1)
    projector = Projector(arch.feat_dim, ds.word_emb_dim)

    figs = Path('tests/figures')
    if figs.exists():
        shutil.rmtree(figs)

    train_projector(
        ds,
        backbone,
        projector,
        num_epochs=1,
        batch_size=1,
        lr=1e-3,
        num_workers=0,
        device='cpu',
        plot_tsne=True,
    )

    assert (figs / 'tsne_backbone.png').exists()
    assert any(figs.glob('tsne_projections_*.png'))
def test_wasserstein_L2():
    p = torch.tensor([0.2, 0.3, 0.5])
    q = torch.tensor([0.1, 0.4, 0.5])
    expected = torch.sqrt(torch.mean((p - q) ** 2))
    from tests.train_by_length import wasserstein_L2

    assert torch.isclose(wasserstein_L2(p, q), expected)


def test_decode_config(monkeypatch):
    from tests import train_by_length as tbl
    from torch.utils.data import DataLoader
    import torch.nn as nn

    calls = []

    def fake_beam(*args, **kwargs):
        calls.append("beam")
        return ["" for _ in range(args[0].shape[1])]

    def fake_greedy(*args, **kwargs):
        calls.append("greedy")
        return ["" for _ in range(args[0].shape[1])]

    monkeypatch.setattr(tbl, "beam_search_ctc_decode", fake_beam)
    monkeypatch.setattr(tbl, "greedy_ctc_decode", fake_greedy)

    class DummyModel(nn.Module):
        def forward(self, imgs, return_feats=False):
            return torch.zeros(2, imgs.size(0), 3)

    sample = (torch.zeros(1, 64, 256), " a ", 0)
    sample2 = (torch.zeros(1, 64, 256), " ab ", 0)
    loader = DataLoader([sample, sample2], batch_size=1)
    tbl.DECODE_CONFIG["method"] = "beam"
    tbl._evaluate_cer(DummyModel(), loader, {1: "a", 2: "b"}, torch.device("cpu"), k=1)
    assert "beam" in calls

    calls.clear()
    tbl.DECODE_CONFIG["method"] = "greedy"
    tbl._evaluate_cer(DummyModel(), loader, {1: "a", 2: "b"}, torch.device("cpu"), k=1)
    assert "greedy" in calls


def test_train_by_length_loads_checkpoint(monkeypatch):
    from tests import train_by_length as tbl

    loaded = {}

    def fake_torch_load(path, map_location=None):
        loaded["path"] = path
        return {"weights": True}

    def fake_load_state(self, state):
        loaded["state"] = state

    monkeypatch.setattr(tbl.torch, "load", fake_torch_load)
    monkeypatch.setattr(HTRNet, "load_state_dict", fake_load_state)

    arch = SimpleNamespace(
        cnn_cfg=[[1, 16]],
        head_type="cnn",
        rnn_type="gru",
        rnn_layers=1,
        rnn_hidden_size=16,
        flattening="maxpool",
        stn=False,
        feat_dim=None,
    )
    net = HTRNet(arch, nclasses=3)

    tbl.LOAD_PRETRAINED_BACKBONE = True
    tbl.maybe_load_pretrained(net, torch.device("cpu"))
    tbl.LOAD_PRETRAINED_BACKBONE = False

    assert loaded.get("path") == "htr_base/saved_models/pretrained_backbone.pt"
    assert loaded.get("state") == {"weights": True}


def test_encode_for_ctc_vocab_size():
    cfg = SimpleNamespace(k_external_words=5, n_aligned=0, word_emb_dim=8)
    base = "htr_base/data/GW/processed_words"
    ds = HTRDataset(base, subset="test", fixed_size=(32, 128), transforms=None, config=cfg)
    ds.data = ds.data[:3]
    ds.transcriptions = ds.transcriptions[:3]
    ds.aligned = ds.aligned[:3]
    ds.is_in_dict = ds.is_in_dict[:3]

    c2i, _ = load_vocab()
    chars = "".join(sorted(ds.character_classes))
    txt = f" {chars} "
    targets, _ = encode_for_ctc([txt], c2i)
    # Include the CTC blank to count all possible labels
    targets = torch.cat([targets, torch.tensor([0])])
    expected = len(set(ds.character_classes) | {" "}) + 1
    assert targets.unique().numel() == expected


def test_pretraining_dataset_filtering(tmp_path):
    src = Path('htr_base/data/GW/processed_words/train/train_000000.png')
    base = tmp_path / 'imgs'
    base.mkdir()

    good = base / 'foo_good_0.png'
    upper = base / 'foo_BAD_1.png'
    bad = base / 'foo_no$good_2.png'
    shutil.copy(src, good)
    shutil.copy(src, upper)
    shutil.copy(src, bad)

    list_file = tmp_path / 'list.txt'
    with open(list_file, 'w') as f:
        f.write('foo_good_0.png\n')
        f.write('foo_BAD_1.png\n')
        f.write('foo_no$good_2.png\n')

    ds = PretrainingHTRDataset(
        str(list_file), fixed_size=(32, 128), base_path=str(base), transforms=aug_transforms
    )
    assert len(ds) == 1
    img, trans = ds[0]
    assert img.shape == (1, 32, 128)
    assert trans.strip() == 'good'


def test_pretraining_dataset_n_random(tmp_path):
    src = Path('htr_base/data/GW/processed_words/train/train_000000.png')
    base = tmp_path / 'imgs'
    base.mkdir()
    for i in range(3):
        shutil.copy(src, base / f'foo_word_{i}.png')

    list_file = tmp_path / 'list.txt'
    with open(list_file, 'w') as f:
        for i in range(3):
            f.write(f'foo_word_{i}.png\n')

    ds1 = PretrainingHTRDataset(
        str(list_file), fixed_size=(32, 128), base_path=str(base), n_random=1
    )
    ds2 = PretrainingHTRDataset(
        str(list_file), fixed_size=(32, 128), base_path=str(base), n_random=1
    )
    assert len(ds1) == 1
    assert ds1.img_paths == ds2.img_paths


def test_pretraining_dataset_preload_images(tmp_path):
    src = Path('htr_base/data/GW/processed_words/train/train_000000.png')
    base = tmp_path
    shutil.copy(src, base / 'foo_word_0.png')

    list_file = tmp_path / 'list.txt'
    with open(list_file, 'w') as f:
        f.write('foo_word_0.png\n')

    ds = PretrainingHTRDataset(
        str(list_file), fixed_size=(32, 128), base_path=str(base), preload_images=True
    )
    assert len(ds.images) == len(ds)
    img, _ = ds[0]
    assert img.shape == (1, 32, 128)


def test_pretraining_dataset_save_image(tmp_path):
    src = Path('htr_base/data/GW/processed_words/train/train_000000.png')
    base = tmp_path
    shutil.copy(src, base / 'foo_word_0.png')

    list_file = tmp_path / 'list.txt'
    with open(list_file, 'w') as f:
        f.write('foo_word_0.png\n')

    ds = PretrainingHTRDataset(
        str(list_file), fixed_size=(32, 128), base_path=str(base)
    )

    out_dir = tmp_path / 'out'
    path = ds.save_image(0, str(out_dir))
    assert Path(path).exists()

    path2 = ds.save_image(0, str(out_dir), filename='second.png')
    assert Path(path2).exists() and Path(path2).name == 'second.png'


def test_loaded_image_shapes(tmp_path):
    src = Path('htr_base/data/GW/processed_words/train/train_000000.png')
    base = tmp_path
    shutil.copy(src, base / 'foo_word_0.png')

    list_file = tmp_path / 'list.txt'
    with open(list_file, 'w') as f:
        f.write('foo_word_0.png\n')

    ds = PretrainingHTRDataset(
        str(list_file), fixed_size=(32, 128), base_path=str(base), preload_images=True
    )
    shapes = ds.loaded_image_shapes()
    assert shapes == [ds.images[0].shape]

    ds2 = PretrainingHTRDataset(
        str(list_file), fixed_size=(32, 128), base_path=str(base)
    )
    with pytest.raises(RuntimeError):
        ds2.loaded_image_shapes()


def test_pretraining_script(tmp_path, capsys):
    src = Path('htr_base/data/GW/processed_words/train/train_000000.png')
    base = tmp_path / 'imgs'
    base.mkdir()
    for i in range(2):
        shutil.copy(src, base / f'foo_word_{i}.png')

    list_file = tmp_path / 'list.txt'
    with open(list_file, 'w') as f:
        for i in range(2):
            f.write(f'foo_word_{i}.png\n')

    save_dir = Path('htr_base/saved_models')
    if save_dir.exists():
        shutil.rmtree(save_dir)

    from alignment import pretraining

    config = {
        "list_file": str(list_file),
        "n_random": 1,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "base_path": str(base),
        "fixed_size": (32, 128),
        "device": "cpu",
        "save_path": str(save_dir / 'pretrained_backbone.pt'),
        "save_backbone": True,
    }

    pretraining.main(config)
    out = capsys.readouterr().out

    assert (save_dir / 'pretrained_backbone.pt').exists()
    assert 'GT:' in out and 'beam5:' in out


def test_pretraining_intermediate_decoding(tmp_path, capsys):
    src = Path('htr_base/data/GW/processed_words/train/train_000000.png')
    base = tmp_path / 'imgs'
    base.mkdir()
    for i in range(2):
        shutil.copy(src, base / f'foo_word_{i}.png')

    list_file = tmp_path / 'list.txt'
    with open(list_file, 'w') as f:
        for i in range(2):
            f.write(f'foo_word_{i}.png\n')

    save_dir = Path('htr_base/saved_models')
    if save_dir.exists():
        shutil.rmtree(save_dir)

    from alignment import pretraining

    config = {
        "list_file": str(list_file),
        "n_random": 2,
        "num_epochs": 6,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "base_path": str(base),
        "fixed_size": (32, 128),
        "device": "cpu",
        "save_path": str(save_dir / 'pretrained_backbone.pt'),
        "save_backbone": True,
    }

    pretraining.main(config)
    out = capsys.readouterr().out

    assert out.count('GT:') >= 2


def test_pretraining_script_logs(tmp_path):
    src = Path('htr_base/data/GW/processed_words/train/train_000000.png')
    base = tmp_path
    shutil.copy(src, base / 'foo_word_0.png')

    list_file = tmp_path / 'list.txt'
    with open(list_file, 'w') as f:
        f.write('foo_word_0.png\n')

    save_dir = tmp_path / 'model'

    log_file = Path('pretraining_results.txt')
    if log_file.exists():
        log_file.unlink()

    from alignment import pretraining

    config = {
        'list_file': str(list_file),
        'n_random': 1,
        'num_epochs': 1,
        'batch_size': 1,
        'learning_rate': 1e-3,
        'base_path': str(base),
        'fixed_size': (32, 128),
        'device': 'cpu',
        'save_path': str(save_dir / 'pretrained_backbone.pt'),
        'save_backbone': True,
        'results_file': True,
    }

    pretraining.main(config)


    assert log_file.exists()
    assert 'GT:' in log_file.read_text()


def test_pretraining_no_results_file(tmp_path):
    src = Path('htr_base/data/GW/processed_words/train/train_000000.png')
    base = tmp_path
    shutil.copy(src, base / 'foo_word_0.png')

    list_file = tmp_path / 'list.txt'
    with open(list_file, 'w') as f:
        f.write('foo_word_0.png\n')

    save_dir = tmp_path / 'model2'

    log_file = Path('pretraining_results.txt')
    if log_file.exists():
        log_file.unlink()

    from alignment import pretraining

    config = {
        'list_file': str(list_file),
        'n_random': 1,
        'num_epochs': 1,
        'batch_size': 1,
        'learning_rate': 1e-3,
        'base_path': str(base),
        'fixed_size': (32, 128),
        'device': 'cpu',
        'save_path': str(save_dir / 'pretrained_backbone.pt'),
        'save_backbone': True,
        'results_file': False,

    }

    pretraining.main(config)

    assert not log_file.exists()


def test_pretraining_uses_scheduler(tmp_path, monkeypatch):
    src = Path('htr_base/data/GW/processed_words/train/train_000000.png')
    base = tmp_path
    shutil.copy(src, base / 'foo_word_0.png')

    list_file = tmp_path / 'list.txt'
    with open(list_file, 'w') as f:
        f.write('foo_word_0.png\n')

    call = {}

    class DummyScheduler:
        def __init__(self, opt, step_size, gamma):
            call['step_size'] = step_size
            call['gamma'] = gamma
            self.opt = opt
            self.steps = 0

        def step(self):
            self.steps += 1
            call['steps'] = self.steps

        def get_last_lr(self):
            return [group['lr'] for group in self.opt.param_groups]

    from alignment import pretraining

    monkeypatch.setattr(pretraining.lr_scheduler, 'StepLR', DummyScheduler)

    config = {
        'list_file': str(list_file),
        'n_random': 1,
        'num_epochs': 1,
        'batch_size': 1,
        'learning_rate': 1e-3,
        'base_path': str(base),
        'fixed_size': (32, 128),
        'device': 'cpu',
        'save_path': str(tmp_path / 'pretrained.pt'),
        'save_backbone': True,
    }

    pretraining.main(config)

    assert call.get('step_size') == 1500
    assert call.get('gamma') == 0.5
    assert call.get('steps', 0) >= 1


def test_pretraining_evaluates_cer(tmp_path, capsys):
    src = Path('htr_base/data/GW/processed_words/train/train_000000.png')
    base = tmp_path / 'imgs'
    base.mkdir()
    for i in range(2):
        shutil.copy(src, base / f'foo_word_{i}.png')

    list_file = tmp_path / 'list.txt'
    with open(list_file, 'w') as f:
        for i in range(2):
            f.write(f'foo_word_{i}.png\n')

    from alignment import pretraining

    config = {
        'list_file': str(list_file),
        'n_random': 2,
        'num_epochs': 10,
        'batch_size': 1,
        'learning_rate': 1e-3,
        'base_path': str(base),
        'fixed_size': (32, 128),
        'device': 'cpu',
        'save_path': str(tmp_path / 'pretrained_backbone.pt'),
        'save_backbone': True,
    }

    pretraining.main(config)
    out = capsys.readouterr().out
    assert '[Eval] CER:' in out


def test_pretraining_saves_and_loads_dicts(tmp_path, capsys):
    src = Path('htr_base/data/GW/processed_words/train/train_000000.png')
    base = tmp_path / 'imgs'
    base.mkdir()
    shutil.copy(src, base / 'foo_word_0.png')

    list_file = tmp_path / 'list.txt'
    with open(list_file, 'w') as f:
        f.write('foo_word_0.png\n')

    save_dir = tmp_path / 'model'

    from alignment import pretraining

    config = {
        'list_file': str(list_file),
        'n_random': 1,
        'num_epochs': 1,
        'batch_size': 1,
        'learning_rate': 1e-3,
        'base_path': str(base),
        'fixed_size': (32, 128),
        'device': 'cpu',
        'save_path': str(save_dir / 'pretrained_backbone.pt'),
        'save_backbone': True,
    }

    pretraining.main(config)
    capsys.readouterr()

    assert (save_dir / 'c2i.pkl').exists()
    assert (save_dir / 'i2c.pkl').exists()

    pretraining.main(config)
    out = capsys.readouterr().out
    assert 'Loading vocabulary' not in out


def test_simple_train_script(tmp_path, capsys):
    from tests import simple_train

    config = {
        "basefolder": "htr_base/data/GW/processed_words",
        "n_examples": 2,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "device": "cpu",
        "eval_interval": 1,
    }
    simple_train.main(config)
    out = capsys.readouterr().out
    assert "CER:" in out


def test_pretraining_single_gpu(monkeypatch, tmp_path):
    from alignment import pretraining
    src = Path('htr_base/data/GW/processed_words/train/train_000000.png')
    base = tmp_path
    shutil.copy(src, base / 'foo_word_0.png')
    list_file = tmp_path / 'list.txt'
    with open(list_file, 'w') as f:
        f.write('foo_word_0.png\n')

    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    called = {}

    def fake_dp(module, device_ids=None):
        called['used'] = True
        return module

    monkeypatch.setattr(torch.nn, 'DataParallel', fake_dp)

    cfg = {
        'list_file': str(list_file),
        'n_random': 1,
        'num_epochs': 1,
        'batch_size': 1,
        'learning_rate': 1e-3,
        'base_path': str(base),
        'fixed_size': (32, 128),
        'device': 'cpu',
        'gpu_id': 3,
    }

    pretraining.main(cfg)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '3'
    assert not called


def test_vocab_dict_loading():
    from tests import train_by_length as tbl
    base = Path('htr_base/saved_models')
    with open(base / 'c2i.pkl', 'rb') as f:
        expected_c2i = pickle.load(f)
    with open(base / 'i2c.pkl', 'rb') as f:
        expected_i2c = pickle.load(f)
    c2i, i2c = load_vocab()
    assert c2i == expected_c2i and i2c == expected_i2c


def test_refine_with_pretraining(tmp_path, capsys):
    from tests import train_by_length
    src = Path('htr_base/data/GW/processed_words/train/train_000000.png')
    base = tmp_path / 'imgs'
    base.mkdir()
    for i in range(2):
        shutil.copy(src, base / f'foo_word_{i}.png')

    list_file = tmp_path / 'list.txt'
    with open(list_file, 'w') as f:
        for i in range(2):
            f.write(f'foo_word_{i}.png\n')

    pre_ds = PretrainingHTRDataset(
        str(list_file), fixed_size=(32, 128), base_path=str(base)
    )

    ds = _tiny_dataset()
    ds.config.n_aligned = 1
    ds.data = ds.data[:1]
    ds.data = [(ds.data[0][0], 'word')]
    ds.transcriptions = ['word']
    ds.aligned = ds.aligned[:1]
    ds.is_in_dict = ds.is_in_dict[:1]

    arch = SimpleNamespace(
        cnn_cfg=[[1, 16]],
        head_type='cnn',
        rnn_type='gru',
        rnn_layers=1,
        rnn_hidden_size=16,
        flattening='maxpool',
        stn=False,
        feat_dim=None,
    )
    c2i, _ = load_vocab()
    net = HTRNet(arch, nclasses=len(c2i) + 1)
    train_by_length.refine_visual_model(
        ds,
        net,
        num_epochs=1,
        batch_size=1,
        lr=1e-3,
        max_length=10,
        pretrain_ds=pre_ds,
        syn_batch_ratio=0.5,
    )
    out = capsys.readouterr().out
    assert 'Epoch 001/1' in out


def _make_pre_ds(tmp_path):
    src = Path('htr_base/data/GW/processed_words/train/train_000000.png')
    base = tmp_path / 'imgs'
    base.mkdir()
    shutil.copy(src, base / 'foo_word_0.png')
    list_file = tmp_path / 'list.txt'
    with open(list_file, 'w') as f:
        f.write('foo_word_0.png\n')
    return PretrainingHTRDataset(str(list_file), fixed_size=(32, 128), base_path=str(base))


class _FakeLoader:
    calls = []

    def __init__(self, dataset, batch_size, shuffle=True, num_workers=0, pin_memory=False):
        self.dataset = dataset
        self.batch_size = batch_size
        _FakeLoader.calls.append(dataset)

    def __iter__(self):
        sample = self.dataset[0]
        if isinstance(sample, tuple) and len(sample) == 3:
            img, t, idx = sample
            yield img.unsqueeze(0), [t], idx
        else:
            img, t = sample
            yield img.unsqueeze(0), [t]

    def __len__(self):
        return 1


def _tiny_refine_setup(tmp_path):
    ds = _tiny_dataset()
    ds.config.n_aligned = 1
    ds.data = ds.data[:1]
    ds.data = [(ds.data[0][0], 'word')]
    ds.transcriptions = ['word']
    ds.aligned = ds.aligned[:1]
    ds.is_in_dict = ds.is_in_dict[:1]

    arch = SimpleNamespace(
        cnn_cfg=[[1, 16]],
        head_type='cnn',
        rnn_type='gru',
        rnn_layers=1,
        rnn_hidden_size=16,
        flattening='maxpool',
        stn=False,
        feat_dim=None,
    )
    from tests import train_by_length as tbl
    c2i, _ = load_vocab()
    net = HTRNet(arch, nclasses=len(c2i) + 1)
    return ds, net


def test_refine_syn_ratio_zero(monkeypatch, tmp_path):
    from tests import train_by_length as tbl
    pre_ds = _make_pre_ds(tmp_path)
    ds, net = _tiny_refine_setup(tmp_path)

    _FakeLoader.calls.clear()
    monkeypatch.setattr(tbl, 'DataLoader', _FakeLoader)
    monkeypatch.setattr(tbl, '_evaluate_cer', lambda *a, **k: 0.0)

    tbl.refine_visual_model(ds, net, num_epochs=1, batch_size=2, lr=1e-3, max_length=10,
                           pretrain_ds=pre_ds, syn_batch_ratio=0)

    from torch.utils.data import Subset
    gt_calls = sum(isinstance(d, Subset) for d in _FakeLoader.calls)
    pre_calls = sum(d is pre_ds for d in _FakeLoader.calls)
    assert gt_calls == 1 and pre_calls == 0


def test_refine_syn_ratio_one(monkeypatch, tmp_path):
    from tests import train_by_length as tbl
    pre_ds = _make_pre_ds(tmp_path)
    ds, net = _tiny_refine_setup(tmp_path)

    _FakeLoader.calls.clear()
    monkeypatch.setattr(tbl, 'DataLoader', _FakeLoader)
    monkeypatch.setattr(tbl, '_evaluate_cer', lambda *a, **k: 0.0)

    tbl.refine_visual_model(ds, net, num_epochs=1, batch_size=2, lr=1e-3, max_length=10,
                           pretrain_ds=pre_ds, syn_batch_ratio=1)

    from torch.utils.data import Subset
    gt_calls = sum(isinstance(d, Subset) for d in _FakeLoader.calls)
    pre_calls = sum(d is pre_ds for d in _FakeLoader.calls)
    assert gt_calls == 0 and pre_calls == 1


def test_ctc_loss_fn():
    torch.manual_seed(0)
    logits = torch.randn(8, 2, 5, requires_grad=True)
    targets = torch.randint(1, 5, (4,), dtype=torch.int32)
    inp_lens = torch.tensor([8, 8], dtype=torch.int32)
    tgt_lens = torch.tensor([2, 2], dtype=torch.int32)

    ref = F.ctc_loss(
        F.log_softmax(logits, dim=2),
        targets,
        inp_lens,
        tgt_lens,
        reduction="mean",
        zero_infinity=True,
    )
    val = _ctc_loss_fn(logits, targets, inp_lens, tgt_lens)
    assert torch.allclose(val, ref)


def test_dataset_image_values(tmp_path):
    """Test that dataset image values are scaled between 0 and 1."""
    # Test HTRDataset
    cfg = SimpleNamespace(k_external_words=5, n_aligned=0, word_emb_dim=8)
    base = 'htr_base/data/GW/processed_words'
    dataset = HTRDataset(base, subset='train', fixed_size=(32, 128), transforms=aug_transforms, config=cfg)
    img, _, _ = dataset[0]
    assert img.min() >= 0.0, f"HTRDataset min value is {img.min()}"
    assert img.max() <= 1.0, f"HTRDataset max value is {img.max()}"

    # Test PretrainingHTRDataset
    src = Path('htr_base/data/GW/processed_words/train/train_000000.png')
    base = tmp_path / 'imgs'
    base.mkdir()
    shutil.copy(src, base / 'foo_word_0.png')

    list_file = tmp_path / 'list.txt'
    with open(list_file, 'w') as f:
        f.write('foo_word_0.png\n')

    pretrain_ds = PretrainingHTRDataset(
        str(list_file), fixed_size=(32, 128), base_path=str(base), transforms=aug_transforms
    )
    img, _ = pretrain_ds[0]
    assert img.min() >= 0.0, f"PretrainingHTRDataset min value is {img.min()}"
    assert img.max() <= 1.0, f"PretrainingHTRDataset max value is {img.max()}"


def test_attentive_pool_shape():
    pool = AttentivePool(4, 6)
    x = torch.randn(2, 4, 8, 8)
    out = pool(x)
    assert out.shape == (2, 6)


def test_htrnet_attentive_pool(tmp_path):
    arch = SimpleNamespace(
        cnn_cfg=[[1, 8]],
        head_type="cnn",
        rnn_type="gru",
        rnn_layers=1,
        rnn_hidden_size=8,
        flattening="maxpool",
        stn=False,
        feat_dim=10,
        feat_pool="attn",
    )
    net = HTRNet(arch, nclasses=3)
    imgs = torch.randn(2, 1, 32, 64)
    logits, feats = net(imgs, return_feats=True)
    assert feats.shape == (2, 10)


def test_htrnet_feat_pool_invalid():
    arch = SimpleNamespace(
        cnn_cfg=[[1, 8]],
        head_type="cnn",
        rnn_type="gru",
        rnn_layers=1,
        rnn_hidden_size=8,
        flattening="maxpool",
        stn=False,
        feat_dim=10,
        feat_pool="bad",
    )
    with pytest.raises(ValueError):
        HTRNet(arch, nclasses=3)

def test_google_drive_upload_import():
    pytest.importorskip("googleapiclient")
    from google_drive_upload import upload_to_drive
