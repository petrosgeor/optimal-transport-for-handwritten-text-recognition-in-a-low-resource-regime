from pathlib import Path
import sys
from types import SimpleNamespace
import torch
import torch.nn as nn

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from htr_base.utils.htr_dataset import HTRDataset
from htr_base.models import HTRNet, Projector
from alignment.alignment_utilities import (
    align_more_instances,
    print_dataset_stats,
    plot_projector_tsne,
    predicted_char_distribution,
)
from alignment.alignment_trainer import tee_output
from alignment.alignment_trainer import (
    refine_visual_backbone,
    train_projector,
    _build_vocab_dicts,
)
from alignment.ctc_utils import encode_for_ctc
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


def test_predicted_char_distribution():
    logits = torch.randn(4, 2, 5)
    dist = predicted_char_distribution(logits)
    probs = logits.softmax(dim=2)
    expected = probs[:, :, 1:].mean(dim=(0, 1))
    assert dist.shape == (logits.size(2) - 1,)
    assert torch.allclose(dist, expected)
    blank_prob = probs[:, :, 0].mean()
    assert torch.allclose(dist.sum(), 1 - blank_prob)


def test_wasserstein_L2():
    p = torch.tensor([0.2, 0.3, 0.5])
    q = torch.tensor([0.1, 0.4, 0.5])
    expected = torch.sqrt(torch.mean((p - q) ** 2))
    from tests.train_by_length import wasserstein_L2

    assert torch.isclose(wasserstein_L2(p, q), expected)


def test_encode_for_ctc_vocab_size():
    cfg = SimpleNamespace(k_external_words=5, n_aligned=0, word_emb_dim=8)
    base = "htr_base/data/GW/processed_words"
    ds = HTRDataset(base, subset="test", fixed_size=(32, 128), transforms=None, config=cfg)
    ds.data = ds.data[:3]
    ds.transcriptions = ds.transcriptions[:3]
    ds.aligned = ds.aligned[:3]
    ds.is_in_dict = ds.is_in_dict[:3]

    c2i, _ = _build_vocab_dicts(ds)
    chars = "".join(sorted(ds.character_classes))
    txt = f" {chars} "
    targets, _ = encode_for_ctc([txt], c2i)
    # Include the CTC blank to count all possible labels
    targets = torch.cat([targets, torch.tensor([0])])
    assert len(c2i) + 1 == 37
    assert targets.unique().numel() == 37

