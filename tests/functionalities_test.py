from pathlib import Path
import sys
from types import SimpleNamespace

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from htr_base.utils.htr_dataset import HTRDataset
from htr_base.models import HTRNet, Projector
from alignment.alignment_utilities import align_more_instances
from alignment.alignment_trainer import tee_output
from htr_base.utils.transforms import aug_transforms


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

    align_more_instances(ds, backbone, projector, batch_size=1, device='cpu', unbalanced=True, reg_m=0.1, k=1)
    assert (ds.aligned != -1).any(), "No samples were pseudo-labelled"
