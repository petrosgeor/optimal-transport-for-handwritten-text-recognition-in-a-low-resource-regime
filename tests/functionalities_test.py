"""Minimal test suite for htr_new."""

from pathlib import Path
import sys
from omegaconf import OmegaConf
import torch
import itertools

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment.ctc_utils import ctc_target_probability
from alignment.trainer import _shuffle_batch
from htr_base.utils.htr_dataset import HTRDataset, PretrainingHTRDataset


def test_trainer_config_has_no_prior_weight():
    """Verify trainer configuration no longer includes ``prior_weight``."""
    cfg = OmegaConf.load("alignment/alignment_configs/trainer_config.yaml")
    assert "prior_weight" not in cfg


def test_shuffle_batch():
    """_shuffle_batch keeps pairs together and randomises order."""
    torch.manual_seed(0)
    imgs = torch.arange(6).view(3, 2)
    words = ["a", "b", "c"]
    shuffled_imgs, shuffled_words = _shuffle_batch(imgs.clone(), list(words))

    expected_pairs = [
        ((4, 5), "c"),
        ((0, 1), "a"),
        ((2, 3), "b"),
    ]
    observed_pairs = [
        (tuple(shuffled_imgs[i].tolist()), shuffled_words[i])
        for i in range(3)
    ]
    assert observed_pairs == expected_pairs
    assert not torch.equal(imgs, shuffled_imgs)


def test_ctc_target_probability():
    """Probability of a short target via dynamic programming."""
    logits = torch.tensor([
        [0.1, 0.7, 0.2],
        [0.6, 0.2, 0.2],
        [0.1, 0.8, 0.1],
    ])
    c2i = {"a": 1, "b": 2}
    prob = ctc_target_probability(logits, "a", c2i)

    log_probs = logits.log_softmax(dim=1).exp()
    brute = 0.0
    for path in itertools.product(range(3), repeat=3):
        p = 1.0
        prev = None
        collapsed = []
        for t, idx in enumerate(path):
            p *= log_probs[t, idx].item()
            if idx != prev and idx != 0:
                collapsed.append("a" if idx == 1 else "b")
            prev = idx
        if "".join(collapsed) == "a":
            brute += p
    assert abs(prob - brute) < 1e-6


def test_ctc_target_probability_longer():
    """Dynamic vs brute force probability for a longer string."""
    logits = torch.tensor([
        [0.6, 0.2, 0.2],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.6, 0.2, 0.2],
        [0.1, 0.8, 0.1],
        [0.6, 0.2, 0.2],
    ])
    c2i = {"a": 1, "b": 2}
    target = "aaba"
    prob = ctc_target_probability(logits, target, c2i)

    log_probs = logits.log_softmax(dim=1).exp()
    brute = 0.0
    for path in itertools.product(range(3), repeat=6):
        p = 1.0
        prev = None
        collapsed = []
        for t, idx in enumerate(path):
            p *= log_probs[t, idx].item()
            if idx != prev and idx != 0:
                collapsed.append("a" if idx == 1 else "b")
            prev = idx
        if "".join(collapsed) == target:
            brute += p
    assert abs(prob - brute) < 1e-6


class DummyHTRDataset(torch.utils.data.Dataset):
    """Minimal real dataset with alignment flags."""

    def __init__(self):
        self.unique_words = ["gt1", "gt2"]
        self.aligned = torch.tensor([0, 1], dtype=torch.int64)
        self.imgs = [torch.zeros(1, 2, 2), torch.ones(1, 2, 2)]
        self.trans = ["gt1", "gt2"]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx], self.trans[idx], self.aligned[idx]


class DummyPretrainDataset(torch.utils.data.Dataset):
    """Synthetic dataset used for backbone refinement."""

    def __init__(self):
        self.imgs = [torch.full((1, 2, 2), 2.0), torch.full((1, 2, 2), 3.0)]
        self.trans = ["syn1", "syn2"]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx], self.trans[idx]


class DummyBackbone(torch.nn.Module):
    """Record images fed during refinement."""

    def __init__(self):
        super().__init__()
        class FakeParam(torch.nn.Parameter):
            @property
            def device(self):
                return torch.device("cuda")

        self.param = FakeParam(torch.zeros(1, requires_grad=True))
        self.calls = []
        self.phoc_head = None

    def forward(self, x, *, return_feats=True):
        self.calls.append(x.detach().clone())
        B = x.size(0)
        logits = torch.zeros(1, B, 38, requires_grad=True)
        return logits, logits

    def parameters(self, recurse=True):
        yield self.param

    def to(self, device):
        return self


def test_refine_visual_backbone_syn_only(monkeypatch):
    """Synthetic-only batches when ``syn_batch_ratio=1``."""

    dataset = DummyHTRDataset()
    pre_ds = DummyPretrainDataset()
    backbone = DummyBackbone()

    from alignment import trainer
    refine_visual_backbone = trainer.refine_visual_backbone

    orig_to = torch.Tensor.to
    orig_full = torch.full
    orig_tensor = torch.tensor

    def fake_to(self, *args, **kwargs):
        if args and (args[0] == torch.device("cuda") or args[0] == "cuda"):
            return self
        if kwargs.get("device") in {torch.device("cuda"), "cuda"}:
            kwargs = {**kwargs, "device": torch.device("cpu")}
        return orig_to(self, *args, **kwargs)

    def fake_full(size, fill_value, **kwargs):
        if kwargs.get("device") in {torch.device("cuda"), "cuda"}:
            kwargs = {**kwargs, "device": torch.device("cpu")}
        return orig_full(size, fill_value, **kwargs)

    def fake_tensor(*args, **kwargs):
        if kwargs.get("device") in {torch.device("cuda"), "cuda"}:
            kwargs = {**kwargs, "device": torch.device("cpu")}
        return orig_tensor(*args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", fake_to, raising=False)
    monkeypatch.setattr(torch, "full", fake_full, raising=False)
    monkeypatch.setattr(torch, "tensor", fake_tensor, raising=False)
    monkeypatch.setattr(trainer, "plot_tsne_embeddings", lambda *a, **k: None)

    refine_visual_backbone(
        dataset,
        backbone,
        num_epochs=1,
        batch_size=2,
        pretrain_ds=pre_ds,
        syn_batch_ratio=1.0,
        enable_phoc=False,
        enable_contrastive=False,
    )

    for batch in backbone.calls:
        assert batch.min() >= 2


def test_word_frequencies():
    """Unique words and probabilities from transcriptions."""

    dataset = HTRDataset.__new__(HTRDataset)
    dataset.transcriptions = ["cat", "cat", "dog"]

    words, probs = HTRDataset.word_frequencies(dataset)

    assert set(words) == {"cat", "dog"}
    mapping = dict(zip(words, probs))
    assert abs(mapping["cat"] - 2 / 3) < 1e-6
    assert abs(mapping["dog"] - 1 / 3) < 1e-6


def test_unique_word_embeddings_attribute():
    """Word embeddings tensor is stored under ``unique_word_embeddings``."""

    dataset = HTRDataset.__new__(HTRDataset)
    dataset.word_emb_dim = 2
    emb = HTRDataset.find_word_embeddings(dataset, ["aa", "ab"], n_components=2)
    dataset.unique_word_embeddings = emb

    assert hasattr(dataset, "unique_word_embeddings")
    assert dataset.unique_word_embeddings.shape == (2, 2)


def test_pretraining_dataset_length_filter(tmp_path):
    """Entries outside the length range are discarded."""

    txt = tmp_path / "paths.txt"
    txt.write_text("\n".join([
        "a_ab_0.png",
        "b_abc_0.png",
        "c_abcdef_0.png",
        "d_abcdefg_0.png",
        "e_a_0.png",
    ]))

    ds = PretrainingHTRDataset(
        list_file=str(txt),
        base_path=str(tmp_path),
        n_random=10,
        min_length=2,
        max_length=6,
    )

    assert sorted(ds.transcriptions) == ["ab", "abc", "abcdef"]


def test_model_components_forward():
    """Instantiate small modules and run a dummy forward pass."""

    from htr_base.models import (
        Projector,
        BasicBlock,
        CNN,
        CTCtopC,
        CTCtopR,
        CTCtopB,
    )

    proj = Projector(4, 2)
    out = proj(torch.randn(3, 4))
    assert out.shape == (3, 2)

    block = BasicBlock(3, 3)
    out = block(torch.randn(1, 3, 8, 8))
    assert out.shape == (1, 3, 8, 8)

    cnn = CNN([[1, 32], "M", [1, 64]], flattening="maxpool")
    feats = cnn(torch.randn(1, 1, 32, 32))
    assert feats.shape == (1, 64, 1, 8)

    ctc_c = CTCtopC(64, 5)
    out = ctc_c(torch.randn(1, 64, 1, 10))
    assert out.shape == (10, 1, 5)

    ctc_r = CTCtopR(64, (32, 2), 5)
    out = ctc_r(torch.randn(1, 64, 1, 10))
    assert out.shape == (10, 1, 5)

    ctc_b = CTCtopB(64, (32, 2), 5)
    main, aux = ctc_b(torch.randn(1, 64, 1, 10))
    assert main.shape == aux.shape == (10, 1, 5)

