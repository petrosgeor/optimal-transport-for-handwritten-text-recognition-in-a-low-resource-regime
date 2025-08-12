"""Minimal test suite for htr_new."""

from pathlib import Path
import sys
from omegaconf import OmegaConf
import torch
import itertools

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment.ctc_utils import ctc_target_probability
from htr_base.utils.htr_dataset import HTRDataset, PretrainingHTRDataset
from alignment.alignment_utilities import compute_ot_responsibilities


def test_trainer_config_has_no_prior_weight():
    """Verify trainer configuration no longer includes ``prior_weight``."""
    cfg = OmegaConf.load("alignment/alignment_configs/trainer_config.yaml")
    assert "prior_weight" not in cfg



## log_round_metrics removed in EM-only workflow


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
    words = ["aa", "ab"]
    emb = HTRDataset.find_word_embeddings(dataset, words, n_components=2)
    dataset.unique_word_embeddings = emb

    assert hasattr(dataset, "unique_word_embeddings")
    # With Landmark MDS, dimensionality is p = min(n_components, m-1)
    # where m is the number of landmarks (here m = len(words)).
    m = min(len(words), 1000)
    expected_p = min(2, max(1, m - 1))
    assert dataset.unique_word_embeddings.shape == (len(words), expected_p)


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
    )

    assert sorted(ds.transcriptions) == [
        "a",
        "ab",
        "abc",
        "abcdef",
        "abcdefg",
    ]


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


def test_alternating_refinement_calls_cer(monkeypatch, tmp_path):
    """compute_cer is invoked appropriately in EM-only alternating loop."""

    from types import SimpleNamespace
    from alignment import trainer

    class FakeDataset(torch.utils.data.Dataset):
        def __init__(
            self,
            basefolder="dummy",
            subset="train_val",
            fixed_size=(1, 1),
            character_classes=None,
            config=None,
            two_views=False,
        ):
            self.basefolder = basefolder
            self.subset = subset
            self.fixed_size = fixed_size
            self.character_classes = character_classes or []
            self.config = config or SimpleNamespace()
            self.two_views = two_views
            self.aligned = torch.tensor([-1, 0])
            self.unique_words = ["x", "y"]
            self.imgs = [torch.zeros(1, 1, 1), torch.zeros(1, 1, 1)]
            self.transcriptions = ["x", "y"]

        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, idx):
            return self.imgs[idx], self.transcriptions[idx], self.aligned[idx]

    class FakeProjector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return x

    calls = []

    def fake_compute_cer(ds, model, **kwargs):
        calls.append(ds)
        return 0.0

    monkeypatch.setattr(trainer, "compute_cer", fake_compute_cer)
    monkeypatch.setattr(trainer, "refine_visual_backbone", lambda *a, **k: None)
    monkeypatch.setattr(trainer, "train_projector", lambda *a, **k: None)
    monkeypatch.setattr(trainer, "maybe_load_backbone", lambda *a, **k: None)
    monkeypatch.setattr(trainer, "HTRDataset", FakeDataset)
    monkeypatch.setattr(trainer.cfg, "device", "cpu")
    monkeypatch.setattr(trainer.cfg, "align_device", "cpu")

    ds = FakeDataset()
    backbone = DummyBackbone()
    proj = FakeProjector()

    trainer.alternating_refinement(ds, backbone, [proj], rounds=1)

    subsets = {getattr(getattr(d, "dataset", d), "subset", None) for d in calls}
    assert {"train_val", "test"} == subsets
    # No pseudo-labelling metrics logging in EM-only loop


## Pseudo-labelling specific tests removed


def test_compute_ot_responsibilities_shapes_and_mass():
    """Responsibilities have shape (N,V), are non-negative and row-stochastic."""
    import numpy as _np

    class TinyDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.imgs = [
                torch.zeros(1, 2, 2),
                torch.ones(1, 2, 2),
                torch.full((1, 2, 2), 2.0),
            ]
            self.transcriptions = ["a", "b", "c"]
            self.aligned = torch.full((3,), -1, dtype=torch.int32)
            self.unique_words = ["w0", "w1", "w2", "w3"]
            # 4 words in a 2-D embedding space
            self.unique_word_embeddings = torch.tensor(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32
            )
            self.unique_word_probs = [0.4, 0.3, 0.2, 0.1]

        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, idx):
            return self.imgs[idx], self.transcriptions[idx], self.aligned[idx]

    class TinyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.phoc_head = None

        def forward(self, x, *, return_feats=True):
            B = x.size(0)
            # Build 2-D descriptors proportional to mean intensity
            means = x.view(B, -1).mean(dim=1, keepdim=True)
            feats = torch.cat([means, 1.0 + means], dim=1)
            logits = torch.zeros(1, B, 4)
            return logits, feats

    class IdProjector(torch.nn.Module):
        def forward(self, z):
            return z

    ds = TinyDataset()
    bb = TinyBackbone()
    proj = IdProjector()

    R = compute_ot_responsibilities(ds, bb, [proj], batch_size=2, device="cpu", reg=0.1)
    assert isinstance(R, torch.Tensor)
    assert R.shape == (len(ds), len(ds.unique_words))
    assert torch.all(R >= 0)
    row_sums = R.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)

    Rk = compute_ot_responsibilities(ds, bb, [proj], batch_size=2, device="cpu", reg=0.1, topk=2)
    nnz = (Rk > 0).sum(dim=1)
    assert torch.all(nnz <= 2)
    row_sums_k = Rk.sum(dim=1)
    assert torch.allclose(row_sums_k, torch.ones_like(row_sums_k), atol=1e-4)



def test_harvest_restores_backbone_mode():
    """harvest_backbone_features preserves and restores module train/eval mode."""
    from alignment.alignment_utilities import harvest_backbone_features

    class TinyDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.imgs = [torch.zeros(1, 2, 2), torch.ones(1, 2, 2)]
            self.transcriptions = ["a", "b"]
            self.aligned = torch.tensor([-1, -1])

        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, idx):
            return self.imgs[idx], self.transcriptions[idx], self.aligned[idx]

    class TinyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.phoc_head = None

        def forward(self, x, *, return_feats=True):
            B = x.size(0)
            feats = torch.randn(B, 2)
            logits = torch.zeros(1, B, 4)
            return logits, feats

    ds = TinyDataset()
    bb = TinyBackbone().train(True)
    # Run harvest and ensure original training mode is restored
    harvest_backbone_features(ds, bb, batch_size=2, device=torch.device("cpu"))
    assert bb.training is True

    # Now test starting from eval mode
    bb2 = TinyBackbone().eval()
    harvest_backbone_features(ds, bb2, batch_size=2, device=torch.device("cpu"))
    assert bb2.training is False


def test_select_seed_indices_random_distinct():
    """Random seeds select distinct words up to n_aligned."""

    ds = HTRDataset.__new__(HTRDataset)
    ds.transcriptions = [
        "longestword",
        "foo",
        "bar",
        "longestword",
        "barbar",
        "baz",
        "foo",
        "qux",
        "longer",
    ]
    ds.n_aligned = 3

    # Make randomness deterministic for the test
    import random as _r
    _state = _r.getstate()
    _r.seed(0)
    try:
        indices = HTRDataset._select_seed_indices(ds)
    finally:
        _r.setstate(_state)

    words = [ds.transcriptions[i] for i in indices]
    assert len(indices) == 3
    assert len(set(words)) == 3
    # All words must come from the dataset
    for w in words:
        assert w in ds.transcriptions


def test_select_seed_indices_limits():
    """Fewer unique words than ``n_aligned`` yields a shorter list."""

    ds = HTRDataset.__new__(HTRDataset)
    ds.transcriptions = ["a", "b", "a"]
    ds.n_aligned = 5

    indices = HTRDataset._select_seed_indices(ds)

    assert len(indices) == 2


def test_final_backbone_refinement_runs(monkeypatch):
    """Call ``refine_visual_backbone`` once all samples are aligned."""

    from types import SimpleNamespace
    import alignment.trainer as trainer

    flag = {"called": False}

    def _fake_refine(dataset, backbone, num_epochs, **kwargs):
        flag["called"] = True

    # Avoid file access and heavy initialisation inside the trainer.
    monkeypatch.setattr(trainer, "refine_visual_backbone", _fake_refine)
    monkeypatch.setattr(trainer, "maybe_load_backbone", lambda *a, **k: None)

    class TinyModule:
        def to(self, device):
            return self

        def parameters(self):
            return []

    dataset = SimpleNamespace(
        basefolder=".",
        fixed_size=(1, 1),
        character_classes=[],
        config=None,
        aligned=torch.tensor([0]),
    )

    # Prevent construction of a real HTRDataset for the test split.
    monkeypatch.setattr(trainer, "HTRDataset", lambda *a, **k: dataset)

    backbone = TinyModule()
    projector = TinyModule()

    trainer.alternating_refinement(
        dataset,
        backbone,
        [projector],
        rounds=0,
        backbone_epochs=1,
    )

    assert flag["called"]


def test_use_wordfreq_probs(tmp_path):
    """wordfreq prior replaces empirical counts and normalises to 1."""

    from types import SimpleNamespace

    base = tmp_path / "data"
    train = base / "train"
    train.mkdir(parents=True)
    with open(train / "gt.txt", "w") as f:
        f.write("0 alpha\n")
        f.write("1 beta\n")
        f.write("2 alpha\n")
        f.write("3 zzzzzzzz\n")

    cfg = SimpleNamespace(n_aligned=0, word_emb_dim=2, use_wordfreq_probs=False)
    ds = HTRDataset(str(base), subset="train", fixed_size=(1, 1), config=cfg)
    assert ds.unique_word_probs == [0.5, 0.25, 0.25]

    cfg_true = SimpleNamespace(n_aligned=0, word_emb_dim=2, use_wordfreq_probs=True)
    ds_wf = HTRDataset(str(base), subset="train", fixed_size=(1, 1), config=cfg_true)
    probs = ds_wf.unique_word_probs
    assert abs(sum(probs) - 1.0) < 1e-6
    assert all(p > 0 for p in probs)
    assert probs != [0.5, 0.25, 0.25]
