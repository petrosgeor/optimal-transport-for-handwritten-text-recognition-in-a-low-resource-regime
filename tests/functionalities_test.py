"""Minimal test suite for htr_new."""

from pathlib import Path
import sys
from collections import Counter
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
    dataset.word_prob_mode = "empirical"
    dataset.transcriptions = ["cat", "cat", "dog"]

    words, probs = dataset.word_frequencies()

    assert set(words) == {"cat", "dog"}
    mapping = dict(zip(words, probs))
    assert abs(mapping["cat"] - 2 / 3) < 1e-6
    assert abs(mapping["dog"] - 1 / 3) < 1e-6


def test_dataset_word_prob_mode(tmp_path):
    """unique_word_probs depend on ``word_prob_mode`` and sum to 1."""

    base = tmp_path
    (base / "train").mkdir()
    (base / "train" / "gt.txt").write_text("\n".join([
        "img1 cat",
        "img2 dog",
        "img3 cat",
    ]))

    ds_emp = HTRDataset(basefolder=str(base), subset="train", fixed_size=(1, 1), word_prob_mode="empirical")
    ds_wf = HTRDataset(basefolder=str(base), subset="train", fixed_size=(1, 1), word_prob_mode="wordfreq")

    assert abs(sum(ds_emp.unique_word_probs) - 1.0) < 1e-6
    assert abs(sum(ds_wf.unique_word_probs) - 1.0) < 1e-6

    map_emp = dict(zip(ds_emp.unique_words, ds_emp.unique_word_probs))
    map_wf = dict(zip(ds_wf.unique_words, ds_wf.unique_word_probs))

    counts = Counter(["cat", "dog", "cat"])
    expected_emp = [counts[w] / 3 for w in sorted(counts)]
    assert [map_emp[k] for k in sorted(map_emp)] == expected_emp
    assert all(p > 0 for p in ds_wf.unique_word_probs)


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


def test_alternating_refinement_calls_cer(monkeypatch):
    """compute_cer is invoked for train and test datasets."""

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
            word_prob_mode="empirical",
        ):
            self.basefolder = basefolder
            self.subset = subset
            self.fixed_size = fixed_size
            self.character_classes = character_classes or []
            self.config = config or SimpleNamespace()
            self.two_views = two_views
            self.word_prob_mode = "empirical"
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

    def fake_align(ds, *a, **k):
        ds.aligned.fill_(0)

    monkeypatch.setattr(trainer, "compute_cer", fake_compute_cer)
    monkeypatch.setattr(trainer, "refine_visual_backbone", lambda *a, **k: None)
    monkeypatch.setattr(trainer, "train_projector", lambda *a, **k: None)
    monkeypatch.setattr(trainer, "align_more_instances", fake_align)
    monkeypatch.setattr(trainer, "maybe_load_backbone", lambda *a, **k: None)
    monkeypatch.setattr(trainer, "HTRDataset", FakeDataset)
    monkeypatch.setattr(trainer, "log_pseudo_labels", lambda *a, **k: None)
    monkeypatch.setattr(trainer.cfg, "device", "cpu")
    monkeypatch.setattr(trainer.cfg, "align_device", "cpu")

    ds = FakeDataset()
    backbone = DummyBackbone()
    proj = FakeProjector()

    trainer.alternating_refinement(ds, backbone, [proj], rounds=1)

    subsets = {getattr(getattr(d, "dataset", d), "subset", None) for d in calls}
    assert {"train_val", "test"} == subsets


def test_align_closest_per_word():
    """Each word receives one pseudo-label."""

    from alignment import alignment_utilities as au

    class TinyDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.unique_words = [f"w{i}" for i in range(5)]
            self.word_emb_dim = 2
            self.unique_word_embeddings = torch.stack([
                torch.tensor([float(i), float(i)]) for i in range(5)
            ])
            self.unique_word_probs = [1 / 5] * 5
            self.aligned = torch.full((5,), -1, dtype=torch.int32)
            self.real_word_indices = torch.arange(len(self.unique_words))
            self.imgs = [torch.full((1, 2, 2), float(i)) for i in range(5)]
            self.transcriptions = ["" for _ in range(5)]
            self.transforms = None

        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, idx):
            return self.imgs[idx], self.transcriptions[idx], self.aligned[idx]

    class SimpleBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.phoc_head = None

        def forward(self, x, *, return_feats=True):
            B = x.size(0)
            feats = x.mean(dim=(2, 3)).repeat(1, 2)
            return torch.zeros(B, 1), feats

    ds = TinyDataset()
    backbone = SimpleBackbone()
    proj = torch.nn.Identity()
    proj.output_dim = 2

    au._ALIGN_CALL_COUNT = 0
    if not hasattr(au.cfg, "pseudo_label_validation"):
        au.cfg.pseudo_label_validation = OmegaConf.create({})
    au.cfg.pseudo_label_validation.enable = False

    au.align_more_instances(
        ds,
        backbone,
        [proj],
        batch_size=3,
        device="cpu",
    )

    assigned = ds.aligned[ds.aligned != -1]
    assert assigned.numel() == len(ds.unique_words)
    assert torch.unique(assigned).numel() == len(ds.unique_words)


def test_initial_pseudo_labels_logged(monkeypatch):
    """Initial aligned samples are logged as round 0."""

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
            word_prob_mode="empirical",
        ):
            self.basefolder = basefolder
            self.subset = subset
            self.fixed_size = fixed_size
            self.character_classes = character_classes or []
            self.config = config or SimpleNamespace()
            self.two_views = two_views
            self.word_prob_mode = word_prob_mode
            self.aligned = torch.tensor([1, -1])
            self.unique_words = ["a", "b"]
            self.imgs = [torch.zeros(1, 1, 1), torch.zeros(1, 1, 1)]
            self.transcriptions = ["a", "b"]

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

    logged = []

    def fake_log(indices, dataset, round_idx, **kwargs):
        logged.append((indices.clone(), round_idx))

    monkeypatch.setattr(trainer, "compute_cer", lambda *a, **k: 0.0)
    monkeypatch.setattr(trainer, "refine_visual_backbone", lambda *a, **k: None)
    monkeypatch.setattr(trainer, "train_projector", lambda *a, **k: None)
    monkeypatch.setattr(trainer, "align_more_instances", lambda ds, *a, **k: ds.aligned.fill_(0))
    monkeypatch.setattr(trainer, "maybe_load_backbone", lambda *a, **k: None)
    monkeypatch.setattr(trainer, "HTRDataset", FakeDataset)
    monkeypatch.setattr(trainer, "log_pseudo_labels", fake_log)
    monkeypatch.setattr(trainer.cfg, "device", "cpu")
    monkeypatch.setattr(trainer.cfg, "align_device", "cpu")

    ds = FakeDataset()
    backbone = DummyBackbone()
    proj = FakeProjector()

    trainer.alternating_refinement(ds, backbone, [proj], rounds=1)

    assert logged[0][1] == 0
    assert logged[0][0].tolist() == [0]


def test_parse_pseudo_files(tmp_path):
    """Latest predictions are returned and correctness is counted."""

    from tests.train_pseudo_labels import _parse_pseudo_files

    file1 = tmp_path / "pseudo_labels_round_0.txt"
    file1.write_text("\n".join([
        "0\tfoo\tfoo",
        "1\tbar\tbaz",
        "3\tbad\tgood",
    ]))

    file2 = tmp_path / "pseudo_labels_round_1.txt"
    file2.write_text("\n".join([
        "1\tbaz\tbaz",
        "2\tqux\tqux",
    ]))

    mapping, correct = _parse_pseudo_files(str(tmp_path))

    assert mapping == {0: "foo", 1: "baz", 2: "qux", 3: "bad"}
    assert correct == 3

    mapping, correct = _parse_pseudo_files(str(tmp_path), [1])
    assert mapping == {1: "baz", 2: "qux"}
    assert correct == 2

    mapping, correct = _parse_pseudo_files(str(tmp_path), exclude_false=True)
    assert mapping == {0: "foo", 1: "baz", 2: "qux"}
    assert correct == 3

