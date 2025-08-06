"""Minimal test suite for htr_new."""

from pathlib import Path
import sys
from omegaconf import OmegaConf
import torch
import itertools

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment.ctc_utils import ctc_target_probability
from alignment.trainer import _shuffle_batch, log_round_metrics
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


def test_log_round_metrics(tmp_path):
    """Metrics are appended as TSV lines."""
    path = tmp_path / "round_metrics.txt"
    log_round_metrics(1, 2, 0.1234, str(path))
    log_round_metrics(2, 3, 0.5, str(path))
    content = path.read_text().splitlines()
    assert content == ["1\t2\t0.1234", "2\t3\t0.5000"]


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
    """compute_cer and log_round_metrics are invoked appropriately."""

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
    metrics = []

    def fake_compute_cer(ds, model, **kwargs):
        calls.append(ds)
        return 0.0

    def fake_align(ds, *a, **k):
        ds.aligned.fill_(0)

    def fake_log_round_metrics(r, c, cer, out_file):
        metrics.append((r, c, cer, out_file))

    monkeypatch.setattr(trainer, "compute_cer", fake_compute_cer)
    monkeypatch.setattr(trainer, "refine_visual_backbone", lambda *a, **k: None)
    monkeypatch.setattr(trainer, "train_projector", lambda *a, **k: None)
    monkeypatch.setattr(trainer, "align_more_instances", fake_align)
    monkeypatch.setattr(trainer, "maybe_load_backbone", lambda *a, **k: None)
    monkeypatch.setattr(trainer, "HTRDataset", FakeDataset)
    monkeypatch.setattr(trainer, "log_pseudo_labels", lambda *a, **k: None)
    monkeypatch.setattr(trainer, "log_round_metrics", fake_log_round_metrics)
    monkeypatch.setattr(trainer.cfg, "device", "cpu")
    monkeypatch.setattr(trainer.cfg, "align_device", "cpu")
    monkeypatch.setattr(trainer.cfg, "round_metrics_file", str(tmp_path / "m.txt"))

    ds = FakeDataset()
    backbone = DummyBackbone()
    proj = FakeProjector()

    trainer.alternating_refinement(ds, backbone, [proj], rounds=1)

    subsets = {getattr(getattr(d, "dataset", d), "subset", None) for d in calls}
    assert {"train_val", "test"} == subsets
    assert metrics == [(1, 1, 0.0, str(tmp_path / "m.txt"))]


def test_validate_pseudo_labels(monkeypatch):
    """Samples with large edit distance are un-aligned."""

    from alignment import alignment_utilities as au

    ds = DummyHTRDataset()
    ds.word_emb_dim = 2
    ds.unique_word_embeddings = torch.zeros((2, 2))
    backbone = DummyBackbone()
    proj = torch.nn.Identity()

    aligner = au.OTAligner(ds, backbone, [proj], batch_size=2, device="cpu")

    def fake_decode(logits, i2c, **kwargs):
        return ["gt1", "oops"]

    monkeypatch.setattr(au, "greedy_ctc_decode", fake_decode)

    removed = aligner.validate_pseudo_labels(edit_threshold=3, batch_size=2)

    assert removed == 1
    assert ds.aligned.tolist() == [0, -1]


def test_align_more_instances_gated_validation(monkeypatch):
    """``validate_pseudo_labels`` runs only after ``start_iteration``."""

    from alignment import alignment_utilities as au

    ds = DummyHTRDataset()
    ds.word_emb_dim = 2
    ds.unique_word_embeddings = torch.zeros((2, 2))
    backbone = DummyBackbone()
    proj = torch.nn.Identity()

    # patch align and validation to track invocations
    monkeypatch.setattr(au.OTAligner, "align", lambda self: (torch.empty(0), torch.empty(0), torch.empty(0)))

    calls = []

    def fake_validate(self, edit_threshold, batch_size, decode_cfg=None):
        calls.append(edit_threshold)

    monkeypatch.setattr(au.OTAligner, "validate_pseudo_labels", fake_validate)

    # set config and reset counter
    au._ALIGN_CALL_COUNT = 0
    if not hasattr(au.cfg, "pseudo_label_validation"):
        au.cfg.pseudo_label_validation = OmegaConf.create({})
    au.cfg.pseudo_label_validation.enable = True
    au.cfg.pseudo_label_validation.edit_distance = 3
    au.cfg.pseudo_label_validation.start_iteration = 2

    au.align_more_instances(ds, backbone, [proj], batch_size=2, device="cpu")
    assert len(calls) == 0

    au.align_more_instances(ds, backbone, [proj], batch_size=2, device="cpu")
    assert len(calls) == 1




def test_select_seed_indices_longest_distinct():
    """Longest distinct words are selected deterministically."""

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
    ds.is_in_dict = torch.ones(len(ds.transcriptions), dtype=torch.int32)

    indices = HTRDataset._select_seed_indices(ds)
    words = [ds.transcriptions[i] for i in indices]

    assert indices == [0, 4, 8]
    assert words == ["longestword", "barbar", "longer"]


def test_select_seed_indices_limits():
    """Fewer unique words than ``n_aligned`` yields a shorter list."""

    ds = HTRDataset.__new__(HTRDataset)
    ds.transcriptions = ["a", "b", "a"]
    ds.n_aligned = 5
    ds.is_in_dict = torch.ones(len(ds.transcriptions), dtype=torch.int32)

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


def test_lexicon_top_k_prunes_words(tmp_path):
    """Prunes vocabulary and updates flags when ``lexicon_top_k`` > 0."""

    from types import SimpleNamespace

    base = tmp_path / "data"
    train = base / "train"
    train.mkdir(parents=True)
    with open(train / "gt.txt", "w") as f:
        f.write("0 alpha\n")
        f.write("1 beta\n")
        f.write("2 alpha\n")
        f.write("3 gamma\n")

    cfg = SimpleNamespace(
        n_aligned=0,
        word_emb_dim=2,
        use_wordfreq_probs=False,
        lexicon_top_k=2,
    )

    ds = HTRDataset(str(base), subset="train", fixed_size=(1, 1), config=cfg)
    assert ds.unique_words == ["alpha", "beta"]
    assert len(ds.unique_word_probs) == 2
    assert abs(sum(ds.unique_word_probs) - 1.0) < 1e-6
    assert ds.is_in_dict.tolist() == [1, 1, 1, 0]

    ds.n_aligned = 3
    idx = ds._select_seed_indices()
    assert idx == [0, 1]

