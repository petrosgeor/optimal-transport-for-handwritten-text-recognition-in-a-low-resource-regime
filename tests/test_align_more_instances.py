import os, sys
from pathlib import Path
import torch
from types import SimpleNamespace
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from alignment.alignment_utilities import align_more_instances
from htr_base.models import HTRNet, Projector

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.images = torch.rand(5, 1, 16, 16)
        self.transcriptions = ["a"] * 5
        self.external_word_embeddings = torch.randn(3, 4)
        self.aligned = torch.full((5,), -1, dtype=torch.int32)
        self.transforms = None

    def __getitem__(self, idx):
        return self.images[idx], self.transcriptions[idx], self.aligned[idx]

    def __len__(self):
        return len(self.images)

dataset = DummyDataset()

arch = SimpleNamespace(
    cnn_cfg=[[1, 8]],
    head_type="cnn",
    rnn_type="gru",
    rnn_layers=1,
    rnn_hidden_size=8,
    flattening="maxpool",
    stn=False,
    feat_dim=8,
)

backbone = HTRNet(arch, nclasses=3)
projector = Projector(arch.feat_dim, dataset.external_word_embeddings.size(1))

plan, proj_feats, moved_dist = align_more_instances(
    dataset, backbone, projector, batch_size=2, k=1
)

def test_no_nan_inf():
    assert not torch.isnan(plan).any()
    assert not torch.isnan(proj_feats).any()
    assert not torch.isnan(moved_dist).any()
    assert not torch.isinf(plan).any()
    assert not torch.isinf(proj_feats).any()
    assert not torch.isinf(moved_dist).any()
