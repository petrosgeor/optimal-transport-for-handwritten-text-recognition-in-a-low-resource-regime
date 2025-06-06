import sys
from pathlib import Path
import torch
from torch.utils.data import Dataset

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from alignment.alignment_utilities import harvest_backbone_features

class DummyDataset(Dataset):
    def __init__(self):
        self.transforms = None
        self.aligned = torch.tensor([0, -1, 1], dtype=torch.int32)
        self.data = torch.randn(3, 1, 8, 8)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], "", self.aligned[idx]

class DummyBackbone(torch.nn.Module):
    def __init__(self, feat_dim=5):
        super().__init__()
        self.feat_dim = feat_dim
    def forward(self, x, *, return_feats=True):
        feats = torch.ones(x.size(0), self.feat_dim)
        return (None, feats) if return_feats else None

def test_harvest_shapes():
    ds = DummyDataset()
    backbone = DummyBackbone()
    feats, aligns = harvest_backbone_features(ds, backbone, batch_size=2, device="cpu")
    assert feats.shape == (len(ds), backbone.feat_dim)
    assert torch.equal(aligns, ds.aligned)
