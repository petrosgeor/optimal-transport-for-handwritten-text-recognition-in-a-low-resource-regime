import os
import sys
import tempfile
import torch
from pathlib import Path

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from alignment.alignment_utilities import plot_dataset_augmentations
from htr_base.utils.htr_dataset import HTRDataset

class DummyTransform:
    def __call__(self, *, image):
        return {'image': image + 1.0}

class DummyDataset(HTRDataset):
    def __init__(self, n=5, transforms=None):
        # Do not call super().__init__
        self.transforms = transforms
        self.n = n
        self.fixed_size = (10, 10)
        self.two_views = False
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        img = torch.full((1, 10, 10), float(idx))
        if self.transforms is not None:
            arr = self.transforms(image=img.squeeze(0).numpy())['image']
            img = torch.tensor(arr).unsqueeze(0)
        return img, "", -1

def test_plot_dataset_augmentations():
    ds = DummyDataset(transforms=DummyTransform())
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "fig.png")
        plot_dataset_augmentations(ds, out_path)
        assert os.path.isfile(out_path) and os.path.getsize(out_path) > 0
