import sys
from pathlib import Path
from types import SimpleNamespace

import torch

# Ensure repo root in Python path
proj_root = Path(__file__).resolve().parents[1]
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from htr_base.models import HTRNet
from htr_base.utils.preprocessing import load_image, preprocess


def _load_batch(n=2, size=(32, 64)):
    """Load *n* images from the sample dataset and return a tensor batch."""
    base = proj_root / "htr_base" / "data" / "GW" / "processed_words" / "train"
    with open(base / "gt.txt", "r") as f:
        lines = [line.strip().split()[0] for line in f][:n]
    imgs = []
    for name in lines:
        img = load_image(str(base / f"{name}.png"))
        img = preprocess(img, size)
        imgs.append(torch.tensor(img).float().unsqueeze(0))
    return torch.stack(imgs)


def test_transformer_head_dataset():
    batch = _load_batch()
    arch = SimpleNamespace(
        cnn_cfg=[[1, 8]],
        head_type="transf",
        transf_d_model=16,
        transf_nhead=2,
        transf_layers=1,
        transf_dim_ff=32,
        flattening="maxpool",
        stn=False,
        feat_dim=None,
    )
    model = HTRNet(arch, nclasses=7)
    out = model(batch, return_feats=False)
    if isinstance(out, (list, tuple)):
        out = out[0]
    # basic shape checks
    assert out.shape[1] == batch.size(0)
    assert out.shape[2] == 7
    # logits should be finite and time dimension positive
    assert out.shape[0] > 0
    assert torch.isfinite(out).all()
