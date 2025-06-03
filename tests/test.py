import os
from pathlib import Path
from types import SimpleNamespace
import sys

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import torch

from htr_base.utils.htr_dataset import HTRDataset
from htr_base.models import HTRNet


def test_forward_pass():
    """Run a tiny forward pass through the network and print output shapes (expecting 3 outputs)."""
    # Resolve dataset folder relative to project root
    root = Path(__file__).resolve().parents[1]
    basefolder = root / 'htr_base' / 'data' / 'GW' / 'processed_words'
    assert basefolder.exists(), f"Dataset folder not found: {basefolder}"

    # Instantiate dataset and network
    dataset = HTRDataset(str(basefolder), 'train', fixed_size=(128, 256))
    arch_cfg = SimpleNamespace(
        cnn_cfg=[[2, 64], 'M', [3, 128], 'M', [2, 256]],
        head_type='both',
        rnn_type='lstm',
        rnn_layers=3,
        rnn_hidden_size=256,
        flattening='maxpool',
        stn=False,
        fit_dim=512,
        feat_dim=512   
    )
    nclasses = len(dataset.character_classes) + 1  # +1 for the CTC blank token
    net = HTRNet(arch_cfg, nclasses)

    imgs = torch.stack([dataset[i][0] for i in range(2)])  # (B, C, H, W)
    out = net(imgs)

    if isinstance(out, tuple):
        print(f"Network produced {len(out)} output tensors:")
        for i, o in enumerate(out):
            print(f"  out[{i}] shape → {tuple(o.shape)}")
        assert len(out) == 3, f"Expected 3 outputs, got {len(out)}!"
        main_out, aux_out, feats_out = out
        assert main_out.shape[1] == imgs.size(0)
        assert main_out.shape[2] == nclasses
    else:
        print("Network produced a single output tensor:")
        print(f"  out shape → {tuple(out.shape)}")
        assert False, "Expected the network to produce 3 outputs!"


def print_external_word_embeddings_shape():
    """Print the shape of the external_word_embeddings tensor from the dataset."""
    root = Path(__file__).resolve().parents[1]
    basefolder = root / 'htr_base' / 'data' / 'GW' / 'processed_words'
    assert basefolder.exists(), f"Dataset folder not found: {basefolder}"

    class DummyConfig:
        k_external_words = 100  
    
    dataset = HTRDataset(str(basefolder), 'train', fixed_size=(128, 256), config=DummyConfig())
    embeddings = dataset.external_word_embeddings
    probs = dataset.external_word_probs
    print(f"external_word_embeddings shape and probs: {embeddings.shape, probs}")

if __name__ == "__main__":
    test_forward_pass()
    print_external_word_embeddings_shape()
