import torch
import torch.nn as nn


def lengths_from_transcriptions(t):
    """Return length indices for each transcription."""
    return torch.tensor([max(len(s) - 1, 0) for s in t])


class _DummyLengthNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        b = x.size(0)
        return torch.zeros(b, 20)


def build_htrnetlength():
    """Return a trivial network used in tests."""
    return _DummyLengthNet()
