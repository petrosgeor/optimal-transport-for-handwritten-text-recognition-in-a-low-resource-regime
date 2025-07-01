from pathlib import Path
from types import SimpleNamespace
import sys
import torch
from omegaconf import OmegaConf

# Add repository root for imports
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from htr_base.utils.htr_dataset import HTRDataset
from htr_base.utils.vocab import load_vocab
from htr_base.models import HTRNet
from alignment.eval import compute_cer


def test_cer_pretrained():
    """Evaluate pretrained backbone on GW test set using greedy and beam decoding."""
    base = Path("htr_base/data/GW/processed_words")
    dataset = HTRDataset(basefolder=str(base), subset="test", fixed_size=(64, 256))

    c2i, _ = load_vocab()
    arch_cfg = OmegaConf.load("alignment/alignment_configs/pretraining_config.yaml")
    arch = SimpleNamespace(**arch_cfg["architecture"])
    model = HTRNet(arch, nclasses=len(c2i) + 1)

    weights = torch.load("htr_base/saved_models/pretrained_backbone.pt", map_location="cpu")
    model.load_state_dict(weights)

    cer_greedy = compute_cer(dataset, model, batch_size=2, device="cpu", decode="greedy")
    cer_beam = compute_cer(dataset, model, batch_size=2, device="cpu", decode="beam", beam_width=5)
    print("Greedy CER:", cer_greedy)
    print("Beam CER:", cer_beam)

    assert isinstance(cer_greedy, float) and cer_greedy >= 0.0
    assert isinstance(cer_beam, float) and cer_beam >= 0.0
