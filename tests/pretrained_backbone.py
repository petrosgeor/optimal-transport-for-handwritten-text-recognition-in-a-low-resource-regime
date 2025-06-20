from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from htr_base.models import HTRNet
from htr_base.utils.htr_dataset import HTRDataset
from htr_base.utils.metrics import CER
from alignment.ctc_utils import greedy_ctc_decode

from tests.train_by_length import ARCHITECTURE_CONFIG, DATASET_FIXED_SIZE, _build_vocab_dicts


def test_pretrained_backbone_cer():
    weights = Path('htr_base/saved_models/pretrained_backbone.pt')
    assert weights.exists(), 'missing pretrained weights'

    arch_cfg = SimpleNamespace(**ARCHITECTURE_CONFIG)
    dataset = HTRDataset(
        'htr_base/data/GW/processed_words',
        subset='test',
        fixed_size=DATASET_FIXED_SIZE,
        transforms=None,
    )

    c2i, i2c = _build_vocab_dicts(dataset)

    model = HTRNet(arch_cfg, nclasses=len(c2i) + 1)
    state = torch.load(weights, map_location='cpu')
    model.load_state_dict(state)
    model.eval()

    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    cer = CER()
    with torch.no_grad():
        for imgs, transcrs, _ in loader:
            logits = model(imgs)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            preds = greedy_ctc_decode(logits, i2c)
            for p, t in zip(preds, transcrs):
                cer.update(p.strip(), t.strip())

    assert 0.0 <= cer.score() <= 1.0

