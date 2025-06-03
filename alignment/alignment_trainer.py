import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os

from htr_base.utils.htr_dataset import HTRDataset
from htr_base.models import HTRNet
from losses import ProjectionLoss
import ctc_utils





def refine_visual_backbone(
        dataset: HTRDataset,
        backbone: HTRNet,
        num_epochs: int,
):
    print(f'starting backbone refinement for {num_epochs} epochs')
    device_local = next(backbone.parameters()).device

    backbone.train().to(device_local)
    for p in backbone.parameters():
        p.required_grad = True

    dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=True)
    ctc_loss_fn = lambda y, t, ly, lt: torch.nn.CTCLoss(reduction='mean', zero_infinity=True, )

    character_classes = np.load(os.path.join(dataset.basefolder), 'classes.npy')
    c2i = {c: (i + 1) for i, c in enumerate(character_classes)}
    i2c = {(i + 1): c for i, c in enumerate(character_classes)}


    optimizer = optim.AdamW(backbone.parameters())
    # TODO specify learning rate

    for epoch in range(num_epochs):
        epoch_total_loss = 0
        num_batches = 0


    






