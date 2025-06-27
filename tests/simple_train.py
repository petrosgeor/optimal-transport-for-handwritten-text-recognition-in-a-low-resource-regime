from __future__ import annotations
import random
from types import SimpleNamespace
from pathlib import Path
from typing import Dict, List
import sys

import torch
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


# Add project root to path for imports
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from htr_base.utils.htr_dataset import HTRDataset
from htr_base.models import HTRNet
from alignment.ctc_utils import encode_for_ctc, greedy_ctc_decode
from alignment.losses import _ctc_loss_fn
from htr_base.utils.metrics import CER
from htr_base.utils.vocab import load_vocab


DEFAULT_CONFIG = {
    "basefolder": "htr_base/data/GW/processed_words",
    "fixed_size": (64, 256),
    "n_examples": 200,
    "num_epochs": 1000,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "device": "cuda",
    "eval_interval": 10,
}

ARCH = {
    "cnn_cfg": [[2, 64], "M", [3, 128], "M", [2, 256]],
    "head_type": "both",
    "rnn_type": "gru",
    "rnn_layers": 3,
    "rnn_hidden_size": 256,
    "flattening": "maxpool",
    "stn": False,
    "feat_dim": None,
}




def _decode_random(net: HTRNet, dataset: HTRDataset, i2c: Dict[int, str], device) -> None:
    net.eval()
    with torch.no_grad():
        idx = random.sample(range(len(dataset)), min(10, len(dataset)))
        for i in idx:
            img, txt, _ = dataset[i]
            logits = net(img.unsqueeze(0).to(device), return_feats=False)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            pred = greedy_ctc_decode(logits, i2c)[0]
            print(f"GT: '{txt.strip()}' | PR: '{pred}'")
    net.train()


def _evaluate(net: HTRNet, loader: DataLoader, i2c: Dict[int, str], device) -> float:
    net.eval()
    metric = CER()
    with torch.no_grad():
        for imgs, txts, _ in loader:
            imgs = imgs.to(device)
            logits = net(imgs, return_feats=False)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            preds = greedy_ctc_decode(logits, i2c)
            for p, t in zip(preds, txts):
                metric.update(p.strip(), t.strip())
    net.train()
    score = metric.score()
    print(f"[Eval] CER: {score:.4f}")
    return score


def main(cfg: dict | None = None) -> None:
    if cfg is None:
        cfg = DEFAULT_CONFIG
    else:
        tmp = DEFAULT_CONFIG.copy()
        tmp.update(cfg)
        cfg = tmp

    device = torch.device(cfg["device"])

    ds_cfg = SimpleNamespace(k_external_words=0, n_aligned=0)
    dataset = HTRDataset(
        cfg["basefolder"],
        subset="train",
        fixed_size=cfg["fixed_size"],
        transforms=None,
        config=ds_cfg,
        character_classes = list(" 0123456789abcdefghijklmnopqrstuvwxyz")
    )

    indices = list(range(len(dataset)))
    if cfg["n_examples"]:
        random.seed(0)
        indices = random.sample(indices, min(cfg["n_examples"], len(indices)))
    subset_ds = Subset(dataset, indices)

    c2i, i2c = load_vocab()

    arch = SimpleNamespace(**ARCH)
    net = HTRNet(arch, nclasses=len(c2i) + 1).to(device)

    loader = DataLoader(subset_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    opt = optim.Adam(net.parameters(), lr=cfg["learning_rate"])
    sched = lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)

    for epoch in range(1, cfg["num_epochs"] + 1):
        epoch_loss = 0.0
        for imgs, txts, _ in loader:
            imgs = imgs.to(device)
            out = net(imgs, return_feats=False)
            if isinstance(out, (tuple, list)):
                main_logits = out[0]
                aux_logits = out[1] if len(out) > 1 else None
            else:
                main_logits = out
                aux_logits = None
            targets, tgt_lens = encode_for_ctc(list(txts), c2i)
            inp_lens = torch.full((imgs.size(0),), main_logits.size(0), dtype=torch.int32)
            loss = _ctc_loss_fn(main_logits, targets, inp_lens, tgt_lens)
            if aux_logits is not None:
                loss = loss + 0.1 * _ctc_loss_fn(aux_logits, targets, inp_lens, tgt_lens)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        sched.step()
        if epoch % cfg["eval_interval"] == 0 or epoch == cfg["num_epochs"]:
            print(f"Epoch {epoch:03d} Loss {epoch_loss/len(loader):.4f}")
            _evaluate(net, loader, i2c, device)
            _decode_random(net, dataset, i2c, device)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train on HTRDataset subset")
    parser.add_argument("--basefolder", type=str, default=DEFAULT_CONFIG["basefolder"])
    parser.add_argument("--n-examples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["num_epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--device", type=str, default=DEFAULT_CONFIG["device"])
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_CONFIG["eval_interval"])
    args = parser.parse_args()
    main({
        "basefolder": args.basefolder,
        "n_examples": args.n_examples,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "device": args.device,
        "eval_interval": args.eval_interval,
    })
