# Copyright
from __future__ import annotations
import os, sys, random, pickle
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager, nullcontext
# ------------------------------------------------------------------
# Configuration parameters for pretraining
# ------------------------------------------------------------------
# Add project root to path for imports
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from htr_base.utils.htr_dataset import PretrainingHTRDataset
from htr_base.models import HTRNet
from htr_base.utils.transforms import aug_transforms
from htr_base.utils.metrics import CER
from alignment.ctc_utils import (
    encode_for_ctc,
    greedy_ctc_decode,
    beam_search_ctc_decode,
)
from alignment.losses import _ctc_loss_fn
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
class _Tee:
    """Write to multiple streams simultaneously."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()
@contextmanager
def tee_output(path: str = "pretraining_results.txt"):
    """Duplicate stdout to *path* while the context is active."""
    original = sys.stdout
    with open(path, "w") as f:
        sys.stdout = _Tee(original, f)
        try:
            yield
        finally:
            sys.stdout = original
# Default pretraining configuration
PRETRAINING_CONFIG = {
    "list_file": "/gpu-data3/pger/handwriting_rec/mnt/ramdisk/max/90kDICT32px/imlist.txt",
    "n_random": 30000,
    "num_epochs": 10000,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "base_path": None,
    "fixed_size": (64, 256),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "gpu_id": 0,
    "use_augmentations": True,
    "main_loss_weight": 1.0,
    "aux_loss_weight": 0.1,
    "save_path": "htr_base/saved_models/pretrained_backbone.pt",
    "save_backbone": True,
    "results_file": False,
}
DEVICE = PRETRAINING_CONFIG["device"]
# Architecture configuration for the pretraining backbone
# Matches exactly the config used in alignment_trainer.py
ARCHITECTURE_CONFIG = {
    "cnn_cfg": [[2, 64], "M", [3, 128], "M", [2, 256]],
    "head_type": "both",
    "rnn_type": "gru",
    "rnn_layers": 3,
    "rnn_hidden_size": 256,
    "flattening": "maxpool",
    "stn": False,
    "feat_dim": 512,
    "feat_pool": "attn",
}


def _build_vocab(transcriptions):
    """Build character-to-index mapping from transcriptions."""
    chars = sorted(set(''.join(transcriptions)))
    if ' ' not in chars:
        chars.append(' ')
    c2i = {c: i + 1 for i, c in enumerate(chars)}
    return c2i
def main(config: dict | None = None) -> Path:
    """Train a small HTRNet on the given image list using dictionary configuration."""
    if config is None:
        config = {}
    config = {**PRETRAINING_CONFIG, **config}
    results_file = config.pop("results_file", False)
    # Extract parameters from config
    list_file = config["list_file"]
    n_random = config.get("n_random", None)
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    lr = config["learning_rate"]
    base_path = config.get("base_path", None)
    fixed_size = config["fixed_size"]
    device = config["device"]
    gpu_id = config.get("gpu_id", None)
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        if str(device).startswith("cuda"):
            device = f"cuda:{gpu_id}"
    use_augmentations = config.get("use_augmentations", True)
    main_weight = config.get("main_loss_weight", 1.0)
    aux_weight = config.get("aux_loss_weight", 0.1)
    save_path = config.get("save_path", "htr_base/saved_models/pretrained_backbone.pt")
    save_backbone = config.get("save_backbone", False)
    ctx = tee_output("pretraining_results.txt") if results_file else nullcontext()
    with ctx:
        print(f"[Pretraining] Starting with config:")
        print(f"  list_file: {list_file}")
        print(f"  n_random: {n_random}")
        print(f"  epochs: {num_epochs}")
        print(f"  batch_size: {batch_size}")
        print(f"  learning_rate: {lr}")
        print(f"  device: {device}")
        print(f"  augmentations: {use_augmentations}")
        print(f"  save_backbone: {save_backbone}")
        print(f"  gpu_id: {gpu_id}")
        if base_path is None:
            base_path = str(Path(list_file).parent)
        # Create training dataset with optional augmentations
        transforms = aug_transforms if use_augmentations else None
        train_set = PretrainingHTRDataset(
            list_file,
            fixed_size=fixed_size,
            base_path=base_path,
            transforms=transforms,
            n_random=n_random,
            preload_images=True,
            random_seed=0,
        )
        # Separate test set without augmentations
        test_set = PretrainingHTRDataset(
            list_file,
            fixed_size=fixed_size,
            base_path=base_path,
            transforms=None,
            n_random=10000,
            preload_images=True,
            random_seed=1,
        )
        print(f"[Pretraining] Dataset size: {len(train_set)}")
        print(f"[Pretraining] Test set size: {len(test_set)}")
        # ------------------------------------------------------------------ #
        # ❶ Always rebuild the vocabulary from the *current* training list   #
        # ------------------------------------------------------------------ #
        save_dir = Path(save_path).parent
        c2i_path = save_dir / "c2i.pkl"
        i2c_path = save_dir / "i2c.pkl"

        c2i = _build_vocab(train_set.transcriptions)   # fresh mapping
        i2c = {i: c for c, i in c2i.items()}

        # Optionally (re-)save the dictionaries next to the backbone weights
        if save_backbone:
            save_dir.mkdir(parents=True, exist_ok=True)
            with open(c2i_path, "wb") as f:
                pickle.dump(c2i, f)
            with open(i2c_path, "wb") as f:
                pickle.dump(i2c, f)

        nclasses = len(c2i) + 1
        print(f"[Pretraining] Vocabulary size: {nclasses} (including blank)")
        arch = SimpleNamespace(**ARCHITECTURE_CONFIG)
        net = HTRNet(arch, nclasses=nclasses).to(device).train()
        # if Path(save_path).exists():
        #     print(f"[Pretraining] Loading checkpoint from {save_path}")
        #     state = torch.load(save_path, map_location=device)
        #     net.load_state_dict(state)
        # Print number of parameters
        n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"[Pretraining] Network parameters: {n_params:,}")
        # Create data loader and optimizer
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=3)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)
        opt = optim.Adam(net.parameters(), lr=lr)
        sched = lr_scheduler.StepLR(opt, step_size=1500, gamma=0.5)
        print(f"[Pretraining] Starting training...")
        def _decode_random_samples(ds):
            """Print predictions for up to ten random samples from *ds*."""
            net.eval()
            with torch.no_grad():
                # 1) pick up to 10 random indices
                indices = random.sample(range(len(ds)), min(10, len(ds)))
                # 2) load all (image, gt) pairs and stack into a batch
                imgs, gts = zip(*(ds[i] for i in indices))
                batch = torch.stack(imgs, dim=0).to(device)   # shape: (B, C, H, W)
                # 3) forward-pass the whole batch at once
                logits = net(batch, return_feats=False)[0]   # shape: (T, B, C)
                # 4) decode the entire batch with greedy and beam search
                greedy_preds = greedy_ctc_decode(logits, i2c)               # List[str], len=B
                beam_preds   = beam_search_ctc_decode(logits, i2c, beam_width=5)
                # 5) print GT vs. predictions
                for gt, gr, bm in zip(gts, greedy_preds, beam_preds):
                    print(f"GT: '{gt.strip()}' | greedy: '{gr}' | beam5: '{bm}'")
            net.train()
        def _evaluate_cer(loader):
            """Return CER on *loader* and print the value."""
            net.eval()
            metric = CER()
            with torch.no_grad():
                for imgs, trans in loader:
                    imgs = imgs.to(device)
                    logits = net(imgs, return_feats=False)[0]
                    preds = greedy_ctc_decode(logits, i2c)
                    for p, t in zip(preds, trans):
                        metric.update(p.strip(), t.strip())
            net.train()
            score = metric.score()
            print(f"[Eval] CER: {score:.4f}")
            return score
        # Training loop
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            for imgs, txts in train_loader:
                imgs = imgs.to(device)
                out = net(imgs, return_feats=False)
                main_logits, aux_logits = out[:2]
                # Prepare CTC targets
                targets, lengths = encode_for_ctc(list(txts), c2i)
                inp_lens = torch.full((imgs.size(0),), main_logits.size(0), dtype=torch.int32, device=device)
                # Compute losses
                loss_main = _ctc_loss_fn(main_logits, targets, inp_lens, lengths)
                loss_aux = _ctc_loss_fn(aux_logits, targets, inp_lens, lengths)
                loss = main_weight * loss_main + aux_weight * loss_aux
                # Optimization step
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                epoch_loss += loss.item()
                num_batches += 1
            avg_loss = epoch_loss / max(1, num_batches)
            sched.step()
            # Print progress every 20 epochs or on last epoch
            if (epoch + 1) % 30 == 0 or epoch == num_epochs - 1:
                lr_val = sched.get_last_lr()[0]
                print(
                    f"[Pretraining] Epoch {epoch+1:03d}/{num_epochs} - Loss: {avg_loss:.4f} - lr: {lr_val:.2e}"
                )
                _evaluate_cer(test_loader)
                _decode_random_samples(test_set)
                if save_backbone:
                    # Save the trained model and vocabulary
                    save_dir = Path(save_path).parent
                    save_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(net.state_dict(), save_path)
                    with open(c2i_path, "wb") as f:
                        pickle.dump(c2i, f)
                    with open(i2c_path, "wb") as f:
                        pickle.dump(i2c, f)
                    print(f"[Pretraining] Model saved to: {save_path}")
        return Path(save_path)
if __name__ == '__main__':
    main()
