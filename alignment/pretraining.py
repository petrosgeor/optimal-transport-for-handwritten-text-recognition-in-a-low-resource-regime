from __future__ import annotations
import os, sys, random
from pathlib import Path
from types import SimpleNamespace
from omegaconf import OmegaConf
from contextlib import contextmanager


# ------------------------------------------------------------------
# Configuration parameters for pretraining
# Load base config and set defaults
# ------------------------------------------------------------------
cfg_file = Path(__file__).resolve().parents[1] / "alignment" / "config.yaml"
if cfg_file.exists():
    cfg = OmegaConf.load(cfg_file)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    DEVICE = cfg.device
else:
    DEVICE = "cpu"

    

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
    "n_random": 10000,
    "num_epochs": 200,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "base_path": None,
    "fixed_size": (64, 256),
    "device": DEVICE,
    "use_augmentations": False,
    "main_loss_weight": 1.0,
    "aux_loss_weight": 0.1,
    "save_path": "htr_base/saved_models/pretrained_backbone.pt"
}

# Architecture configuration for the pretraining backbone
# Matches exactly the config used in alignment_trainer.py
ARCHITECTURE_CONFIG = {
    "cnn_cfg": [[2, 64], "M", [3, 128], [2, 256]],
    "head_type": "both",
    "rnn_type": "gru",
    "rnn_layers": 3,
    "rnn_hidden_size": 256,
    "flattening": "maxpool",
    "stn": False,
    "feat_dim": None,
}

def _build_vocab(transcriptions):
    """Build character-to-index mapping from transcriptions."""
    chars = sorted(set(''.join(transcriptions)))
    if ' ' not in chars:
        chars.append(' ')
    c2i = {c: i + 1 for i, c in enumerate(chars)}
    return c2i

def main(config: dict = None) -> Path:
    """Train a small HTRNet on the given image list using dictionary configuration."""
    if config is None:
        config = PRETRAINING_CONFIG.copy()
    
    # Extract parameters from config
    list_file = config["list_file"]
    n_random = config.get("n_random", None)
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    lr = config["learning_rate"]
    base_path = config.get("base_path", None)
    fixed_size = config["fixed_size"]
    device = config["device"]
    use_augmentations = config.get("use_augmentations", True)
    main_weight = config.get("main_loss_weight", 1.0)
    aux_weight = config.get("aux_loss_weight", 0.1)
    save_path = config.get("save_path", "htr_base/saved_models/pretrained_backbone.pt")
    
    print(f"[Pretraining] Starting with config:")
    print(f"  list_file: {list_file}")
    print(f"  n_random: {n_random}")
    print(f"  epochs: {num_epochs}")
    print(f"  batch_size: {batch_size}")
    print(f"  learning_rate: {lr}")
    print(f"  device: {device}")
    print(f"  augmentations: {use_augmentations}")
    
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
    
    # Build vocabulary and architecture
    c2i = _build_vocab(train_set.transcriptions)
    nclasses = len(c2i) + 1
    print(f"[Pretraining] Vocabulary size: {nclasses} (including blank)")
    i2c = {i: c for c, i in c2i.items()}
    
    arch = SimpleNamespace(**ARCHITECTURE_CONFIG)
    net = HTRNet(arch, nclasses=nclasses).to(device).train()
    
    # Print number of parameters
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"[Pretraining] Network parameters: {n_params:,}")
    
    # Create data loader and optimizer
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=3)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    opt = optim.Adam(net.parameters(), lr=lr)
    sched = lr_scheduler.StepLR(opt, step_size=1000, gamma=0.5)
    
    print(f"[Pretraining] Starting training...")
    
    def _decode_random_samples(ds):
        """Print predictions for up to ten random samples from *ds*."""
        net.eval()
        with torch.no_grad():
            indices = random.sample(range(len(ds)), min(10, len(ds)))
            for idx in indices:
                img, gt = ds[idx]
                logits = net(img.unsqueeze(0).to(device), return_feats=False)[0]
                greedy = greedy_ctc_decode(logits, i2c)[0]
                beam = beam_search_ctc_decode(logits, i2c, beam_width=5)[0]
                print(
                    f"GT: '{gt.strip()}' | greedy: '{greedy}' | beam5: '{beam}'"
                )
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
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / max(1, num_batches)

        sched.step()

        # Print progress every 20 epochs or on last epoch
        if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
            lr_val = sched.get_last_lr()[0]
            print(
                f"[Pretraining] Epoch {epoch+1:03d}/{num_epochs} - Loss: {avg_loss:.4f} - lr: {lr_val:.2e}"
            )

        # Evaluate on the test set and decode examples every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            _evaluate_cer(test_loader)
            _decode_random_samples(test_set)

    # Save the trained model
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), save_path)
    print(f"[Pretraining] Model saved to: {save_path}")
    
    return Path(save_path)

if __name__ == '__main__':
    # Update config with command line arguments if provided
    config = PRETRAINING_CONFIG.copy()
    args = None

    if len(sys.argv) > 1:
        # Simple command line interface for common parameters
        import argparse
        parser = argparse.ArgumentParser(description='Pretrain HTR backbone using dictionary configuration')
        parser.add_argument('--list-file', type=str, help='text file with image paths')
        parser.add_argument('--n-random', type=int, help='sample this many images at random')
        parser.add_argument('--epochs', type=int, help='number of training epochs')
        parser.add_argument('--batch-size', type=int, help='batch size for training')
        parser.add_argument('--lr', type=float, help='learning rate')
        parser.add_argument('--device', type=str, help='device for training (cpu/cuda)')
        parser.add_argument('--no-augmentations', action='store_true', help='disable data augmentations')
        parser.add_argument('--save-path', type=str, help='path to save the trained model')
        parser.add_argument('--no-results-file', action='store_true',
                            help='do NOT duplicate stdout to pretraining_results.txt')

        args = parser.parse_args()

        # Update config with provided arguments
        if args.list_file is not None:
            config["list_file"] = args.list_file
        if args.n_random is not None:
            config["n_random"] = args.n_random
        if args.epochs is not None:
            config["num_epochs"] = args.epochs
        if args.batch_size is not None:
            config["batch_size"] = args.batch_size
        if args.lr is not None:
            config["learning_rate"] = args.lr
        if args.device is not None:
            config["device"] = args.device
        if args.no_augmentations:
            config["use_augmentations"] = False
        if args.save_path is not None:
            config["save_path"] = args.save_path

    # Run pretraining with the configuration while logging output
    if args and getattr(args, 'no_results_file', False):
        main(config)
    else:
        with tee_output("pretraining_results.txt"):
            main(config)
