from pathlib import Path
from types import SimpleNamespace
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from htr_base.utils.htr_dataset import PretrainingHTRDataset
from htr_base.models import HTRNet
from alignment.ctc_utils import encode_for_ctc
from alignment.losses import _ctc_loss_fn


def _build_vocab(transcriptions):
    chars = sorted(set(''.join(transcriptions)))
    if ' ' not in chars:
        chars.append(' ')
    c2i = {c: i + 1 for i, c in enumerate(chars)}
    return c2i


def main(
    list_file: str,
    *,
    n_random: int | None = None,
    num_epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    base_path: str | None = None,
    fixed_size: tuple = (64, 256),
    device: str = 'cpu',
) -> Path:
    """Train a small HTRNet on the given image list."""
    if base_path is None:
        base_path = str(Path(list_file).parent)
    dataset = PretrainingHTRDataset(
        list_file,
        fixed_size=fixed_size,
        base_path=base_path,
        transforms=None,
        n_random=n_random,
    )

    c2i = _build_vocab(dataset.transcriptions)
    nclasses = len(c2i) + 1

    arch = SimpleNamespace(
        cnn_cfg=[[2, 32], 'M', [2, 64]],
        head_type='both',
        rnn_type='gru',
        rnn_layers=1,
        rnn_hidden_size=128,
        flattening='maxpool',
        stn=False,
        feat_dim=None,
    )
    net = HTRNet(arch, nclasses=nclasses).to(device).train()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    opt = optim.Adam(net.parameters(), lr=lr)

    for _ in range(num_epochs):
        for imgs, txts in loader:
            imgs = imgs.to(device)
            out = net(imgs, return_feats=False)
            main_logits, aux_logits = out[:2]
            targets, lengths = encode_for_ctc(list(txts), c2i)
            inp_lens = torch.full((imgs.size(0),), main_logits.size(0), dtype=torch.int32, device=device)
            loss_main = _ctc_loss_fn(main_logits, targets, inp_lens, lengths)
            loss_aux = _ctc_loss_fn(aux_logits, targets, inp_lens, lengths)
            loss = loss_main + 0.1 * loss_aux
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    save_dir = Path('htr_base/saved_models')
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'pretrained_backbone.pt'
    torch.save(net.state_dict(), save_path)
    return save_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Pretrain HTR backbone on image list')
    parser.add_argument('list_file', help='text file with image paths')
    parser.add_argument('--n-random', type=int, default=None, help='sample this many images at random')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--base-path', type=str, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    main(
        args.list_file,
        n_random=args.n_random,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        base_path=args.base_path,
        device=args.device,
    )
