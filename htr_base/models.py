import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace


###############################################################################
#                            ─────  MLP Projector  ─────                       #
###############################################################################
class Projector(nn.Module):
    """Map CNN descriptors to an embedding space using a small MLP.

    Args:
        input_dim (int): Dimensionality of the input features.
        output_dim (int): Size of the target embedding space.
        dropout (float): Dropout rate applied after activations.

    Returns:
        torch.Tensor: Projected descriptors of shape ``(N, output_dim)``.
    """

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.2) -> None:
        super(Projector, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.sequential = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.input_dim, self.input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.input_dim, self.output_dim)
        )

    def forward(self, x):
        return self.sequential(x)


###############################################################################
#                        ─────  CNN building blocks  ─────                    #
###############################################################################
class BasicBlock(nn.Module):
    """Residual CNN block with two convolutions and an optional shortcut.

    Args:
        in_planes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int): Convolution stride for ``conv1``.

    Returns:
        torch.Tensor: Output tensor with ``planes`` channels.
    """

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.shortcut = (nn.Sequential()
                         if stride == 1 and in_planes == planes
                         else nn.Sequential(
                                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                                nn.BatchNorm2d(planes)))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class CNN(nn.Module):
    """Configurable convolutional feature extractor for handwriting images.

    Args:
        cnn_cfg (list): List describing convolutional layers and pooling.
        flattening (str): Output flattening mode, ``'maxpool'`` or ``'concat'``.

    Returns:
        torch.Tensor: Feature map after optional flattening.
    """

    def __init__(self, cnn_cfg, flattening='maxpool'):
        super().__init__()
        self.k = 1
        self.flattening = flattening
        self.features = nn.ModuleList([nn.Conv2d(1, 32, 7, (4,2), 3), nn.ReLU()])
        in_channels = 32
        pool_cnt, blk_cnt = 0, 1
        for m in cnn_cfg:
            if m == 'M':
                self.features.add_module(f'mxp{pool_cnt}', nn.MaxPool2d(2,2))
                pool_cnt += 1
            else:
                reps, ch = m
                for _ in range(int(reps)):
                    self.features.add_module(f'blk{blk_cnt}',
                                             BasicBlock(in_channels, int(ch)))
                    in_channels = int(ch)
                    blk_cnt += 1
    def forward(self, x):
        y = x
        for m in self.features:
            y = m(y)
        if self.flattening == 'maxpool':
            y = F.max_pool2d(y, [y.size(2), self.k],
                             stride=[y.size(2), 1],
                             padding=[0, self.k // 2])
        elif self.flattening == 'concat':
            y = y.view(y.size(0), -1, 1, y.size(3))
        return y

class AttentivePool(nn.Module):
    """Collapses a feature map via learnable attention weights."""

    def __init__(self, ch: int, dim_out: int) -> None:
        super().__init__()
        self.attn = nn.Conv2d(ch, 1, 1)
        self.proj = nn.Linear(ch, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        a = self.attn(x).flatten(2)
        a = F.softmax(a, dim=2)
        x_flat = x.flatten(2).transpose(1, 2)
        pooled = torch.bmm(a, x_flat).squeeze(1)
        return self.proj(pooled)

###############################################################################
#                    ─────  CTC recognition heads  ─────               #
###############################################################################
class CTCtopC(nn.Module):
    """CTC head implemented with a single convolutional layer.

    Args:
        input_size (int): Number of input channels.
        nclasses (int): Number of output classes.
        dropout (float): Dropout rate before the convolution.

    Returns:
        torch.Tensor: Sequence logits of shape ``(T, B, nclasses)``.
    """

    def __init__(self, input_size, nclasses, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.cnn_top = nn.Conv2d(input_size, nclasses, (1,3), 1, (0,1))
    def forward(self, x):
        x = self.dropout(x)
        y = self.cnn_top(x)
        return y.permute(2,3,0,1)[0]

class CTCtopR(nn.Module):
    """Recurrent CTC head using GRU or LSTM layers.

    Args:
        input_size (int): Number of input channels.
        rnn_cfg (Tuple[int, int]): ``(hidden_size, num_layers)`` for the RNN.
        nclasses (int): Number of output classes.
        rnn_type (str): ``'gru'`` or ``'lstm'``.

    Returns:
        torch.Tensor: Sequence logits of shape ``(T, B, nclasses)``.
    """

    def __init__(self, input_size, rnn_cfg, nclasses, rnn_type='gru'):
        super().__init__()
        hidden, num_layers = rnn_cfg
        if rnn_type == 'gru':
            self.rec = nn.GRU(input_size, hidden, num_layers=num_layers,
                              bidirectional=True, dropout=.2)
        elif rnn_type == 'lstm':
            self.rec = nn.LSTM(input_size, hidden, num_layers=num_layers,
                               bidirectional=True, dropout=.2)
        else:
            raise ValueError('Unknown rnn_type {}'.format(rnn_type))
        self.fnl = nn.Sequential(nn.Dropout(.2), nn.Linear(2*hidden, nclasses))
    def forward(self, x):
        y = x.permute(2,3,0,1)[0]   # T × B × C
        y = self.rec(y)[0]
        return self.fnl(y)

class CTCtopB(nn.Module):
    """CTC head with both recurrent and convolutional branches.

    Args:
        input_size (int): Number of input channels.
        rnn_cfg (Tuple[int, int]): ``(hidden_size, num_layers)`` for the RNN.
        nclasses (int): Number of output classes.
        rnn_type (str): ``'gru'`` or ``'lstm'``.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Main and auxiliary logits.
    """

    def __init__(self, input_size, rnn_cfg, nclasses, rnn_type='gru'):
        super().__init__()
        hidden, num_layers = rnn_cfg
        if rnn_type == 'gru':
            self.rec = nn.GRU(input_size, hidden, num_layers=num_layers,
                              bidirectional=True, dropout=.2)
        elif rnn_type == 'lstm':
            self.rec = nn.LSTM(input_size, hidden, num_layers=num_layers,
                               bidirectional=True, dropout=.2)
        else:
            raise ValueError('Unknown rnn_type {}'.format(rnn_type))
        self.fnl = nn.Sequential(nn.Dropout(.5), nn.Linear(2*hidden, nclasses))
        self.cnn = nn.Sequential(nn.Dropout(.5),
                                 nn.Conv2d(input_size, nclasses, (1,3), 1, (0,1)))
    def forward(self, x):
        y = x.permute(2,3,0,1)[0]
        y = self.rec(y)[0]
        y = self.fnl(y)
        aux = self.cnn(x).permute(2,3,0,1)[0]
        return y, aux

class CTCtopT(nn.Module):
    """Transformer-based CTC head."""
    def __init__(self, input_size, transf_cfg, nclasses):
        super().__init__()
        d_model, nhead, nlayers, dim_ff = transf_cfg
        self.proj = nn.Linear(input_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=0.2,
            batch_first=False
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.fc = nn.Linear(d_model, nclasses)

    def forward(self, x):
        y = x.permute(2,3,0,1)[0]
        y = self.proj(y)
        y = self.encoder(y)
        return self.fc(y)

###############################################################################
#                 ─────  HTRNet with optional feat head  ─────                #
###############################################################################
class HTRNet(nn.Module):
    """
    HTRNet backbone with optional global feature projection.

    Additional arg in *arch_cfg*:
        feat_dim (int | None): size of per-image descriptor.
    """
    def __init__(self, arch_cfg, nclasses):
        super().__init__()
        self.feat_pool = getattr(arch_cfg, 'feat_pool', 'avg')
        if getattr(arch_cfg, 'stn', False):
            raise NotImplementedError('STN not implemented in this repo.')

        self.features = CNN(arch_cfg.cnn_cfg, flattening=arch_cfg.flattening)

        # output channels after CNN
        if arch_cfg.flattening in ('maxpool', 'avgpool'):
            cnn_out_ch = arch_cfg.cnn_cfg[-1][-1]
        elif arch_cfg.flattening == 'concat':
            cnn_out_ch = 2 * 8 * arch_cfg.cnn_cfg[-1][-1]
        else:
            raise ValueError(f'Unknown flattening: {arch_cfg.flattening}')

        # choose CTC head
        head = arch_cfg.head_type
        if head == 'cnn':
            self.top = CTCtopC(cnn_out_ch, nclasses)
        elif head == 'rnn':
            self.top = CTCtopR(cnn_out_ch,
                               (arch_cfg.rnn_hidden_size,
                                arch_cfg.rnn_layers),
                               nclasses,
                               arch_cfg.rnn_type)
        elif head == 'both':
            self.top = CTCtopB(cnn_out_ch,
                               (arch_cfg.rnn_hidden_size,
                                arch_cfg.rnn_layers),
                               nclasses,
                               arch_cfg.rnn_type)
        elif head == 'transf':
            self.top = CTCtopT(cnn_out_ch,
                               (arch_cfg.transf_d_model,
                                arch_cfg.transf_nhead,
                                arch_cfg.transf_layers,
                                arch_cfg.transf_dim_ff),
                               nclasses)
        else:
            raise ValueError(f'Unknown head_type: {head}')

        # optional per-image feature vector
        self.feat_dim = getattr(arch_cfg, 'feat_dim', None)
        if self.feat_dim:
            if self.feat_pool == 'attn':
                self.feat_head = AttentivePool(cnn_out_ch, self.feat_dim)
            elif self.feat_pool == 'avg':
                self.feat_head = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(1),
                    nn.Linear(cnn_out_ch, self.feat_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.feat_dim, self.feat_dim)
                )
            else:
                raise ValueError(
                    f"Unknown feat_pool '{self.feat_pool}'. Supported: 'avg', 'attn'"
                )
        
        self.phoc_levels = getattr(arch_cfg, 'phoc_levels', None)
        if self.feat_dim and self.phoc_levels:
            phoc_dim = (nclasses - 1) * sum(self.phoc_levels)
            self.phoc_head = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(self.feat_dim, phoc_dim)
            )
        else:
            self.phoc_head = None

    def forward(self, x, *, return_feats: bool = True):
        y = self.features(x)
        feat = self.feat_head(y) if self.feat_dim and return_feats else None
        logits = self.top(y)
        phoc_logits = None
        if self.phoc_head is not None and feat is not None:
            phoc_logits = self.phoc_head(feat)
        if feat is None and phoc_logits is None:
            return logits
        if isinstance(logits, tuple):
            out = (*logits, feat)
        else:
            out = (logits, feat)
        if phoc_logits is not None:
            out = (*out, phoc_logits)
        return out
    



###############################################################################
#                       ─────  HTRNetLength  (length CNN‑RNN)  ─────          #
###############################################################################
class HTRNetLength(nn.Module):
    """
    CNN + bidirectional RNN classifier that predicts the **number of characters**
    in a handwritten‑word image.

    Parameters
    ----------
    arch_cfg : Namespace | dict
        Same architecture configuration used by :class:`HTRNet`.  The keys that
        matter here are:
          • ``cnn_cfg``           – convolutional backbone definition  
          • ``flattening``        – 'maxpool' | 'avgpool' | 'concat'  
          • ``rnn_type``          – 'gru' | 'lstm' (optional, default 'gru')  
          • ``rnn_hidden_size``   – hidden units per direction (default =256)  
          • ``rnn_layers``        – number of stacked RNN layers (default =2)
    n_lengths : int
        How many discrete length classes you want to predict.  
        E.g. `n_lengths = 20` for classes *{1, …, 20}*.

    Output
    ------
    logits : torch.Tensor, shape = (B, n_lengths)
        Raw (unnormalised) scores for every possible length.
    """
    def __init__(self, arch_cfg, n_lengths: int):
        super().__init__()

        # ---------- convolutional feature extractor ----------
        self.features = CNN(arch_cfg.cnn_cfg, flattening=arch_cfg.flattening)

        # CNN output channels (same logic as in HTRNet)
        if arch_cfg.flattening in ("maxpool", "avgpool"):
            cnn_out_ch = arch_cfg.cnn_cfg[-1][-1]
        elif arch_cfg.flattening == "concat":
            cnn_out_ch = 2 * 8 * arch_cfg.cnn_cfg[-1][-1]
        else:
            raise ValueError(f"Unknown flattening: {arch_cfg.flattening}")

        # ---------- recurrent encoder ----------
        hidden_size   = getattr(arch_cfg, "rnn_hidden_size", 256)
        num_layers    = getattr(arch_cfg, "rnn_layers", 2)
        rnn_type      = getattr(arch_cfg, "rnn_type", "gru").lower()

        if rnn_type == "gru":
            self.rec = nn.GRU(
                cnn_out_ch,
                hidden_size,
                num_layers=num_layers,
                bidirectional=True,
                dropout=0.2 if num_layers > 1 else 0.0,
            )
        elif rnn_type == "lstm":
            self.rec = nn.LSTM(
                cnn_out_ch,
                hidden_size,
                num_layers=num_layers,
                bidirectional=True,
                dropout=0.2 if num_layers > 1 else 0.0,
            )
        else:
            raise ValueError(f"Unsupported rnn_type '{rnn_type}'")

        # ---------- final linear classifier ----------
        self.fc = nn.Linear(2 * hidden_size, n_lengths)

        # Store for convenience
        self.n_lengths = n_lengths
        self.hidden_size = hidden_size

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : torch.Tensor, shape = (B, 1, H, W)
        """
        # 1) CNN feature map  –––––––––––––––––––––––––––––––––––––––––––––––
        y = self.features(x)                               # (B, C, 1, T)
        y = y.squeeze(2)                                   # (B, C, T)

        # 2) sequence → RNN expects (T, B, C)
        seq = y.permute(2, 0, 1).contiguous()              # (T, B, C)

        # 3) Run bidirectional RNN  ––––––––––––––––––––––––––––––––––––––––
        _, h_n = self.rec(seq)                             # h_n: (num_layers*2, B, H)

        # 4) Concatenate last forward + backward hidden states
        if isinstance(h_n, tuple):                         # LSTM returns (h, c)
            h_n = h_n[0]
        h_fwd  = h_n[-2]                                   # (B, H)
        h_back = h_n[-1]                                   # (B, H)
        h_cat  = torch.cat([h_fwd, h_back], dim=1)         # (B, 2H)

        # 5) Final length logits
        return self.fc(h_cat)                              # (B, n_lengths)






