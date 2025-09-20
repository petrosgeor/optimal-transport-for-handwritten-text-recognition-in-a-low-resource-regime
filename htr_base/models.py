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

    Purpose:
        Build a CNN trunk following ``cnn_cfg`` and apply an optional flattening
        step that prepares the feature map for CTC heads.
    Args:
        cnn_cfg (list): Sequence describing convolutional blocks and pooling
            layers. Integers denote residual blocks, ``'M'`` denotes max pooling.
        flattening (str): Flattening strategy. ``'maxpool'`` collapses the height
            dimension with a global max-pool; ``'concat'`` keeps all spatial rows
            by reshaping; ``'avgpool'`` leaves the feature map untouched (the head
            must therefore cope with multi-row inputs).
    Returns:
        torch.Tensor: Feature map after the requested flattening step.
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

    Purpose:
        Map the CNN feature map to sequence logits in a lightweight fashion.
        The design assumes the incoming tensor has height ``1``; when upstream
        layers output taller maps (e.g. ``flattening='concat'``) only the first
        row is consumed. Keep this in mind when extending the architecture.
    Args:
        input_size (int): Number of input channels produced by the CNN trunk.
        nclasses (int): Number of output classes (including blank).
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

    Additional args in ``arch_cfg``:
        feat_dim (int | None): Size of per-image descriptor.
        feat_pool (str): Pooling for CNN-based feat head, ``'avg'`` or ``'attn'``.
        feat_source (str): Source of holistic feature vector. One of:
            - ``'cnn'``: use CNN feature map + ``feat_pool``.
            - ``'rnn_mean'``: mean over time of the RNN output (for RNN heads) projected
              to ``feat_dim``. Falls back to ``'cnn'`` path if head is not RNN-based.

    Notes:
        • For ``flattening='concat'`` the number of CNN channels is derived via a
          heuristic (``2 * 8 * last_block_channels``) matching the default paper
          setup. Changing the spatial resolution or block layout requires
          updating this calculation manually.
        • When ``feat_source='rnn_mean'`` the implementation attaches a temporary
          forward hook to the recurrent head so the sequence output can be
          averaged. Heads without a ``rec`` attribute automatically fall back to
          the CNN pooling path.
    """
    def __init__(self, arch_cfg, nclasses):
        super().__init__()
        self.feat_pool = getattr(arch_cfg, 'feat_pool', 'avg')
        self.feat_source = getattr(arch_cfg, 'feat_source', 'cnn')
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
            # Optional projection for RNN-based holistic feature when requested.
            # Create only when the head is RNN-based to avoid relying on missing dims.
            head = arch_cfg.head_type
            if self.feat_source == 'rnn_mean' and head in {'rnn', 'both'}:
                rnn_hidden = int(getattr(arch_cfg, 'rnn_hidden_size'))
                self.seq_feat_proj = nn.Linear(2 * rnn_hidden, self.feat_dim)
            else:
                self.seq_feat_proj = None
        
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
        """
        Forward through CNN and head; derive features from the RNN output when enabled.

        Assumes a recurrent head provides a sequence output at ``self.top.rec`` and
        ``feat_source == 'rnn_mean'`` so the sequence is averaged over time and
        projected to ``feat_dim``. The return signature follows the contract:
        logits first, then ``feat`` (and optional PHOC).

        Args:
            x (torch.Tensor): Input images ``(B, 1, H, W)``.
            return_feats (bool): If False, only logits are returned.

        Returns:
            torch.Tensor | tuple: Logits, optionally followed by ``feat`` and
            ``phoc_logits`` if enabled.

        Notes:
            • When a recurrent head is present and ``feat_source='rnn_mean'``, a
              forward hook captures the sequence tensor so it can be averaged. If
              you subclass the heads, keep the ``rec`` attribute to retain this
              behaviour.
            • If the hook does not trigger (e.g. CNN-only heads) the feature head
              falls back to the CNN pooling branch defined by ``feat_pool``.
        """
        y = self.features(x)

        # Capture RNN sequence outputs via a temporary hook when building features.
        captured = {}
        handle = None
        if self.feat_dim and return_feats and hasattr(self.top, 'rec') and getattr(self, 'seq_feat_proj', None) is not None:
            def _hook(_, _inputs, output):
                seq = output[0] if isinstance(output, tuple) else output
                captured['seq'] = seq  # (T, B, 2H)
            handle = self.top.rec.register_forward_hook(_hook)

        logits = self.top(y)

        if handle is not None:
            handle.remove()

        feat = None
        if self.feat_dim and return_feats:
            if 'seq' in captured:
                seq_mean = captured['seq'].mean(dim=0)  # (T,B,2H) -> (B,2H)
                feat = self.seq_feat_proj(seq_mean)
            else:
                feat = self.feat_head(y)
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
    
