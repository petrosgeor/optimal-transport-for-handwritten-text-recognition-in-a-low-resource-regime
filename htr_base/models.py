import torch
import torch.nn as nn
import torch.nn.functional as F


###############################################################################
#                            ─────  MLP Projector  ─────                       #
###############################################################################
class Projector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(Projector, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.sequential = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.output_dim)
        )

    def forward(self, x):
        return self.sequential(x)


###############################################################################
#                        ─────  CNN building blocks  ─────                    #
###############################################################################
class BasicBlock(nn.Module):
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
    def __init__(self, input_size, nclasses, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.cnn_top = nn.Conv2d(input_size, nclasses, (1,3), 1, (0,1))
    def forward(self, x):
        x = self.dropout(x)
        y = self.cnn_top(x)
        return y.permute(2,3,0,1)[0]

class CTCtopR(nn.Module):
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
    



# from types import SimpleNamespace

# # ---- 1. Define dummy arch config (match your config.yaml!) ----
# arch_cfg = SimpleNamespace(
#     cnn_cfg=[[2, 64], 'M', [3, 128], 'M', [2, 256]],
#     head_type='both',        # could be 'cnn', 'rnn', or 'both'
#     rnn_type='lstm',         # 'gru' or 'lstm'
#     rnn_layers=3,
#     rnn_hidden_size=256,
#     flattening='maxpool',
#     feat_dim=512,
#     stn=False,
#     feat_dim=512           # 
# )

# # ---- 2. Set output classes (e.g., 80 classes + 1 for CTC blank) ----
# nclasses = 80 + 1

# # ---- 3. Create HTRNet ----
# model = HTRNet(arch_cfg, nclasses)

# # ---- 4. Create a dummy input tensor (B, C, H, W) ----
# # e.g., batch of 2 images, 1 channel, height 128, width 1024
# dummy_input = torch.randn(2, 1, 128, 1024)

# # ---- 5. Run forward pass ----
# model.eval()
# with torch.no_grad():
#     main, aux, feats = model(dummy_input, return_feats=True)

