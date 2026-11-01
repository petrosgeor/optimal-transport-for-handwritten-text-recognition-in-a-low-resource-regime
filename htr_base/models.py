import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    """Map CNN descriptors to an embedding space using a small MLP.

    Args:
        input_dim (int): Dimensionality of the input features.
        output_dim (int): Size of the target embedding space.
        dropout (float): Dropout rate applied after activations.

    Returns:
        torch.Tensor: Projected descriptors of shape (N, output_dim).
    """

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.2) -> None:
        """Initialize the projector MLP.

        Args:
        - input_dim (int): Feature size of each input vector.
        - output_dim (int): Feature size of each projected vector.
        - dropout (float): Dropout probability applied between linear layers.
        """
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
            nn.Linear(self.input_dim, self.output_dim),
        )

    def forward(self, x):
        """Project descriptor vectors to the target embedding space.

        Args:
        - x (torch.Tensor): Input descriptors with shape (N, input_dim).

        Returns:
        - (torch.Tensor): Projected descriptors with shape (N, output_dim).
        """
        return self.sequential(x)


###############################################################################
#                        ─────  CNN building blocks  ─────                    #
###############################################################################
class BasicBlock(nn.Module):
    """Residual CNN block with two convolutions and an optional shortcut.

    Args:
        in_planes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int): Convolution stride for conv1.

    Returns:
        torch.Tensor: Output tensor with planes channels.
    """

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        """Initialize the residual block.

        Args:
        - in_planes (int): Number of input channels.
        - planes (int): Number of output channels.
        - stride (int): Stride used by the first convolution.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = (
            nn.Sequential()
            if stride == 1 and in_planes == planes
            else nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes)
            )
        )

    def forward(self, x):
        """Apply residual convolutional transformation.

        Args:
        - x (torch.Tensor): Input feature map with shape (B, in_planes, H, W).

        Returns:
        - (torch.Tensor): Output feature map with shape (B, planes, H', W').
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class CNN(nn.Module):
    """Configurable convolutional feature extractor for handwriting images.

    Purpose:
        Build a CNN trunk following cnn_cfg, then collapse the height
        dimension with a global max-pool to prepare the feature map for the
        CTC head.
    Args:
        cnn_cfg (list): Sequence describing convolutional blocks and pooling
            layers. Integers denote residual blocks, M denotes max pooling.
    Returns:
        torch.Tensor: Feature map with shape (B, C_out, 1, W').
    """

    def __init__(self, cnn_cfg):
        """Build the CNN trunk from configuration tokens.

        Args:
        - cnn_cfg (list): Convolution/pooling layout definition.
        """
        super().__init__()
        self.k = 1
        self.features = nn.ModuleList([nn.Conv2d(1, 32, 7, (4, 2), 3), nn.ReLU()])
        in_channels = 32
        pool_cnt, blk_cnt = 0, 1
        for m in cnn_cfg:
            if m == "M":
                self.features.add_module(f"mxp{pool_cnt}", nn.MaxPool2d(2, 2))
                pool_cnt += 1
            else:
                reps, ch = m
                for _ in range(int(reps)):
                    self.features.add_module(f"blk{blk_cnt}", BasicBlock(in_channels, int(ch)))
                    in_channels = int(ch)
                    blk_cnt += 1

    def forward(self, x):
        """Run the CNN trunk and max-pool over the height dimension.

        Args:
        - x (torch.Tensor): Input images with shape (B, 1, H, W).

        Returns:
        - (torch.Tensor): Feature map with shape (B, C_out, 1, W').
        """
        y = x
        for m in self.features:
            y = m(y)
        y = F.max_pool2d(
            y, [y.size(2), self.k], stride=[y.size(2), 1], padding=[0, self.k // 2]
        )
        return y


###############################################################################
#                    ─────  CTC recognition heads  ─────               #
###############################################################################
class CTCtopB(nn.Module):
    """CTC head with both recurrent and convolutional branches.

    Args:
        input_size (int): Number of input channels.
        rnn_cfg (Tuple[int, int]): (hidden_size, num_layers) for the RNN.
        nclasses (int): Number of output classes.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Main and auxiliary logits.
    """

    def __init__(self, input_size, rnn_cfg, nclasses):
        """Initialize the dual-branch CTC head.

        Args:
        - input_size (int): Number of input channels from the CNN trunk.
        - rnn_cfg (tuple[int, int]): Recurrent config as (hidden_size, num_layers).
        - nclasses (int): Number of CTC classes including blank.
        """
        super().__init__()
        hidden, num_layers = rnn_cfg
        self.rec = nn.GRU(
            input_size, hidden, num_layers=num_layers, bidirectional=True, dropout=0.2
        )
        self.fnl = nn.Sequential(nn.Dropout(0.5), nn.Linear(2 * hidden, nclasses))
        self.cnn = nn.Sequential(
            nn.Dropout(0.5), nn.Conv2d(input_size, nclasses, (1, 3), 1, (0, 1))
        )

    def forward(self, x):
        """Compute main recurrent logits and auxiliary convolutional logits.

        Args:
        - x (torch.Tensor): Feature map with shape (B, C_in, 1, W').

        Returns:
        - (tuple[torch.Tensor, torch.Tensor]): Main and auxiliary logits,
          each with shape (T, B, nclasses).
        """
        y = x.permute(2, 3, 0, 1)[0]
        y = self.rec(y)[0]
        y = self.fnl(y)
        aux = self.cnn(x).permute(2, 3, 0, 1)[0]
        return y, aux


###############################################################################
#                 ─────  HTRNet with optional feat head  ─────                #
###############################################################################
class HTRNet(nn.Module):
    """
    HTRNet backbone with optional global feature projection.

    Additional args in arch_cfg:
        feat_dim (int | None): Size of per-image descriptor. The descriptor is the
            mean over time of the recurrent head's output, projected to feat_dim.
        phoc_levels (list[int] | None): PHOC pyramid levels. With feat_dim set, a
            PHOC head predicts the word's PHOC vector from the descriptor.

    Notes:
        - The implementation attaches a temporary forward hook to the recurrent
          head so the sequence output can be averaged.
    """

    def __init__(self, arch_cfg, nclasses):
        """Initialize the full HTR network.

        Args:
        - arch_cfg (Any): Architecture configuration namespace.
        - nclasses (int): Number of CTC classes including blank.
        """
        super().__init__()
        self.features = CNN(arch_cfg.cnn_cfg)

        # output channels after CNN
        cnn_out_ch = arch_cfg.cnn_cfg[-1][-1]

        # CTC head: recurrent main branch plus convolutional auxiliary branch
        self.top = CTCtopB(
            cnn_out_ch,
            (arch_cfg.rnn_hidden_size, arch_cfg.rnn_layers),
            nclasses,
        )

        # optional per-image feature vector: time-mean of the recurrent output
        self.feat_dim = getattr(arch_cfg, "feat_dim", None)
        if self.feat_dim:
            rnn_hidden = int(getattr(arch_cfg, "rnn_hidden_size"))
            self.seq_feat_proj = nn.Linear(2 * rnn_hidden, self.feat_dim)

        self.phoc_levels = getattr(arch_cfg, "phoc_levels", None)
        if self.feat_dim and self.phoc_levels:
            phoc_dim = (nclasses - 1) * sum(self.phoc_levels)
            self.phoc_head = nn.Sequential(
                nn.ReLU(), nn.Linear(self.feat_dim, phoc_dim)
            )
        else:
            self.phoc_head = None

    def forward(self, x, *, return_feats: bool = True):
        """
        Forward through CNN and head; derive features from the RNN output when enabled.

        The recurrent head's sequence output is averaged over time and projected
        to feat_dim. The return signature follows the contract: logits first,
        then feat (and optional PHOC).

        Args:
            x (torch.Tensor): Input images (B, 1, H, W).
            return_feats (bool): If False, only logits are returned.

        Returns:
        - (tuple): (main_logits, aux_logits), each with shape (T, B, C), when no
          features are returned.
          If return_feats is True and feat_dim is enabled:
          append feat with shape (B, D) after logits.
          If PHOC head is enabled and feat is present:
          append phoc_logits with shape (B, P) as the final element.

        Notes:
            - A forward hook on self.top.rec captures the sequence tensor so it
              can be averaged. If you subclass the head, keep the rec attribute.
        """
        y = self.features(x)

        # Capture RNN sequence outputs via a temporary hook when building features.
        captured = {}
        handle = None
        if self.feat_dim and return_feats:

            def _hook(_, _inputs, output):
                """Capture recurrent sequence output for mean pooling.

                Args:
                - _ (torch.nn.Module): Hooked recurrent module.
                - _inputs (tuple[torch.Tensor, ...]): Forward inputs where the
                  first tensor has shape (T, B, C_in).
                - output (torch.Tensor | tuple): Recurrent output with sequence
                  tensor shape (T, B, 2H), optionally packed in a tuple.
                """
                seq = output[0] if isinstance(output, tuple) else output
                captured["seq"] = seq  # (T, B, 2H)

            handle = self.top.rec.register_forward_hook(_hook)
        logits = self.top(y)

        if handle is not None:
            handle.remove()

        feat = None
        if self.feat_dim and return_feats:
            seq_mean = captured["seq"].mean(dim=0)  # (T,B,2H) -> (B,2H)
            feat = self.seq_feat_proj(seq_mean)
        phoc_logits = None
        if self.phoc_head is not None and feat is not None:
            phoc_logits = self.phoc_head(feat)
        if feat is None and phoc_logits is None:
            return logits
        out = (*logits, feat)
        if phoc_logits is not None:
            out = (*out, phoc_logits)
        return out
