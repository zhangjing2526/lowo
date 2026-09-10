# Shared UNet building blocks for the Haar variants (Mamba / Attention / CNN).
#
# These blocks are backbone-agnostic: they depend on lowo (wavelet operators)
# and MONAI, but NOT on mamba_ssm or any sequence-mixing module. Keeping them
# here lets the attention backbone import them without pulling in mamba.

from __future__ import annotations

import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetResBlock

from model_segmamba.lowo import HaarWavelet3D, WaveletConv3d


class MlpChannel(nn.Module):
    """Channel-mixing MLP (1x1 conv -> GELU -> 1x1 conv), channels-first."""

    def __init__(self, hidden_size, mlp_dim):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class WaveletGSC(nn.Module):
    """GSC with its two 3x3x3 convs replaced by wavelet-domain convs.

    Macro-structure (residual, two-branch gating, norms, ReLUs) is identical
    to the original GSC in segmamba.py.
    """

    def __init__(self, in_channels: int, learnable: bool = True,
                 mode: str = "theta") -> None:
        super().__init__()

        self.proj = WaveletConv3d(in_channels, learnable, mode)
        self.norm = nn.InstanceNorm3d(in_channels)
        self.nonliner = nn.ReLU()

        self.proj2 = WaveletConv3d(in_channels, learnable, mode)
        self.norm2 = nn.InstanceNorm3d(in_channels)
        self.nonliner2 = nn.ReLU()

        self.proj3 = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channels)
        self.nonliner3 = nn.ReLU()

        self.proj4 = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channels)
        self.nonliner4 = nn.ReLU()

    def forward(self, x):
        x_residual = x

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)

        return x + x_residual


class HaarUpBlock(nn.Module):
    """Replacement for UnetrUpBlock: transposed conv -> 1x1 expansion + IDWT.

    Skip connection handling and the fusion UnetResBlock are identical to
    monai's UnetrUpBlock (res_block=True).
    """

    def __init__(self, in_channels: int, out_channels: int,
                 norm_name="instance", learnable: bool = True, mode: str = "theta"):
        super().__init__()
        self.wave = HaarWavelet3D(learnable, mode)
        self.expand = nn.Conv3d(in_channels, 8 * out_channels, kernel_size=1)
        self.conv_block = UnetResBlock(
            3,
            out_channels + out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )

    def forward(self, inp, skip):
        out = self.wave.idwt(self.expand(inp))
        out = torch.cat((out, skip), dim=1)
        return self.conv_block(out)
