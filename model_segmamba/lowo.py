# LOWO: Learnable Orthogonal Wavelet Operator.
#
# A backbone-agnostic, drop-in replacement for spatial down/up-sampling and
# (equal-channel) spatial convolutions, based on a learnable 3D Haar wavelet
# transform. This module has no mamba / attention / UNet dependency: it only imports torch, so it can be imported from a CNN,
# attention, or SSM backbone without pulling in mamba_ssm or MONAI.
#
# Strict "replacement" (not an add-on branch) of:
#   1. stem Conv3d(k=7, s=2)            -> HaarDownsample (3D Haar DWT + 1x1 subband mixing)
#   2. downsample Conv3d(k=2, s=2)      -> HaarDownsample
#   3. decoder ConvTranspose3d(k=2,s=2) -> HaarUpsample (1x1 subband expansion + 3D Haar IDWT)
#   4. equal-channel 3x3x3 conv         -> WaveletConv3d (DWT -> grouped 1x1 -> IDWT)
#
# The 2-tap analysis filters are parameterized by a per-axis rotation angle theta:
#     low  = [cos(t),  sin(t)]
#     high = [-sin(t), cos(t)]
# which is orthonormal for ANY t, so DWT <-> IDWT perfect reconstruction is
# preserved even while the filters are being learned. t = pi/4 is standard Haar.
#
# wavelet_mode="so8" replaces the 3-DOF separable parameterization with the
# full 28-DOF non-separable one: K = Q @ K_haar with Q = expm(A - A^T) in
# SO(8) (theta frozen at pi/4). Q is orthogonal for any A, so perfect
# reconstruction still holds; A=0 init makes it start exactly at Haar.
#
# NOTE (2026-08-26 patch): HaarDownsample.mix changed from groups=1 to
# groups=8 so the learnable so8 basis Q is no longer absorbed by the mixer,
# and the kernel is cached when learnable=False.

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarWavelet3D(nn.Module):
    """Axis-separable orthogonal 2-tap wavelet transform, Haar-initialized.

    dwt : (B, C, D, H, W)   -> (B, 8C, D/2, H/2, W/2)   subband-major layout
    idwt: (B, 8C, d, h, w)  -> (B, C, 2d, 2h, 2w)

    Spatial sizes must be even for dwt. Subband order (major to minor bit):
    D-axis, H-axis, W-axis; index 0 = LLL, index 7 = HHH.
    """

    def __init__(self, learnable: bool = True, mode: str = "theta"):
        super().__init__()
        assert mode in ("theta", "so8"), mode
        self.mode = mode
        theta = torch.full((3,), math.pi / 4)
        if learnable and mode == "theta":
            self.theta = nn.Parameter(theta)
        else:
            # in "so8" mode the separable angles are frozen at Haar; all
            # learnable capacity lives in the non-separable SO(8) factor.
            self.register_buffer("theta", theta)
        if mode == "so8":
            # 28-DOF skew-symmetric generator A; Q = expm(A - A^T) is exactly
            # orthogonal (det=1) for any A, so perfect reconstruction holds
            # throughout training. A=0 -> Q=I -> bit-identical to fixed Haar.
            skew = torch.zeros(8, 8)
            if learnable:
                self.skew = nn.Parameter(skew)
            else:
                self.register_buffer("skew", skew)
        # FIX 2: cache kernel when nothing is learnable
        if not learnable:
            self.register_buffer("_k_cache", self._kernel())

    def _kernel(self) -> torch.Tensor:
        """Fused (8, 1, 2, 2, 2) analysis kernel: tensor product of the three
        per-axis 2-tap filters. Output row k = bD*4 + bH*2 + bW (0=LLL, 7=HHH).
        Mathematically identical to applying the three per-axis filters
        sequentially, but runs as ONE conv3d (far fewer kernel launches)."""
        c, s = torch.cos(self.theta), torch.sin(self.theta)
        f = torch.stack([torch.stack([c, s], dim=-1),
                         torch.stack([-s, c], dim=-1)], dim=-2)  # (3, 2, 2)
        k = torch.einsum('ad,bh,cw->abcdhw', f[0], f[1], f[2]).reshape(8, 8)
        if self.mode == "so8":
            q = torch.matrix_exp(self.skew - self.skew.transpose(0, 1))
            k = q @ k
        return k.reshape(8, 1, 2, 2, 2)

    @torch.no_grad()
    def ortho_dev(self) -> float:
        """||Q - I||_F: drift of the SO(8) mixing from identity (0.0 in theta mode)."""
        if self.mode != "so8":
            return 0.0
        q = torch.matrix_exp(self.skew - self.skew.transpose(0, 1))
        return (q - torch.eye(8, device=q.device, dtype=q.dtype)).norm().item()

    def _get_kernel(self) -> torch.Tensor:
        if hasattr(self, "_k_cache"):
            return self._k_cache
        return self._kernel()

    def dwt(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape[:2]
        k = self._get_kernel()
        y = F.conv3d(x.reshape(B * C, 1, *x.shape[2:]), k, stride=2)
        y = y.reshape(B, C, 8, *y.shape[2:])
        y = y.permute(0, 2, 1, 3, 4, 5)  # subband-major
        return y.reshape(B, 8 * C, *y.shape[3:])

    def idwt(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        C = x.shape[1] // 8
        y = x.reshape(B, 8, C, *x.shape[2:])
        y = y.permute(0, 2, 1, 3, 4, 5)  # (B, C, 8, ...)
        y = y.reshape(B * C, 8, *y.shape[3:])
        k = self._get_kernel()
        y = F.conv_transpose3d(y, k, stride=2)
        return y.reshape(B, C, *y.shape[2:])


class HaarDownsample(nn.Module):
    """Replacement for strided Conv3d downsampling: DWT + 1x1 subband mixing.

    Default (grouped_mix=False): one shared 1x1 mixes all 8*in_channels
    subband-channels at once (groups=1) — legacy behaviour, which lets the
    1x1 mixer absorb the learnable Q in the DWT.

    grouped_mix=True: the shared mixer is groups=8 (per-subband), so the
    learnable Q in the DWT is the ONLY cross-subband mixer and becomes
    load-bearing. Requires out_channels % 8 == 0.

    per_channel=True: each input channel (e.g. each MRI modality) gets its own
    8 -> out_channels/in_channels subband mixer, then results are concatenated.
    """

    def __init__(self, in_channels: int, out_channels: int, learnable: bool = True,
                 mode: str = "theta", per_channel: bool = False,
                 per_channel_wavelet: bool = False, grouped_mix: bool = False):
        super().__init__()
        self.per_channel = per_channel
        if per_channel:
            assert out_channels % in_channels == 0, \
                "per_channel requires out_channels to be divisible by in_channels"
            out_c = out_channels // in_channels
            if per_channel_wavelet:
                self.waves = nn.ModuleList(
                    [HaarWavelet3D(learnable, mode) for _ in range(in_channels)])
            else:
                self.wave = HaarWavelet3D(learnable, mode)
            self.mixes = nn.ModuleList(
                [nn.Conv3d(8, out_c, kernel_size=1) for _ in range(in_channels)])
        else:
            assert not per_channel_wavelet, \
                "per_channel_wavelet requires per_channel=True"
            self.wave = HaarWavelet3D(learnable, mode)
            if grouped_mix:
                # groups=8 so Q is not absorbed by the mixer.
                assert out_channels % 8 == 0, \
                    "groups=8 subband mixer requires out_channels divisible by 8"
                groups = 8
            else:
                groups = 1
            self.mix = nn.Conv3d(8 * in_channels, out_channels, kernel_size=1,
                                 groups=groups)

    def forward(self, x):
        if not self.per_channel:
            return self.mix(self.wave.dwt(x))
        outs = []
        for c in range(x.shape[1]):
            xc = x[:, c:c + 1]
            wave = self.waves[c] if hasattr(self, "waves") else self.wave
            outs.append(self.mixes[c](wave.dwt(xc)))
        return torch.cat(outs, dim=1)


class HaarUpsample(nn.Module):
    """Replacement for ConvTranspose3d(k=2, s=2) upsampling: 1x1 subband
    expansion + 3D Haar IDWT. Pure operator (no skip-concatenation or fusion),
    so it drops into any backbone's decoder the same way a transposed conv does.

    (B, C_in, d, h, w) -> (B, C_out, 2d, 2h, 2w)
    """

    def __init__(self, in_channels: int, out_channels: int, learnable: bool = True,
                 mode: str = "theta"):
        super().__init__()
        self.wave = HaarWavelet3D(learnable, mode)
        self.expand = nn.Conv3d(in_channels, 8 * out_channels, kernel_size=1)

    def forward(self, x):
        return self.wave.idwt(self.expand(x))


class WaveletConv3d(nn.Module):
    """Replacement for an equal-channel 3x3x3 conv: DWT -> per-subband 1x1
    channel mixing -> IDWT. Already groups=8 in the original (Q is load-bearing
    here too); left unchanged.
    """

    def __init__(self, channels: int, learnable: bool = True, mode: str = "theta"):
        super().__init__()
        self.wave = HaarWavelet3D(learnable, mode)
        self.mix = nn.Conv3d(8 * channels, 8 * channels, kernel_size=1, groups=8)

    def forward(self, x):
        return self.wave.idwt(self.mix(self.wave.dwt(x)))
