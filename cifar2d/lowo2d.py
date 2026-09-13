# 2D LOWO: drop-in Haar / SO(4) resampler for CIFAR (and any even-extent 2D map).
#
# Same algebra as the 3D operator, one axis dropped:
#   K(θ, A) = Q(A) K_haar(θ),  Q = exp(A - A^T) ∈ SO(4)   (6 DOF)
#   K_haar  = F_H(θ0) ⊗ F_W(θ1) ∈ R^{4×4}
# Subband-major layout, index s*C + c.  t = π/4 is Haar.
# groups=4 on the 1×1 mixer makes Q identifiable; groups=1 absorbs it.

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarWavelet2D(nn.Module):
    """Axis-separable orthogonal 2-tap wavelet, Haar-initialized.

    dwt : (B, C, H, W)  -> (B, 4C, H/2, W/2)
    idwt: (B, 4C, h, w) -> (B, C, 2h, 2w)

    Subband order (H major, W minor): 0=LL, 1=LH, 2=HL, 3=HH.
    """

    def __init__(self, learnable: bool = True, mode: str = "so4"):
        super().__init__()
        assert mode in ("theta", "so4"), mode
        self.mode = mode
        theta = torch.full((2,), math.pi / 4)
        if learnable and mode == "theta":
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer("theta", theta)
        if mode == "so4":
            skew = torch.zeros(4, 4)
            if learnable:
                self.skew = nn.Parameter(skew)
            else:
                self.register_buffer("skew", skew)
        if not learnable:
            self.register_buffer("_k_cache", self._kernel())

    def _kernel(self) -> torch.Tensor:
        c, s = torch.cos(self.theta), torch.sin(self.theta)
        f = torch.stack(
            [torch.stack([c, s], dim=-1), torch.stack([-s, c], dim=-1)],
            dim=-2,
        )  # (2, 2, 2)
        k = torch.einsum("ah,bw->abhw", f[0], f[1]).reshape(4, 4)
        if self.mode == "so4":
            q = torch.matrix_exp(self.skew - self.skew.transpose(0, 1))
            k = q @ k
        return k.reshape(4, 1, 2, 2)

    def _get_kernel(self) -> torch.Tensor:
        if hasattr(self, "_k_cache"):
            return self._k_cache
        return self._kernel()

    @torch.no_grad()
    def ortho_dev(self) -> float:
        if self.mode != "so4":
            return 0.0
        q = torch.matrix_exp(self.skew - self.skew.transpose(0, 1))
        return (q - torch.eye(4, device=q.device, dtype=q.dtype)).norm().item()

    def dwt(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        k = self._get_kernel()
        y = F.conv2d(x.reshape(b * c, 1, *x.shape[2:]), k, stride=2)
        y = y.reshape(b, c, 4, *y.shape[2:]).permute(0, 2, 1, 3, 4)
        return y.reshape(b, 4 * c, *y.shape[3:])

    def idwt(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        c = x.shape[1] // 4
        y = x.reshape(b, 4, c, *x.shape[2:]).permute(0, 2, 1, 3, 4)
        y = y.reshape(b * c, 4, *y.shape[3:])
        y = F.conv_transpose2d(y, self._get_kernel(), stride=2)
        return y.reshape(b, c, *y.shape[2:])


class HaarDownsample2D(nn.Module):
    """DWT + 1×1 subband mixer.  grouped_mix=True → groups=4, Q is load-bearing."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        learnable: bool = True,
        mode: str = "so4",
        grouped_mix: bool = True,
    ):
        super().__init__()
        self.wave = HaarWavelet2D(learnable=learnable, mode=mode)
        if grouped_mix:
            assert out_channels % 4 == 0, "groups=4 needs out_channels % 4 == 0"
            groups = 4
        else:
            groups = 1
        self.mix = nn.Conv2d(
            4 * in_channels, out_channels, kernel_size=1, groups=groups, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mix(self.wave.dwt(x))


class HWDOfficial2D(HaarDownsample2D):
    """Official HWD mixer (Xu et al. 2023; apple1986/HWD Down_wt).

    Haar DWT then a dense Conv2d(4C, Cout, 1).  BN/ReLU stay in the
    ResNet block, as they do after Down_wt's linear 1x1.  Any Q on
    this DWT has rho=0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        learnable: bool = False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            learnable=learnable,
            mode="so4",
            grouped_mix=False,
        )
