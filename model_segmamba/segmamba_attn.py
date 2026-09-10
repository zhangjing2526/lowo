# SegAttn-Haar: attention backbone with the SAME Haar wavelet operators and
# the SAME UNet decoder as SegMambaHaar, but with the per-stage sequence mixer
# swapped from Mamba (SSM scan) to shifted-window multi-head self-attention
# (Swin-style). This is the "attention" arm of the LOWO cross-architecture
# ablation: only the sequence-mixing block changes, everything else (Haar
# stem/downsample, wavelet GSC, skip connections, Haar up-sampling, deep
# supervision) is byte-identical to SegMambaHaar.
#
# This file does NOT import mamba_ssm. It depends only on:
#   - lowo.py          (wavelet operators)
#   - haar_blocks.py   (MlpChannel / WaveletGSC / HaarUpBlock)
#   - MONAI            (SwinTransformerBlock + UnetrBasicBlock + UnetOutBlock)

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _grad_checkpoint

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock
from monai.networks.nets.swin_unetr import (
    SwinTransformerBlock,
    compute_mask,
    get_window_size,
)

from model_segmamba.lowo import HaarDownsample
from model_segmamba.haar_blocks import MlpChannel, WaveletGSC, HaarUpBlock


class WindowAttnLayer(nn.Module):
    """Drop-in replacement for MambaLayer: shifted-window self-attention.

    Same contract as MambaLayer: (B, C, D, H, W) -> (B, C, D, H, W), with the
    residual applied inside the block. Internally wraps MONAI's
    SwinTransformerBlock (pre-norm window attention + FFN), i.e. the standard
    attention block used by SwinUNETR, so reviewers get a familiar, well-tuned
    attention baseline rather than a bespoke one.

    Args:
        dim: feature channels (must be divisible by num_heads).
        num_heads: attention heads per block; defaults to dim // 16 (head_dim
            = 16), matching MONAI SwinUNETR's default for 48/96/192/384 dims.
        window_size: per-axis local window size.
        mlp_ratio: FFN hidden dim ratio.
        drop_path: stochastic depth rate for this block.
        use_checkpoint: gradient-checkpoint the attention sub-block.
    """

    def __init__(self, dim, num_heads=None, window_size=(7, 7, 7),
                 mlp_ratio=4.0, qkv_bias=True, drop_path=0.0,
                 use_checkpoint=False, use_shift=True):
        super().__init__()
        if num_heads is None:
            assert dim % 16 == 0, f"dim {dim} not divisible by 16; pass num_heads explicitly"
            num_heads = dim // 16
        self.window_size = tuple(window_size)
        # use_shift=False 时关闭 shifted-window，等价于 shift_size 全 0：
        # 没有 roll，也不产生跨边界 mask（Swin 退化为纯窗口注意力）。
        self.use_shift = use_shift
        self.shift_size = (tuple(ws // 2 for ws in self.window_size)
                           if use_shift else (0,) * len(self.window_size))
        self.block = SwinTransformerBlock(
            dim=dim,
            num_heads=num_heads,
            window_size=self.window_size,
            shift_size=self.shift_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=0.0,
            attn_drop=0.0,
            drop_path=drop_path,
            act_layer="GELU",
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
        )

    def forward(self, x):
        # x: (B, C, D, H, W) channels-first -> Swin block wants (B, D, H, W, C)
        d, h, w = x.shape[2:]
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        x_t = x.permute(0, 2, 3, 4, 1).contiguous()

        if any(s > 0 for s in shift_size):
            # 只有真正有 shift 时才预计算跨边界 mask；shift=0 时 block 内部
            # 走 attn_mask=None 分支，这里也跳过生成，省掉这一笔开销。
            dp = math.ceil(d / window_size[0]) * window_size[0]
            hp = math.ceil(h / window_size[1]) * window_size[1]
            wp = math.ceil(w / window_size[2]) * window_size[2]
            attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device)
        else:
            attn_mask = None

        x_t = self.block(x_t, attn_mask)  # block applies its own residual
        return x_t.permute(0, 4, 1, 2, 3).contiguous()


class RoutedWindowAttn(nn.Module):
    """Routing wrapper around :class:`WindowAttnLayer` — **soft supernet probe**.

    Two semantics, deliberately kept distinct:

    **Training (``gate_hardness`` controls the blend)**::

        g   = sigmoid(Linear(token_features) + rf_prior)
        # soft  : out = g*Attn(x) + (1-g)*(x + MlpChannel(x))        (supernet)
        # ste   : out = hard_mask*Attn + (1-mask)*(x+Mixer)  fwd,
        #         soft gradient bwd  (straight-through estimator)
        # gumbel: out = GumbelSoftmax([g,-g]; tau)[0]*Attn
        #              + [...] [1]*(x+Mixer)   (tau annealed -> hard)

    The default ``"soft"`` mode is the supernet probe: both branches are
    computed and softly blended, so it does NOT save FLOPs (slightly more
    expensive than plain WindowAttnLayer). ``"ste"`` and ``"gumbel"`` instead
    train the net to be robust *under* hard routing — the forward sees a
    discrete branch choice while gradients still reach both branches — which
    shrinks the soft->hard Dice gap measured at eval. All three answer the
    probe questions; the harder modes are the v0.5->v1.0 upgrade path.

    The mixer branch carries a residual (``x + mixer(x)``) so that it is
    structurally symmetric with the attention branch — SwinTransformerBlock
    already applies its own residual internally — otherwise a gate sliding
    toward 0 would jump the representation from a residual stream to a bare
    MLP output and contaminate the probe.

    **Evaluation with ``hard_routing=True`` (conditional compute)**: the
    block-level mean gate decides ONE branch; the unselected branch is not
    computed at all (real FLOPs/latency saving, measurable per block). This
    is the deploy-time semantics the paper narrative eventually needs. Note
    per-token dispatch is NOT implemented: window attention's atom is the 7^3
    window, so token-level hard routing cannot skip compute without breaking
    window structure — per-window dispatch is the production upgrade.

    ``rf_prior`` is a per-stage learnable scalar injecting the
    receptive-field prior: deeper stages (larger Haar downsampling -> local
    multi-scale context already in the token) are initialised negative (bias
    toward the cheap mixer), shallow stages positive (toward attention).

    Stats bookkeeping keeps everything as detached 0-dim GPU tensors (no
    ``.item()`` -> no host sync per forward); ``collect_gate_stats`` converts
    to floats at the caller's chosen interval.
    """

    def __init__(self, dim, num_heads=None, window_size=(7, 7, 7),
                 mlp_ratio=4.0, qkv_bias=True, drop_path=0.0,
                 use_checkpoint=False, stage_idx=0, n_stages=4,
                 rf_prior_scale=1.5, router_hidden=None, block_idx=0,
                 routing_threshold=0.5, use_shift=True,
                 gate_hardness="soft", gumbel_tau_start=1.0,
                 gumbel_tau_end=0.05, gumbel_tau_epochs=300,
                 ste_warmup_epochs=0):
        super().__init__()
        self.attn = WindowAttnLayer(dim, num_heads=num_heads,
                                    window_size=window_size, mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, drop_path=drop_path,
                                    use_checkpoint=use_checkpoint,
                                    use_shift=use_shift)
        # cheap channel-mixing alternative (1x1 conv -> GELU -> 1x1 conv)
        self.mixer = MlpChannel(dim, int(dim * mlp_ratio))
        # per-token gate head: token feature (C) -> scalar logit
        if router_hidden is not None:
            self.gate_fc = nn.Sequential(
                nn.Linear(dim, router_hidden), nn.GELU(), nn.Linear(router_hidden, 1))
        else:
            self.gate_fc = nn.Linear(dim, 1)
        # receptive-field prior: +scale at stage0 -> -scale at last stage
        frac = 0.0 if n_stages <= 1 else stage_idx / (n_stages - 1)
        init = (1.0 - 2.0 * frac) * rf_prior_scale
        self.rf_prior = nn.Parameter(torch.tensor(float(init)))
        self.stage_idx = stage_idx
        self.block_idx = block_idx
        # deploy-time conditional compute (off during soft training)
        self.hard_routing = False
        self.routing_threshold = routing_threshold
        # cached hard-routing decision. For fixed-input benchmarks (same x every
        # timed iteration) a per-block gate.mean()>thr would otherwise force a
        # GPU->CPU sync on EVERY forward, inflating the hard-mode latency and
        # under-reporting the real conditional-compute speedup. The decision is
        # computed once (first hard forward), cached, and reused for subsequent
        # forwards on the SAME input. Call clear_hard_cache() when x changes.
        self._cached_hard_decision = None
        # ── gate training hardness (default "soft" = current supernet blend) ──
        #   "soft"  : g*Attn + (1-g)*Mixer  (supernet, no FLOPs saving)
        #   "ste"   : forward uses hard mask, backward passes soft gradient
        #             (straight-through estimator) so the net is robust under
        #             hard routing; STE only after ste_warmup_epochs
        #   "gumbel": 2-class Gumbel-Softmax over [Attn, Mixer], temperature
        #             tau annealed start->end across gumbel_tau_epochs (then ~hard)
        self.gate_hardness = gate_hardness
        self.gumbel_tau_start = float(gumbel_tau_start)
        self.gumbel_tau_end = float(gumbel_tau_end)
        self.gumbel_tau_epochs = int(gumbel_tau_epochs)
        self.ste_warmup_epochs = int(ste_warmup_epochs)
        self._epoch = 0
        self.tau = float(gumbel_tau_start)
        self._stats = {}

    def forward(self, x):
        b, c, d, h, w = x.shape
        xf = x.permute(0, 2, 3, 4, 1).reshape(-1, c)            # (N, C)
        logit = self.gate_fc(xf).reshape(b, d, h, w, 1).permute(0, 4, 1, 2, 3)
        logit = logit + self.rf_prior                            # broadcast (B,1,D,H,W)
        gate = torch.sigmoid(logit)                              # prob(attention)

        use_attn = None
        if self.hard_routing and not self.training:
            # ── conditional compute: block-level decision, skip the loser ──
            # Decision is computed once and cached; for fixed-input benchmarks
            # this removes the per-forward GPU->CPU sync that bool() would cause
            # (see __init__ doc). clear_hard_cache() must be called when x changes.
            if self._cached_hard_decision is None:
                self._cached_hard_decision = bool(gate.mean() > self.routing_threshold)
            use_attn = self._cached_hard_decision
            out = self.attn(x) if use_attn else x + self.mixer(x)
        else:
            # both branches computed (training cost identical to soft; only
            # eval hard routing saves FLOPs — see docstring)
            a = self.attn(x)                                     # residual inside block
            m = x + self.mixer(x)                                # residual mixer (symmetric)
            if self.training and self.gate_hardness == "ste" \
                    and self._epoch >= self.ste_warmup_epochs:
                # ── STE: forward = hard mask, backward = soft gradient ──
                # out_hard = selected branch; we add it via a stop-gradient
                # residual so the forward value is hard but the gradient flows
                # through the soft blend (out_soft). This trains the net to be
                # robust under hard routing, shrinking the soft->hard Dice gap.
                out_soft = gate * a + (1.0 - gate) * m
                mask_h = (gate > 0.5).float()
                out_hard = mask_h * a + (1.0 - mask_h) * m
                out = out_soft + (out_hard - out_soft).detach()
            elif self.training and self.gate_hardness == "gumbel":
                # ── Gumbel-Softmax: differentiable 2-way sample over
                #    [Attn, Mixer]; tau annealed to ~0 -> near one-hot (hard) ──
                # logit is (B,1,D,H,W); squeeze the channel dim before stacking
                # so the 2-class axis lands at dim=1 -> (B,2,D,H,W), not 6D.
                lg = logit.squeeze(1)                            # (B,D,H,W)
                logits2 = torch.stack([lg, -lg], dim=1)           # (B,2,D,H,W)
                w = F.gumbel_softmax(logits2, tau=self.tau,
                                     hard=False, dim=1)          # [w_a, w_m]
                out = w[:, 0:1] * a + w[:, 1:2] * m
            else:
                # ── soft supernet blend (default; also eval non-hard) ──
                out = gate * a + (1.0 - gate) * m

        # stats: detached 0-dim tensors only — NO host sync here
        with torch.no_grad():
            g = gate.float()
            self._stats = {
                "mean": g.mean(),
                "std": g.std() if g.numel() > 1 else g.new_zeros(()),
                "attn_frac": (g > 0.5).to(g.dtype).mean(),
                "rf_prior": self.rf_prior.detach().float(),
            }
            if use_attn is not None:
                self._stats["use_attn"] = float(use_attn)
        return out

    def gate_stats(self):
        return self._stats

    def clear_hard_cache(self):
        """Invalidate the cached hard-routing decision.

        Must be called if the input tensor changes identity/content between
        hard-routing forwards (e.g. sliding-window inference over distinct
        patches, or a new batch). For the fixed-patch latency harness the cache
        is what removes the per-forward GPU->CPU sync.
        """
        self._cached_hard_decision = None

    def set_epoch(self, epoch: int):
        """Advance the training-epoch counter and anneal Gumbel temperature.

        Drives ``tau`` linearly from ``gumbel_tau_start`` to ``gumbel_tau_end``
        across ``gumbel_tau_epochs``; beyond that it stays at the end value
        (near one-hot, i.e. effectively hard routing). No-op for "soft"/"ste".
        Called once per epoch by the trainer via :meth:`SegAttnHaar.set_gate_epoch`.
        """
        self._epoch = int(epoch)
        if self.gate_hardness == "gumbel" and self.gumbel_tau_epochs > 0:
            frac = min(1.0, self._epoch / self.gumbel_tau_epochs)
            self.tau = max(self.gumbel_tau_end,
                           self.gumbel_tau_start * (1.0 - frac)
                           + self.gumbel_tau_end * frac)
        elif self.gate_hardness == "gumbel":
            self.tau = self.gumbel_tau_end


class AttnEncoderHaar(nn.Module):
    """MambaEncoderHaar with Mamba stages replaced by windowed attention.

    Structure is identical to MambaEncoderHaar: Haar stem -> [WaveletGSC ->
    N x sequence-mixer block] x4, with per-stage InstanceNorm + MlpChannel
    output projections. The ONLY change is MambaLayer -> WindowAttnLayer.
    """

    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6,
                 out_indices=[0, 1, 2, 3], learnable_wavelet=True,
                 gsc_checkpoint=False, wavelet_mode="theta",
                 per_channel_stem=False, per_channel_wavelet=False,
                 num_heads=None, window_size=(7, 7, 7),
                 attn_checkpoint=False, use_routing=False,
                 rf_prior_scale=1.5, router_hidden=None,
                 use_shift=True, gate_hardness="soft",
                 gumbel_tau_start=1.0, gumbel_tau_end=0.05,
                 gumbel_tau_epochs=300, ste_warmup_epochs=0,
                 grouped_mix=False):
        super().__init__()
        self.gsc_checkpoint = gsc_checkpoint

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            HaarDownsample(in_chans, dims[0], learnable=learnable_wavelet,
                           mode=wavelet_mode,
                           per_channel=per_channel_stem,
                           per_channel_wavelet=per_channel_wavelet,
                           grouped_mix=grouped_mix),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.InstanceNorm3d(dims[i]),
                HaarDownsample(dims[i], dims[i + 1], learnable=learnable_wavelet,
                               mode=wavelet_mode, grouped_mix=grouped_mix),
            )
            self.downsample_layers.append(downsample_layer)

        if num_heads is None:
            num_heads = [d // 16 for d in dims]
        # stochastic depth schedule (off by default, matching the Mamba arm
        # which ignores drop_path_rate at its default of 0).
        total = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total)] if drop_path_rate > 0 else [0.0] * total

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        self.use_routing = use_routing
        cur = 0
        for i in range(4):
            gsc = WaveletGSC(dims[i], learnable=learnable_wavelet,
                             mode=wavelet_mode)

            if use_routing:
                blocks = [
                    RoutedWindowAttn(dim=dims[i], num_heads=num_heads[i],
                                     window_size=window_size,
                                     drop_path=dpr[cur + j],
                                     use_checkpoint=attn_checkpoint,
                                     stage_idx=i, n_stages=4,
                                     rf_prior_scale=rf_prior_scale,
                                     router_hidden=router_hidden,
                                     block_idx=j,
                                     use_shift=use_shift,
                                     gate_hardness=gate_hardness,
                                     gumbel_tau_start=gumbel_tau_start,
                                     gumbel_tau_end=gumbel_tau_end,
                                     gumbel_tau_epochs=gumbel_tau_epochs,
                                     ste_warmup_epochs=ste_warmup_epochs)
                    for j in range(depths[i])
                ]
                stage = nn.Sequential(*blocks)
            else:
                stage = nn.Sequential(
                    *[WindowAttnLayer(dim=dims[i], num_heads=num_heads[i],
                                      window_size=window_size,
                                      drop_path=dpr[cur + j],
                                      use_checkpoint=attn_checkpoint,
                                      use_shift=use_shift)
                      for j in range(depths[i])]
                )

            self.stages.append(stage)
            self.gscs.append(gsc)
            cur += depths[i]

        # registry for gate-stat collection (empty when routing disabled)
        self.routed_modules = (
            [m for m in self.modules() if isinstance(m, RoutedWindowAttn)]
            if use_routing else []
        )

        self.out_indices = out_indices

        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            if self.gsc_checkpoint and self.training and x.requires_grad:
                x = _grad_checkpoint(self.gscs[i], x, use_reentrant=False)
            else:
                x = self.gscs[i](x)
            x = self.stages[i](x)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x


class SegAttnHaar(nn.Module):
    """Attention backbone with Haar wavelet operators, mirroring SegMambaHaar.

    The encoder uses WindowAttnLayer instead of MambaLayer; the decoder, skip
    connections, Haar up-sampling and deep-supervision heads are identical to
    SegMambaHaar so the two arms differ ONLY in the sequence-mixing block.
    """

    def __init__(
        self,
        in_chans=1,
        out_chans=13,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name="instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims=3,
        learnable_wavelet: bool = True,
        gsc_checkpoint: bool = False,
        wavelet_mode: str = "theta",
        per_channel_stem: bool = False,
        per_channel_wavelet: bool = False,
        num_heads=None,
        window_size=(7, 7, 7),
        attn_checkpoint: bool = False,
        use_routing: bool = False,
        rf_prior_scale: float = 1.5,
        router_hidden=None,
        use_shift: bool = True,
        gate_hardness: str = "soft",
        gumbel_tau_start: float = 1.0,
        gumbel_tau_end: float = 0.05,
        gumbel_tau_epochs: int = 300,
        ste_warmup_epochs: int = 0,
        grouped_mix: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value

        self.spatial_dims = spatial_dims
        self.vit = AttnEncoderHaar(in_chans,
                                   depths=depths,
                                   dims=feat_size,
                                   drop_path_rate=drop_path_rate,
                                   layer_scale_init_value=layer_scale_init_value,
                                   learnable_wavelet=learnable_wavelet,
                                   gsc_checkpoint=gsc_checkpoint,
                                   wavelet_mode=wavelet_mode,
                                   per_channel_stem=per_channel_stem,
                                   per_channel_wavelet=per_channel_wavelet,
                                   num_heads=num_heads,
                                   window_size=window_size,
                                   attn_checkpoint=attn_checkpoint,
                                   use_routing=use_routing,
                                   rf_prior_scale=rf_prior_scale,
                                   router_hidden=router_hidden,
                                   use_shift=use_shift,
                                   gate_hardness=gate_hardness,
                                   gumbel_tau_start=gumbel_tau_start,
                                   gumbel_tau_end=gumbel_tau_end,
                                   gumbel_tau_epochs=gumbel_tau_epochs,
                                   ste_warmup_epochs=ste_warmup_epochs,
                                   grouped_mix=grouped_mix,
                                   )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = HaarUpBlock(self.hidden_size, self.feat_size[3],
                                    norm_name=norm_name, learnable=learnable_wavelet,
                                    mode=wavelet_mode)
        self.decoder4 = HaarUpBlock(self.feat_size[3], self.feat_size[2],
                                    norm_name=norm_name, learnable=learnable_wavelet,
                                    mode=wavelet_mode)
        self.decoder3 = HaarUpBlock(self.feat_size[2], self.feat_size[1],
                                    norm_name=norm_name, learnable=learnable_wavelet,
                                    mode=wavelet_mode)
        self.decoder2 = HaarUpBlock(self.feat_size[1], self.feat_size[0],
                                    norm_name=norm_name, learnable=learnable_wavelet,
                                    mode=wavelet_mode)
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims,
                                in_channels=self.feat_size[0],
                                out_channels=self.out_chans)
        # deep-supervision auxiliary heads (identical to SegMambaHaar).
        self.ds_out3 = UnetOutBlock(spatial_dims=spatial_dims,
                                    in_channels=self.feat_size[3],
                                    out_channels=self.out_chans)
        self.ds_out2 = UnetOutBlock(spatial_dims=spatial_dims,
                                    in_channels=self.feat_size[2],
                                    out_channels=self.out_chans)
        self.ds_out1 = UnetOutBlock(spatial_dims=spatial_dims,
                                    in_channels=self.feat_size[1],
                                    out_channels=self.out_chans)

    def forward(self, x_in, deep_supervision: bool = False,
                 return_gate_stats: bool = False):
        outs = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = outs[0]
        enc2 = self.encoder2(x2)
        x3 = outs[1]
        enc3 = self.encoder3(x3)
        x4 = outs[2]
        enc4 = self.encoder4(x4)
        enc_hidden = self.encoder5(outs[3])
        dec3 = self.decoder5(enc_hidden, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0)

        logits = self.out(out)
        if deep_supervision:
            aux = [self.ds_out3(dec3), self.ds_out2(dec2), self.ds_out1(dec1)]
        else:
            aux = None
        if return_gate_stats:
            return logits, aux, self.collect_gate_stats()
        if deep_supervision:
            return logits, aux
        return logits

    def collect_gate_stats(self):
        """Aggregate per-block gate stats from all RoutedWindowAttn modules.

        Converts the detached 0-dim GPU tensors to python floats — the only
        place a host sync happens, so call this at a chosen interval (e.g.
        every N training steps), not every forward.
        """
        stats = {}
        for m in self.vit.routed_modules:
            s = m.gate_stats()
            if s:
                entry = {k: float(v) for k, v in s.items()}
                stats[f"stage{m.stage_idx}_blk{m.block_idx}"] = entry
        return stats

    def set_hard_routing(self, enabled: bool = True, threshold: float = 0.5):
        """Switch routed blocks to conditional-compute semantics (eval only).

        With hard routing ON and the module in eval() mode, each block takes a
        block-level gate decision and computes ONLY the selected branch —
        real FLOPs/latency saving. Use after soft training, e.g. to measure
        the subnet Dice vs the soft supernet Dice.

        When ``enabled`` we ALSO invalidate any cached hard-routing decision
        (``clear_hard_cache``). The decision is computed from the block-level
        gate mean on the *first* hard forward and then reused for subsequent
        forwards on the SAME tensor — that cache is what removes the per-forward
        GPU->CPU sync for the latency harness. But for distinct inputs (every
        validation sample, every sliding-window patch) the cache MUST be dropped
        so routing is recomputed per-input; otherwise all samples would inherit
        the first sample's routing pattern and the hard Dice/HD95/skip metrics
        would be silently wrong. Clearing on enable (rather than only at the
        call site) makes every caller correct by construction.
        """
        for m in self.vit.routed_modules:
            m.hard_routing = enabled
            m.routing_threshold = threshold
            if enabled:
                m.clear_hard_cache()

    def clear_hard_cache(self):
        """Invalidate cached hard-routing decisions on all routed blocks.

        Must be called between distinct input tensors (e.g. sliding-window
        patches, new batch) so routing is recomputed per-input. The latency
        harness relies on the cache staying valid for repeated forwards over
        the *same* tensor — this is what removes the per-forward GPU→CPU sync.

        No-op on models without routing (empty ``routed_modules``).
        """
        for m in self.vit.routed_modules:
            m.clear_hard_cache()

    def set_gate_epoch(self, epoch: int):
        """Advance gate-training hardness state (Gumbel tau anneal) per epoch.

        No-op unless ``gate_hardness`` is "gumbel". Safe to call on models
        without routing (empty ``routed_modules``).
        """
        for m in self.vit.routed_modules:
            m.set_epoch(epoch)
