# SegMamba-Haar: SegMamba with 3D Haar wavelet operators replacing spatial
# convolutions. The wavelet operators themselves live in lowo.py (backbone-
# agnostic, no mamba/attention/MONAI dependency); the shared UNet blocks live
# in haar_blocks.py. This file only contains the Mamba-specific encoder and
# the full SegMambaHaar network.
#
# Strict "replacement" (not an add-on branch) of:
#   1. stem Conv3d(k=7, s=2)            -> 3D Haar DWT + 1x1 subband-mixing conv
#   2. downsample Conv3d(k=2, s=2)      -> 3D Haar DWT + 1x1 subband-mixing conv
#   3. GSC 3x3 convs                    -> wavelet-domain conv (DWT -> grouped 1x1 -> IDWT)
#   4. decoder ConvTranspose3d(k=2,s=2) -> 1x1 subband-expansion conv + 3D Haar IDWT

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _grad_checkpoint

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock

from model_segmamba.segmamba import MambaLayer
from model_segmamba.lowo import HaarDownsample, HaarWavelet3D  # HaarWavelet3D re-exported for 3_train script
from model_segmamba.haar_blocks import MlpChannel, WaveletGSC, HaarUpBlock


class MambaEncoderHaar(nn.Module):
    """MambaEncoder with Haar-DWT downsampling and wavelet-domain GSC."""

    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6,
                 out_indices=[0, 1, 2, 3], learnable_wavelet=True,
                 gsc_checkpoint=False, wavelet_mode="theta",
                 per_channel_stem=False, per_channel_wavelet=False,
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

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        num_slices_list = [64, 32, 16, 8]
        cur = 0
        for i in range(4):
            gsc = WaveletGSC(dims[i], learnable=learnable_wavelet,
                             mode=wavelet_mode)

            stage = nn.Sequential(
                *[MambaLayer(dim=dims[i], num_slices=num_slices_list[i]) for j in range(depths[i])]
            )

            self.stages.append(stage)
            self.gscs.append(gsc)
            cur += depths[i]

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
                # identical math, trades recompute for activation memory
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


class SegMambaHaar(nn.Module):
    """SegMamba with all spatial down/up-sampling and GSC spatial convs
    replaced by (learnable) 3D Haar wavelet operators.

    Set learnable_wavelet=False for the fixed-Haar ablation.
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
        self.vit = MambaEncoderHaar(in_chans,
                                    depths=depths,
                                    dims=feat_size,
                                    drop_path_rate=drop_path_rate,
                                    layer_scale_init_value=layer_scale_init_value,
                                    learnable_wavelet=learnable_wavelet,
                                    gsc_checkpoint=gsc_checkpoint,
                                    wavelet_mode=wavelet_mode,
                                    per_channel_stem=per_channel_stem,
                                    per_channel_wavelet=per_channel_wavelet,
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
        # deep-supervision auxiliary heads on the three deeper decoder levels
        # (dec3=16^3, dec2=32^3, dec1=64^3). The trainer upsamples them to full
        # res and applies the same region loss; disabled at inference.
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
                 return_gate_stats: bool = False, **kwargs):
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
        # SegMambaHaar has no routing gates; return None for gate_stats so the
        # trainer's return_gate_stats path stays compatible (mirrors SegAttnHaar,
        # which returns real stats). No behaviour change when False.
        if return_gate_stats:
            return logits, aux, None
        if deep_supervision:
            return logits, aux
        return logits
