"""
Unified training script for SegMamba variants on BraTS2021.

Model arms are selected via the `model` field in a YAML config:
  segmamba         – original SegMamba  (Mamba SSM per-stage mixer)
  segmamba_haar    – SegMamba-Haar      (Haar DWT/IDWT + Mamba SSM mixer)
  segattn_haar     – SegAttn-Haar       (Haar DWT/IDWT + Window-Attn mixer)
  unetr            – MONAI UNETR        (isotropic ViT, mixer_from=12)
  unetr_deepmix    – UNETR, deep ViT blocks use a cheap mixer (no QK^T)

Each run is driven by a YAML config file; the script also saves a copy
of the YAML itself into the log directory so you can always look back at
exactly how a run was configured.

BraTS2021 label mapping (after preprocessing remapping):
  0: background
  1: NCR (Necrotic/Non-Enhancing Tumor Core)
  2: ED (Peritumoral Edema)
  3: ET (Enhancing Tumor) – originally 4 in BraTS2021, remapped to 3

Evaluation regions:
  TC (Tumor Core): labels 1, 3
  WT (Whole Tumor): labels 1, 2, 3
  ET (Enhancing Tumor): label 3

Usage:
    python train_brats2021.py --config configs/B0_segmamba.yaml
    python train_brats2021.py --config configs/A2_fixed_g8.yaml
    python train_brats2021.py --config configs/A6_so8_g8.yaml
    python train_brats2021.py --config configs/C1_segattn_so8_g1.yaml
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.metrics import HausdorffDistanceMetric
from light_training.utils.files_helper import save_new_model_and_delete_last
from monai.inferers import SlidingWindowInferer
from monai.losses import TverskyLoss
from monai.utils import set_determinism

# Seed is applied in __main__ from --seed (default 123) so that runs can be
# repeated with different seeds (e.g. 3 seeds for mean±std). The train/val/test
# partition is the frozen internal held-out split in splits/ (not glob+shuffle
# with seed 42). Unsorted glob+shuffle is not portable across machines.

# ═══════════════════════════════════════════════════════════════════════
# Constants (same for all variants)
# ═══════════════════════════════════════════════════════════════════════

data_dir = "./data/fullres/train"
env = "pytorch"

# ═══════════════════════════════════════════════════════════════════════
# Config loading
# ═══════════════════════════════════════════════════════════════════════

# Sensible defaults – only used when a key is missing from the YAML.
_CFG_DEFAULTS = dict(
    model="segmamba",
    wavelet_mode=None,
    learnable_wavelet=None,
    gsc_checkpoint=None,
    per_channel_stem=None,
    per_channel_wavelet=None,
    wavelet_boundary_weight=0.0,
    window_size=None,
    num_heads=None,
    attn_checkpoint=None,
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    use_dice_ce=True,
    deep_supervision=True,
    batch_size=2,
    max_epoch=500,
    roi_size=[128, 128, 128],
    val_every=2,
    device="cuda:0",
    resume=True,
    master_port=17759,
    train_process=18,
)


def load_config(path: str) -> dict:
    """Load YAML config, filling missing keys with defaults.

    Also converts list → tuple for ``window_size`` / ``roi_size`` so that
    model constructors receive the expected types.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    for k, v in _CFG_DEFAULTS.items():
        cfg.setdefault(k, v)
    # list → tuple for fields that expect tuples
    if isinstance(cfg.get("window_size"), list):
        cfg["window_size"] = tuple(cfg["window_size"])
    if isinstance(cfg.get("roi_size"), list):
        cfg["roi_size"] = list(cfg["roi_size"])       # keep as list (Trainer uses list)
    return cfg

# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _safe_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "nogit"


def _fixed_haar_kernel(device, dtype):
    """Fixed (8,1,2,2,2) orthonormal 3D Haar analysis kernel (theta = pi/4).
    Subband 0 = LLL, 1..7 = high-frequency subbands."""
    f = torch.tensor([[1.0, 1.0], [-1.0, 1.0]], device=device, dtype=dtype) / math.sqrt(2.0)
    k = torch.einsum('ad,bh,cw->abcdhw', f, f, f)
    return k.reshape(8, 1, 2, 2, 2)


def _build_model(cfg):
    """Instantiate the model from config."""
    if cfg["model"] == "segmamba":
        from model_segmamba.segmamba import SegMamba
        return SegMamba(in_chans=4, out_chans=4,
                        depths=[2, 2, 2, 2], feat_size=[48, 96, 192, 384])
    elif cfg["model"] == "segmamba_haar":
        from model_segmamba.segmamba_haar import SegMambaHaar
        return SegMambaHaar(in_chans=4, out_chans=4,
                            depths=[2, 2, 2, 2], feat_size=[48, 96, 192, 384],
                            learnable_wavelet=cfg["learnable_wavelet"],
                            gsc_checkpoint=cfg["gsc_checkpoint"],
                            wavelet_mode=cfg["wavelet_mode"],
                            per_channel_stem=cfg["per_channel_stem"],
                            per_channel_wavelet=cfg["per_channel_wavelet"],
                            grouped_mix=cfg.get("grouped_mix", False))
    elif cfg["model"] == "segattn_haar":
        from model_segmamba.segmamba_attn import SegAttnHaar
        return SegAttnHaar(in_chans=4, out_chans=4,
                           depths=[2, 2, 2, 2], feat_size=[48, 96, 192, 384],
                           learnable_wavelet=cfg["learnable_wavelet"],
                           gsc_checkpoint=cfg["gsc_checkpoint"],
                           wavelet_mode=cfg["wavelet_mode"],
                           per_channel_stem=cfg["per_channel_stem"],
                           per_channel_wavelet=cfg["per_channel_wavelet"],
                           num_heads=cfg["num_heads"],
                           window_size=cfg["window_size"],
                           attn_checkpoint=cfg["attn_checkpoint"],
                           use_routing=cfg.get("use_routing", False),
                           rf_prior_scale=cfg.get("rf_prior_scale", 1.5),
                           router_hidden=cfg.get("router_hidden", None),
                           gate_hardness=cfg.get("gate_hardness", "soft"),
                           gumbel_tau_start=cfg.get("gumbel_tau_start", 1.0),
                           gumbel_tau_end=cfg.get("gumbel_tau_end", 0.05),
                           gumbel_tau_epochs=cfg.get("gumbel_tau_epochs", 300),
                           ste_warmup_epochs=cfg.get("ste_warmup_epochs", 0),
                           grouped_mix=cfg.get("grouped_mix", False))
    elif cfg["model"] in ("unetr", "unetr_deepmix"):
        from model_segmamba.unetr_deepmix import UNETRDeepMix
        default_from = 12 if cfg["model"] == "unetr" else 6
        mixer_from = int(cfg.get("mixer_from", default_from))
        heads = cfg.get("num_heads") or 12
        return UNETRDeepMix(
            in_channels=4, out_channels=4,
            img_size=tuple(cfg["roi_size"]),
            feature_size=int(cfg.get("feature_size", 16)),
            hidden_size=int(cfg.get("hidden_size", 768)),
            mlp_dim=int(cfg.get("mlp_dim", 3072)),
            num_heads=int(heads),
            mixer_from=int(mixer_from),
            norm_name="instance",
        )
    else:
        raise ValueError(f"Unknown model: {cfg['model']}")

# ═══════════════════════════════════════════════════════════════════════
# Tee stdout/stderr
# ═══════════════════════════════════════════════════════════════════════

class _Tee:
    def __init__(self, stream, fpath):
        self.stream = stream
        self.f = open(fpath, "a", buffering=1)

    def write(self, msg):
        self.stream.write(msg)
        self.f.write(msg)

    def flush(self):
        self.stream.flush()
        self.f.flush()

# ═══════════════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════════════

class BraTSTrainer(Trainer):
    def __init__(self, cfg, env_type, max_epochs, batch_size, device="cpu",
                 val_every=1, num_gpus=1, logdir="./logs/",
                 master_ip='localhost', master_port=17750,
                 training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every,
                         num_gpus, logdir, master_ip, master_port, training_script)
        self.cfg = cfg
        self._model_save_path = os.path.join(self.logdir, "model")

        self.window_infer = SlidingWindowInferer(
            roi_size=cfg["roi_size"], sw_batch_size=1, overlap=0.5)
        self.augmentation = True
        self.model = _build_model(cfg)
        self.patch_size = cfg["roi_size"]

        # foreground oversampling (config-driven). 0.66 = aggressive variant
        # used by exp1-8; 0.33 = nnU-Net default used by the old protocol.
        self.oversample_foreground_percent = cfg.get(
            "oversample_foreground_percent", 0.66)
        self.best_mean_dice = 0.0
        self.mse = nn.MSELoss()
        self.train_process = cfg.get("train_process", 18)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=1e-2, weight_decay=3e-5,
            momentum=0.99, nesterov=True)

        self.scheduler_type = "poly"
        self.cross = nn.CrossEntropyLoss()

        # Region-aligned, small-region-aware loss (config-driven).
        # exp1-8 default: weighted CE (bg 0.05, ET 1.5) + Tversky(0.3/0.7).
        # exp9 lossfix:   CE [1,1,1,1.2] + Tversky(0.5/0.5) + oversample 0.33
        #                 (nnU-Net-style recipe, to attribute the WT drop).
        ce_weight = torch.tensor(
            cfg.get("ce_weight", [0.05, 1.0, 1.0, 1.5]), device=device)
        self.ce = nn.CrossEntropyLoss(weight=ce_weight)
        self.tversky = TverskyLoss(to_onehot_y=True, softmax=True,
                                   include_background=False,
                                   alpha=cfg.get("tversky_alpha", 0.3),
                                   beta=cfg.get("tversky_beta", 0.7))
        # deep-supervision weights [dec3(16^3), dec2(32^3), dec1(64^3)]
        self.ds_weights = [0.125, 0.25, 0.5]

    # ── training ──────────────────────────────────────────────────────

    def training_step(self, batch):
        if not hasattr(self, "_eta_t0"):
            self._eta_t0, self._eta_ep0 = time.time(), self.epoch
        image, label = self.get_input(batch)

        # gate 训练硬度状态推进 (Gumbel tau 退火); 无路由/soft 模式为 no-op
        if hasattr(self.model, "set_gate_epoch"):
            self.model.set_gate_epoch(self.epoch)

        # gate stats are sampled at fixed intervals; host sync only on collect
        log_routing = bool(self.cfg.get("log_routing", False))
        gate_every = max(1, int(self.cfg.get("gate_log_interval", 25)))
        want_stats = log_routing and (self.global_step % gate_every == 0)
        out = self.model(image, deep_supervision=self.cfg["deep_supervision"],
                         return_gate_stats=want_stats)
        if want_stats:
            pred, aux_logits, gate_stats = out
        else:
            gate_stats = None
            if self.cfg["deep_supervision"]:
                pred, aux_logits = out
            else:
                pred = out
                aux_logits = None

        if self.cfg["use_dice_ce"]:
            loss_ce = self.ce(pred, label) + self.tversky(pred, label[:, None])
            if aux_logits is not None:
                total_w = 1.0 + sum(self.ds_weights)
                ds = 0.0
                for w, a in zip(self.ds_weights, aux_logits):
                    a_up = F.interpolate(a, size=tuple(label.shape[1:]),
                                         mode="trilinear", align_corners=False)
                    ds = ds + w * (self.ce(a_up, label)
                                  + self.tversky(a_up, label[:, None]))
                loss_ce = (loss_ce + ds) / total_w
        else:
            loss_ce = self.cross(pred, label)

        loss = loss_ce

        if self.cfg["wavelet_boundary_weight"] > 0:
            loss_wb = self.wavelet_boundary_loss(pred, label)
            loss = loss + self.cfg["wavelet_boundary_weight"] * loss_wb
            self.log("wavelet_boundary_loss", loss_wb, step=self.global_step)

        self.log("training_loss", loss_ce, step=self.global_step)

        # conditional-routing diagnostics: per-stage/block gate utilisation
        if gate_stats is not None:
            for key, s in gate_stats.items():
                self.log(f"gate/{key}_mean", s["mean"], step=self.global_step)
                self.log(f"gate/{key}_frac", s["attn_frac"], step=self.global_step)
                self.log(f"gate/{key}_prior", s["rf_prior"], step=self.global_step)
                if "use_attn" in s:  # hard-routing (eval) mode only
                    self.log(f"gate/{key}_use_attn", s["use_attn"], step=self.global_step)

        return loss

    def wavelet_boundary_loss(self, pred, label):
        """L1 distance between the high-frequency Haar subbands of the
        predicted probabilities and of the one-hot ground truth."""
        prob = pred.softmax(1)
        n_cls = prob.shape[1]
        with torch.no_grad():
            onehot = F.one_hot(label, num_classes=n_cls)
            onehot = onehot.permute(0, 4, 1, 2, 3).to(prob.dtype)
        k = _fixed_haar_kernel(prob.device, prob.dtype)
        B, C = prob.shape[:2]
        p = F.conv3d(prob.reshape(B * C, 1, *prob.shape[2:]), k, stride=2)
        t = F.conv3d(onehot.reshape(B * C, 1, *onehot.shape[2:]), k, stride=2)
        return F.l1_loss(p[:, 1:], t[:, 1:])

    # ── evaluation ────────────────────────────────────────────────────

    def convert_labels(self, labels):
        # After remapping: 1=NCR, 2=ED, 3=ET
        # TC = NCR + ET,  WT = all FG,  ET = label 3
        result = [(labels == 1) | (labels == 3),
                  (labels == 1) | (labels == 3) | (labels == 2),
                  labels == 3]
        return torch.cat(result, dim=1).float()

    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"][:, 0].long()
        return image, label

    def cal_metric(self, gt, pred):
        # gt, pred: (B, H, W, D) numpy masks.
        #   dice() — medpy pools over the whole 4D batch (unchanged behaviour).
        if pred.sum() > 0 and gt.sum() > 0:
            d = dice(pred, gt)
            return d
        elif gt.sum() == 0 and pred.sum() == 0:
            return 1.0
        else:
            return 0.0

    def _compute_hd95_gpu(self, pred, gt):
        """Compute per-sample HD95 on GPU via MONAI (fast & hang-free).

        Args:
            pred, gt: (B, H, W, D) float tensors on GPU, values in {0, 1}.

        Returns:
            list of float, one per sample in the batch (NaN for empty masks).
        """
        B = pred.shape[0]
        # (B, 1, H, W, D) — MONAI expects channel dim
        p = pred[:, None]
        g = gt[:, None]
        hd_metric = HausdorffDistanceMetric(
            percentile=95, reduction="none", include_background=False,
        )
        hd_metric(p, g)
        vals = hd_metric.aggregate().squeeze(-1)  # (B,) or (B, C)
        hd_metric.reset()
        # average across channels if multi-channel (shouldn't be here)
        if vals.ndim > 1:
            vals = vals.nanmean(dim=-1)
        return [float(v) for v in vals]

    def validation_step(self, batch):
        image, label = self.get_input(batch)
        output = self.model(image).argmax(dim=1)[:, None]
        output = self.convert_labels(output)
        label  = self.convert_labels(label[:, None])

        # ── GPU tensors for HD95 ──
        output_gpu = output          # (B, 3, H, W, D) on GPU
        target_gpu = label           # (B, 3, H, W, D) on GPU

        # numpy for dice (medpy)
        output_np = output.cpu().numpy()
        target_np = label.cpu().numpy()

        # ── soft pass (training-time supernet) ──
        soft_dice, soft_hd95 = [], []
        for i in range(3):
            d = self.cal_metric(target_np[:, i], output_np[:, i])
            soft_dice.append(d)
            # HD95 on GPU — fast, no subprocess, no hang
            hd_vals = self._compute_hd95_gpu(output_gpu[:, i], target_gpu[:, i])
            hd = float(np.nanmean(hd_vals)) if not all(np.isnan(v) for v in hd_vals) else np.nan
            soft_hd95.append(hd)
        # returned list: soft_dice[3] + soft_hd95[3]
        dices = soft_dice + soft_hd95

        # Conditional-routing probe: hard second pass at inference.
        if getattr(self, "_routing_probe", None) is None:
            self._routing_probe = (bool(self.cfg.get("use_routing", False))
                                   and hasattr(self.model, "set_hard_routing"))
        if self._routing_probe:
            self.model.set_hard_routing(True, self.cfg.get("routing_threshold", 0.5))
            logits_h, _, stats_h = self.model(image, return_gate_stats=True)
            self.model.set_hard_routing(False)
            output_h = self.convert_labels(logits_h.argmax(dim=1)[:, None])
            output_h_np = output_h.cpu().numpy()
            hard_dice, hard_hd95 = [], []
            for i in range(3):
                d = self.cal_metric(target_np[:, i], output_h_np[:, i])
                hard_dice.append(d)
                hd_vals = self._compute_hd95_gpu(output_h[:, i], target_gpu[:, i])
                hd = float(np.nanmean(hd_vals)) if not all(np.isnan(v) for v in hd_vals) else np.nan
                hard_hd95.append(hd)
            dices.extend(hard_dice)   # hard_dice[3]
            dices.extend(hard_hd95)   # hard_hd95[3]
            # 本 patch 上被跳过的 attention block 比例 (1 = 全跳)
            if stats_h:
                dices.append(1.0 - sum(s.get("use_attn", 1.0)
                                       for s in stats_h.values()) / max(len(stats_h), 1))
        return dices

    def validation_end(self, val_outputs):
        dices = val_outputs

        # indices (row-major: Dice then HD95 for TC/WT/ET):
        #   [0:3]   soft Dice
        #   [3:6]   soft HD95
        #   [6:9]   hard Dice  (only when use_routing)
        #   [9:12]  hard HD95  (only when use_routing)
        #   [12]    skip_frac  (only when use_routing)
        tc_d, wt_d, et_d = dices[0].mean(), dices[1].mean(), dices[2].mean()
        # Use nanmean for HD95: empty-class samples produce NaN (undefined
        # distance), which should be excluded from the average rather than
        # poisoning the entire metric.
        tc_h = dices[3].nanmean()
        wt_h = dices[4].nanmean()
        et_h = dices[5].nanmean()
        print(f"soft dices is {tc_d, wt_d, et_d}, hd95 is {tc_h, wt_h, et_h}")
        mean_dice = (tc_d + wt_d + et_d) / 3
        mean_hd95 = (tc_h + wt_h + et_h) / 3

        self.log("soft/tc_dice", tc_d, step=self.epoch)
        self.log("soft/wt_dice", wt_d, step=self.epoch)
        self.log("soft/et_dice", et_d, step=self.epoch)
        self.log("soft/mean_dice", mean_dice, step=self.epoch)
        self.log("soft/tc_hd95", tc_h, step=self.epoch)
        self.log("soft/wt_hd95", wt_h, step=self.epoch)
        self.log("soft/et_hd95", et_h, step=self.epoch)
        self.log("soft/mean_hd95", mean_hd95, step=self.epoch)

        # ── backward-compat dual-write (BREAKING rename recorded) ──
        # This probe renamed the TB keys from the flat legacy scheme
        # (tc/wt/et/mean_dice + tc_hd95/...) to the nested soft/* + hard/*
        # scheme. Old parsing scripts / historical runs expect the flat keys,
        # so we keep writing both until the parsers are migrated.
        self.log("tc", tc_d, step=self.epoch)
        self.log("wt", wt_d, step=self.epoch)
        self.log("et", et_d, step=self.epoch)
        self.log("mean_dice", mean_dice, step=self.epoch)
        self.log("tc_hd95", tc_h, step=self.epoch)
        self.log("wt_hd95", wt_h, step=self.epoch)
        self.log("et_hd95", et_h, step=self.epoch)
        self.log("mean_hd95", mean_hd95, step=self.epoch)

        # Hard-routing diagnostics.
        if len(dices) > 6:
            tc_hd, wt_hd, et_hd = dices[6].mean(), dices[7].mean(), dices[8].mean()
            hard_mean = (tc_hd + wt_hd + et_hd) / 3
            tc_hh, wt_hh, et_hh = dices[9].mean(), dices[10].mean(), dices[11].mean()
            hard_hd95_mean = (tc_hh + wt_hh + et_hh) / 3
            skip = dices[12].mean() if len(dices) > 12 else float('nan')
            self.log("hard/tc_dice", tc_hd, step=self.epoch)
            self.log("hard/wt_dice", wt_hd, step=self.epoch)
            self.log("hard/et_dice", et_hd, step=self.epoch)
            self.log("hard/mean_dice", hard_mean, step=self.epoch)
            self.log("hard/tc_hd95", tc_hh, step=self.epoch)
            self.log("hard/wt_hd95", wt_hh, step=self.epoch)
            self.log("hard/et_hd95", et_hh, step=self.epoch)
            self.log("hard/mean_hd95", hard_hd95_mean, step=self.epoch)
            self.log("hard/skip_frac", skip, step=self.epoch)
            # dual-write flat hard keys (same breaking-rename compat as soft)
            self.log("hard_tc", tc_hd, step=self.epoch)
            self.log("hard_wt", wt_hd, step=self.epoch)
            self.log("hard_et", et_hd, step=self.epoch)
            self.log("hard_mean_dice", hard_mean, step=self.epoch)
            self.log("hard_tc_hd95", tc_hh, step=self.epoch)
            self.log("hard_wt_hd95", wt_hh, step=self.epoch)
            self.log("hard_et_hd95", et_hh, step=self.epoch)
            self.log("hard_mean_hd95", hard_hd95_mean, step=self.epoch)
            print(f"hard-routing dices is {tc_hd, wt_hd, et_hd}, "
                  f"mean {hard_mean:.4f} (soft {mean_dice:.4f}), "
                  f"hd95 mean {hard_hd95_mean:.4f} (soft {mean_hd95:.4f}), "
                  f"attn-skip {skip:.3f}")

        # ── wavelet diagnostics (haar / attn arms only) ───────────
        if self.cfg["model"] in ("segmamba_haar", "segattn_haar"):
            thetas = [p.detach().float() for n, p in self.model.named_parameters()
                      if n.endswith("theta")]
            if thetas:
                dev = torch.stack(thetas) - math.pi / 4
                self.log("theta_dev_mean", dev.mean().item(), step=self.epoch)
                self.log("theta_dev_absmax", dev.abs().max().item(), step=self.epoch)

            if self.cfg.get("wavelet_mode") == "so8":
                from model_segmamba.lowo import HaarWavelet3D
                q_devs = [m.ortho_dev() for m in self.model.modules()
                          if isinstance(m, HaarWavelet3D) and m.mode == "so8"]
                if q_devs:
                    self.log("q_dev_mean", sum(q_devs) / len(q_devs),
                             step=self.epoch)
                    self.log("q_dev_max", max(q_devs), step=self.epoch)

        # ── checkpoint ────────────────────────────────────────────
        os.makedirs(self._model_save_path, exist_ok=True)

        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(
                self.model,
                os.path.join(self._model_save_path,
                             f"best_model_{mean_dice:.4f}.pt"),
                delete_symbol="best_model")

        save_new_model_and_delete_last(
            self.model,
            os.path.join(self._model_save_path,
                         f"final_model_{mean_dice:.4f}.pt"),
            delete_symbol="final_model")

        if (self.epoch + 1) % 100 == 0:
            torch.save(
                self.model.state_dict(),
                os.path.join(self._model_save_path,
                             f"tmp_model_ep{self.epoch}_{mean_dice:.4f}.pt"))

        print(f"mean_dice is {mean_dice}")
        self.save_resume_checkpoint(
            os.path.join(self._model_save_path, "resume_latest.pt"))

        # ── ETA ───────────────────────────────────────────────────
        if hasattr(self, "_eta_t0"):
            done = max(self.epoch - self._eta_ep0 + 1, 1)
            sec_per_ep = (time.time() - self._eta_t0) / done
            remain = (self.max_epochs - 1 - self.epoch) * sec_per_ep
            finish = time.strftime("%m-%d %H:%M",
                                   time.localtime(time.time() + remain))
            print(f"[ETA] epoch {self.epoch + 1}/{self.max_epochs} | "
                  f"{sec_per_ep / 60:.1f} min/epoch | "
                  f"remaining {remain / 3600:.1f} h | finish ~{finish}")
            self.log("eta_hours", remain / 3600, step=self.epoch)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified BraTS2021 training: SegMamba / SegMamba-Haar / SegAttn-Haar",
    )
    parser.add_argument("--config", required=True,
                        help="Path to YAML config file (e.g. configs/A6_so8_g8.yaml)")
    parser.add_argument("--tag", default=None,
                        help="Deterministic run-tag (logdir subfolder); enables resume across restarts")
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed passed to set_determinism() — controls weight init, "
                             "augmentation and dataloader worker seeding. The train/val/test "
                             "split is the frozen internal held-out lists in splits/, so "
                             "different --seed gives independent training runs on the same "
                             "cases (use 3 seeds for mean±std). Baked into the logdir tag "
                             "so seeds never collide or resume each other.")
    cli_args = parser.parse_args()

    # ── load config ───────────────────────────────────────────────
    cfg = load_config(cli_args.config)

    # ── determinism: seed from CLI (default 123) ──────────────────
    # Must run before model construction and data loading so that weight
    # init / augmentation / dataloader worker seeding are reproducible
    # per --seed (and differ across the 3 runs).
    set_determinism(cli_args.seed)

    # ── run metadata ──────────────────────────────────────────────
    _RUN_TS     = time.strftime("%Y%m%d_%H%M%S")
    _RUN_COMMIT = _safe_git_commit() or "nogit"
    _SEED       = cli_args.seed
    # seed is baked into the run tag so different seeds land in different
    # dirs (no ckpt collision, no accidental resume across seeds).
    _RUN_TAG    = (cli_args.tag if cli_args.tag else f"{_RUN_TS}_{_RUN_COMMIT}_pid{os.getpid()}") + f"_s{_SEED}"

    # ── logdir ────────────────────────────────────────────────────
    base_logdir = f"./logs/{cfg['model']}"
    if cfg["wavelet_mode"] is not None and cfg["wavelet_mode"] != "theta":
        base_logdir += f"_{cfg['wavelet_mode']}"
    if cfg["use_dice_ce"]:
        base_logdir += "_dicece"
    if cfg["deep_supervision"]:
        base_logdir += "_ds"

    logdir          = os.path.join(base_logdir, _RUN_TAG)
    model_save_path = os.path.join(logdir, "model")

    # ── create dirs, save yaml copy ───────────────────────────────
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)

    # Persist the seed for full reproducibility / post-hoc auditing.
    with open(os.path.join(logdir, "seed.txt"), "w") as _sf:
        _sf.write(str(_SEED) + "\n")

    # Copy the YAML config file itself for full reproducibility
    yaml_copy = os.path.join(logdir, "config.yaml")
    shutil.copy2(cli_args.config, yaml_copy)

    # ── tee stdout/stderr ─────────────────────────────────────────
    _stdout_log = os.path.join(
        logdir, f"stdout_{time.strftime('%Y%m%d_%H%M%S')}.log")
    sys.stdout = _Tee(sys.stdout, _stdout_log)
    sys.stderr = _Tee(sys.stderr, _stdout_log)

    print(f"[tee] stdout/stderr also saved to {_stdout_log}")
    print(f"[cfg] config yaml  → {yaml_copy}")
    print(f"[run] base_logdir  = {base_logdir}")
    print(f"[run] logdir       = {logdir}")
    print(f"[run] run_tag      = {_RUN_TAG}")
    print(f"[run] commit       = {_RUN_COMMIT}")
    print(f"[run] seed         = {_SEED}  (set_determinism applied)")
    for k in sorted(cfg):
        print(f"[cfg] {k}={cfg[k]!r}")

    # ── trainer ───────────────────────────────────────────────────
    trainer = BraTSTrainer(
        cfg=cfg,
        env_type=env,
        max_epochs=cfg["max_epoch"],
        batch_size=cfg["batch_size"],
        device=cfg["device"],
        logdir=logdir,
        val_every=cfg["val_every"],
        num_gpus=1,
        master_port=cfg["master_port"],
        training_script=__file__,
    )

    train_ds, val_ds, test_ds = get_train_val_test_loader_from_train(
        data_dir, dump_split_dir=logdir)

    resume_path = os.path.join(model_save_path, "resume_latest.pt")
    if cfg["resume"] and os.path.exists(resume_path):
        trainer.resume_from = resume_path

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
