"""
Training script for SegAttn-Haar on BraTS2021 dataset.

Cross-architecture arm of the LOWO ablation: identical protocol to
train_brats2021_haar.py (same data, loss, optimizer, schedule, eval) -- the
ONLY difference is the per-stage sequence mixer: MambaLayer (SSM scan) is
replaced by WindowAttnLayer (Swin-style shifted-window self-attention). The
Haar stem/downsample, wavelet GSC, skip connections, Haar up-sampling and deep
supervision are byte-identical, so any Dice/HD95 delta is attributable to the
sequence mixer, not the wavelet operators.

This script does NOT import mamba_ssm.

BraTS2021 label mapping (after preprocessing remapping):
  0: background
  1: NCR (Necrotic/Non-Enhancing Tumor Core)
  2: ED (Peritumoral Edema)
  3: ET (Enhancing Tumor) - originally was 4 in BraTS2021, remapped to 3

Evaluation regions:
  TC (Tumor Core): labels 1, 3
  WT (Whole Tumor): labels 1, 2, 3
  ET (Enhancing Tumor): label 3
"""

import math

import numpy as np
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.losses import TverskyLoss
from monai.utils import set_determinism
from light_training.utils.files_helper import save_new_model_and_delete_last

set_determinism(123)
import os
import subprocess
import sys
import time

data_dir = "./data/fullres/train"
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


_RUN_TS = time.strftime("%Y%m%d_%H%M%S")
_RUN_COMMIT = _safe_git_commit() or "nogit"
_RUN_TAG = f"{_RUN_TS}_{_RUN_COMMIT}_pid{os.getpid()}"

# "theta": 3-DOF separable per-axis rotation angles (original SegMamba-Haar).
# "so8":   28-DOF non-separable orthogonal mixing K = expm(A-A^T) @ K_haar;
#          still perfect-reconstruction, starts exactly at fixed Haar.
wavelet_mode = "so8"

# region-aligned loss: Dice (foreground classes) + CE, replacing plain CE.
# Must stay in sync with train_brats2021.py / _haar.py (fair ablation protocol).
use_dice_ce = True

# deep supervision: aux heads on the 3 deeper decoder levels (16^3/32^3/64^3),
# same region loss, upsampled to full res. Must stay in sync with baseline.
deep_supervision = True

# ---- attention-specific knobs (the ONLY difference vs the mamba arm) ----
# local window per axis (SwinUNETR default). Keep (7,7,7) to match the
# standard Swin baseline; the deepest 8^3 stage is padded to 14^3 internally.
window_size = (7, 7, 7)

# attention heads per stage; None auto-derives dim // 16 (head_dim = 16),
# matching MONAI SwinUNETR's default for feat_size [48,96,192,384] -> [3,6,12,24].
num_heads = None

# gradient-checkpoint the attention + FFN sub-blocks (math-identical recompute).
# Attention materializes num_windows x heads x 343 x 343 per block at stage 0
# (~1.4GB @ 64^3, bs1), so this is ON by default to keep bs1 trainable.
attn_checkpoint = True
# -------------------------------------------------------------

base_logdir = "./logs/" + os.path.splitext(os.path.basename(__file__))[0] + (
    f"_{wavelet_mode}" if wavelet_mode != "theta" else "") + (
    "_dicece" if use_dice_ce else "")
logdir = os.path.join(base_logdir, _RUN_TAG)

model_save_path = os.path.join(logdir, "model")

# ---- tee stdout/stderr to a log file ----
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

os.makedirs(logdir, exist_ok=True)
_stdout_log = os.path.join(logdir, f"stdout_{time.strftime('%Y%m%d_%H%M%S')}.log")
sys.stdout = _Tee(sys.stdout, _stdout_log)
sys.stderr = _Tee(sys.stderr, _stdout_log)
print(f"[tee] stdout/stderr also saved to {_stdout_log}")
print(f"[run] base_logdir={base_logdir}")
print(f"[run] wavelet_mode={wavelet_mode}")
print(f"[run] window_size={window_size}")
print(f"[run] num_heads={num_heads}")
print(f"[run] attn_checkpoint={attn_checkpoint}")
print(f"[run] run_tag={_RUN_TAG}")
print(f"[run] commit={_RUN_COMMIT}")
print(f"[run] logdir={logdir}")
# -----------------------------------------

augmentation = True

env = "pytorch"
resume = True  # True: auto-resume from model_save_path/resume_latest.pt if it exists
max_epoch = 500
batch_size = 1  # window attention is memory-heavy; bs2 usually OOMs at 128^3
val_every = 2
num_gpus = 1
device = "cuda:0"
roi_size = [128, 128, 128]

# set False for the fixed-Haar ablation
learnable_wavelet = True

# activation checkpointing on the 4 GSC blocks: identical math, lower peak
# memory at the cost of recompute. Keep False to match the mamba arm.
gsc_checkpoint = False

# Per-modality stem: give each of the 4 MRI channels (T2/FLAIR/T1/T1ce) its own
# subband mixer instead of one shared 32->48 1x1, so modalities with different
# small-object evidence are weighted independently. per_channel_wavelet=True
# additionally gives each modality its own learnable wavelet basis.
# NOTE: changes the stem state_dict keys -> retrain from scratch (no old ckpt).
per_channel_stem = False
per_channel_wavelet = False

# weight of the wavelet-domain boundary (high-frequency) auxiliary loss.
# 0 disables it (== exact baseline protocol). It penalizes the L1 gap between
# the 7 high-frequency Haar subbands of softmax(pred) and the one-hot label,
# directly targeting boundary fidelity (HD95). Kept in sync with the haar arm.
wavelet_boundary_weight = 0.1


def _fixed_haar_kernel(device, dtype):
    """Fixed (8,1,2,2,2) orthonormal 3D Haar analysis kernel (theta = pi/4).
    Subband 0 = LLL, 1..7 = high-frequency subbands."""
    f = torch.tensor([[1.0, 1.0], [-1.0, 1.0]], device=device, dtype=dtype) / math.sqrt(2.0)
    k = torch.einsum('ad,bh,cw->abcdhw', f, f, f)
    return k.reshape(8, 1, 2, 2, 2)


def func(m, epochs):
    return np.exp(-10 * (1 - m / epochs) ** 2)


class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/",
                 master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip,
                         master_port, training_script)
        self.window_infer = SlidingWindowInferer(roi_size=roi_size,
                                                  sw_batch_size=1,
                                                  overlap=0.5)
        self.augmentation = augmentation
        from model_segmamba.segmamba_attn import SegAttnHaar

        self.model = SegAttnHaar(in_chans=4,
                                 out_chans=4,
                                 depths=[2, 2, 2, 2],
                                 feat_size=[48, 96, 192, 384],
                                 learnable_wavelet=learnable_wavelet,
                                 gsc_checkpoint=gsc_checkpoint,
                                 wavelet_mode=wavelet_mode,
                                 per_channel_stem=per_channel_stem,
                                 per_channel_wavelet=per_channel_wavelet,
                                 num_heads=num_heads,
                                 window_size=window_size,
                                 attn_checkpoint=attn_checkpoint)

        self.patch_size = roi_size
        # foreground oversampling: raise from the 0.33 default so small lesions
        # (esp. ET / NCR cores) see more patches. Kept in sync with the baseline
        # trainer for a fair ablation.
        self.oversample_foreground_percent = 0.66
        self.best_mean_dice = 0.0
        self.mse = nn.MSELoss()
        self.train_process = 18
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2, weight_decay=3e-5,
                                         momentum=0.99, nesterov=True)

        self.scheduler_type = "poly"
        self.cross = nn.CrossEntropyLoss()
        # Region-aligned, small-region-aware loss (kept in sync with the baseline
        # for a fair ablation):
        #   weighted CE  -- down-weight background (0.05) and up-weight ET (1.5) so
        #                   the 95%+ background voxels stop dominating the gradient
        #   + Tversky(FG, beta>alpha) -- beta=0.7 penalizes FALSE NEGATIVES harder
        #                   than false positives, directly fighting small-region miss
        # to_onehot_y=True: label passed as (B,1,...) integer map
        ce_weight = torch.tensor([0.05, 1.0, 1.0, 1.5], device=device)
        self.ce = nn.CrossEntropyLoss(weight=ce_weight)
        self.tversky = TverskyLoss(to_onehot_y=True, softmax=True,
                                   include_background=False,
                                   alpha=0.3, beta=0.7)
        # deep-supervision weights for the aux levels [dec3(16^3), dec2(32^3),
        # dec1(64^3)]; the final output implicitly carries weight 1.0. Normalized
        # in training_step so the total loss stays on the same scale.
        self.ds_weights = [0.125, 0.25, 0.5]

    def training_step(self, batch):
        if not hasattr(self, "_eta_t0"):
            self._eta_t0, self._eta_ep0 = time.time(), self.epoch
        image, label = self.get_input(batch)

        if deep_supervision:
            pred, aux_logits = self.model(image, deep_supervision=True)
        else:
            pred = self.model(image)
            aux_logits = None

        if use_dice_ce:
            # main output: weighted CE + Tversky (region-aligned, recall-favoring)
            loss_ce = self.ce(pred, label) + self.tversky(pred, label[:, None])
            # deep supervision: same loss on each decoder level, upsampled to full
            # res, weighted by depth (deepest gets the smallest weight), normalized.
            if aux_logits is not None:
                total_w = 1.0 + sum(self.ds_weights)
                ds = 0.0
                for w, a in zip(self.ds_weights, aux_logits):
                    a_up = F.interpolate(a, size=tuple(label.shape[1:]),
                                         mode="trilinear", align_corners=False)
                    ds = ds + w * (self.ce(a_up, label) + self.tversky(a_up, label[:, None]))
                loss_ce = (loss_ce + ds) / total_w
        else:
            loss_ce = self.cross(pred, label)
        loss = loss_ce

        if wavelet_boundary_weight > 0:
            loss_wb = self.wavelet_boundary_loss(pred, label)
            loss = loss + wavelet_boundary_weight * loss_wb
            self.log("wavelet_boundary_loss", loss_wb, step=self.global_step)

        self.log("training_loss", loss_ce, step=self.global_step)

        return loss

    def wavelet_boundary_loss(self, pred, label):
        """L1 distance between the high-frequency Haar subbands of the
        predicted probabilities and of the one-hot ground truth. Only the
        7 detail subbands are compared, so this term is sensitive purely
        to boundary/high-frequency structure."""
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

    def convert_labels(self, labels):
        ## TC, WT and ET
        # After remapping: 1=NCR, 2=ED, 3=ET
        # TC = NCR + ET = label 1 | label 3
        # WT = NCR + ED + ET = label 1 | label 2 | label 3
        # ET = label 3
        result = [(labels == 1) | (labels == 3),
                  (labels == 1) | (labels == 3) | (labels == 2),
                  labels == 3]

        return torch.cat(result, dim=1).float()

    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"]

        label = label[:, 0].long()
        return image, label

    def cal_metric(self, gt, pred, voxel_spacing=[1.0, 1.0, 1.0]):
        if pred.sum() > 0 and gt.sum() > 0:
            d = dice(pred, gt)
            return np.array([d, 50])

        elif gt.sum() == 0 and pred.sum() == 0:
            return np.array([1.0, 50])

        else:
            return np.array([0.0, 50])

    def validation_step(self, batch):
        image, label = self.get_input(batch)

        output = self.model(image)

        output = output.argmax(dim=1)

        output = output[:, None]
        output = self.convert_labels(output)

        label = label[:, None]
        label = self.convert_labels(label)

        output = output.cpu().numpy()
        target = label.cpu().numpy()

        dices = []

        c = 3
        for i in range(0, c):
            pred_c = output[:, i]
            target_c = target[:, i]

            cal_dice, _ = self.cal_metric(target_c, pred_c)
            dices.append(cal_dice)

        return dices

    def validation_end(self, val_outputs):
        dices = val_outputs

        tc, wt, et = dices[0].mean(), dices[1].mean(), dices[2].mean()

        print(f"dices is {tc, wt, et}")

        mean_dice = (tc + wt + et) / 3

        self.log("tc", tc, step=self.epoch)
        self.log("wt", wt, step=self.epoch)
        self.log("et", et, step=self.epoch)

        self.log("mean_dice", mean_dice, step=self.epoch)

        # track how far the learnable wavelet angles have drifted from the
        # standard Haar init (pi/4): direct evidence for the learnable-wavelet
        # claim if nonzero, early warning if they never move.
        thetas = [p.detach().float() for n, p in self.model.named_parameters()
                  if n.endswith("theta")]
        if thetas:
            dev = torch.stack(thetas) - math.pi / 4
            self.log("theta_dev_mean", dev.mean().item(), step=self.epoch)
            self.log("theta_dev_absmax", dev.abs().max().item(), step=self.epoch)

        # so8 mode: ||Q - I||_F per wavelet module -- drift of the learnable
        # non-separable orthogonal mixing away from the fixed-Haar identity.
        # Imported from lowo (not segmamba_haar) so this script stays mamba-free.
        from model_segmamba.lowo import HaarWavelet3D
        q_devs = [m.ortho_dev() for m in self.model.modules()
                  if isinstance(m, HaarWavelet3D) and m.mode == "so8"]
        if q_devs:
            self.log("q_dev_mean", sum(q_devs) / len(q_devs), step=self.epoch)
            self.log("q_dev_max", max(q_devs), step=self.epoch)

        os.makedirs(model_save_path, exist_ok=True)

        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(self.model,
                                           os.path.join(model_save_path,
                                                        f"best_model_{mean_dice:.4f}.pt"),
                                           delete_symbol="best_model")

        save_new_model_and_delete_last(self.model,
                                       os.path.join(model_save_path,
                                                    f"final_model_{mean_dice:.4f}.pt"),
                                       delete_symbol="final_model")

        if (self.epoch + 1) % 100 == 0:
            torch.save(self.model.state_dict(),
                       os.path.join(model_save_path, f"tmp_model_ep{self.epoch}_{mean_dice:.4f}.pt"))

        print(f"mean_dice is {mean_dice}")

        self.save_resume_checkpoint(os.path.join(model_save_path, "resume_latest.pt"))

        if hasattr(self, "_eta_t0"):
            done = max(self.epoch - self._eta_ep0 + 1, 1)
            sec_per_ep = (time.time() - self._eta_t0) / done
            remain = (self.max_epochs - 1 - self.epoch) * sec_per_ep
            finish = time.strftime("%m-%d %H:%M", time.localtime(time.time() + remain))
            print(f"[ETA] epoch {self.epoch + 1}/{self.max_epochs} | "
                  f"{sec_per_ep / 60:.1f} min/epoch | remaining {remain / 3600:.1f} h | "
                  f"finish ~{finish}")
            self.log("eta_hours", remain / 3600, step=self.epoch)


if __name__ == "__main__":
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)

    trainer = BraTSTrainer(env_type=env,
                           max_epochs=max_epoch,
                           batch_size=batch_size,
                           device=device,
                           logdir=logdir,
                           val_every=val_every,
                           num_gpus=num_gpus,
                           master_port=17762,  # distinct from _haar.py (17761) so both arms can run concurrently
                           training_script=__file__)

    train_ds, val_ds, test_ds = get_train_val_test_loader_from_train(
        data_dir, dump_split_dir=logdir)

    resume_path = os.path.join(model_save_path, "resume_latest.pt")
    if resume and os.path.exists(resume_path):
        trainer.resume_from = resume_path

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
