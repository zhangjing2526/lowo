"""
Training script for Swin UNETR on BraTS2021 dataset.

Uses the MONAI SwinUNETR implementation from the local monai package.
Follows the same training pipeline as train_brats2021.py (SegMamba).

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

import argparse
import shutil

import numpy as np
import yaml
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.utils.files_helper import save_new_model_and_delete_last
from monai.losses.dice import DiceLoss

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
base_logdir = "./logs/swin_unetr_brats2021"
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
print(f"[run] run_tag={_RUN_TAG}")
print(f"[run] commit={_RUN_COMMIT}")
print(f"[run] logdir={logdir}")
# -----------------------------------------


# augmentation = "nomirror"
augmentation = True

env = "pytorch"
num_gpus = 1

# Overridden in __main__ from --config (defaults match the trained S1 run).
_S1_CFG = {
    "model": "swin_unetr",
    "feature_size": 48,
    "depths": [2, 2, 2, 2],
    "num_heads": [3, 6, 12, 24],
    "norm_name": "instance",
    "drop_rate": 0.0,
    "attn_drop_rate": 0.0,
    "dropout_path_rate": 0.0,
    "use_checkpoint": False,
    "batch_size": 2,
    "max_epoch": 500,
    "val_every": 2,
    "device": "cuda:0",
    "master_port": 17760,
    "roi_size": [128, 128, 128],
    "resume": True,
    "train_process": 18,
}


def load_s1_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if cfg.get("model", "swin_unetr") != "swin_unetr":
        raise ValueError(
            f"{path} has model={cfg.get('model')!r}; S1 must be model: swin_unetr"
        )
    out = dict(_S1_CFG)
    out.update({k: v for k, v in cfg.items() if v is not None})
    if isinstance(out.get("roi_size"), tuple):
        out["roi_size"] = list(out["roi_size"])
    return out


resume = _S1_CFG["resume"]
max_epoch = _S1_CFG["max_epoch"]
batch_size = _S1_CFG["batch_size"]
val_every = _S1_CFG["val_every"]
device = _S1_CFG["device"]
roi_size = list(_S1_CFG["roi_size"])


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
        from monai.networks.nets import SwinUNETR

        self.model = SwinUNETR(
            img_size=roi_size,
            in_channels=4,
            out_channels=4,
            feature_size=_S1_CFG["feature_size"],
            depths=list(_S1_CFG["depths"]),
            num_heads=list(_S1_CFG["num_heads"]),
            norm_name=_S1_CFG["norm_name"],
            drop_rate=_S1_CFG["drop_rate"],
            attn_drop_rate=_S1_CFG["attn_drop_rate"],
            dropout_path_rate=_S1_CFG["dropout_path_rate"],
            normalize=True,
            use_checkpoint=_S1_CFG["use_checkpoint"],
            spatial_dims=3,
        )

        self.patch_size = roi_size
        self.best_mean_dice = 0.0
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.train_process = _S1_CFG["train_process"]
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2, weight_decay=3e-5,
                                         momentum=0.99, nesterov=True)

        self.scheduler_type = "poly"
        self.cross = nn.CrossEntropyLoss()

    def training_step(self, batch):
        if not hasattr(self, "_eta_t0"):
            self._eta_t0, self._eta_ep0 = time.time(), self.epoch
        image, label = self.get_input(batch)

        pred = self.model(image)

        loss = self.cross(pred, label)

        self.log("training_loss", loss, step=self.global_step)

        return loss

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
    parser = argparse.ArgumentParser(description="SwinUNETR BraTS2021 — paper arm S1")
    parser.add_argument("--config", default="configs/S1_swin_unetr.yaml",
                        help="S1 YAML (must be model: swin_unetr)")
    args = parser.parse_args()

    _S1_CFG.update(load_s1_config(args.config))
    resume = _S1_CFG["resume"]
    max_epoch = _S1_CFG["max_epoch"]
    batch_size = _S1_CFG["batch_size"]
    val_every = _S1_CFG["val_every"]
    device = _S1_CFG["device"]
    roi_size = list(_S1_CFG["roi_size"])

    os.makedirs(logdir, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copy2(args.config, os.path.join(logdir, "config.yaml"))

    trainer = BraTSTrainer(env_type=env,
                           max_epochs=max_epoch,
                           batch_size=batch_size,
                           device=device,
                           logdir=logdir,
                           val_every=val_every,
                           num_gpus=num_gpus,
                           master_port=int(_S1_CFG["master_port"]),
                           training_script=__file__)

    train_ds, val_ds, test_ds = get_train_val_test_loader_from_train(
        data_dir, dump_split_dir=logdir)

    resume_path = os.path.join(model_save_path, "resume_latest.pt")
    if resume and os.path.exists(resume_path):
        trainer.resume_from = resume_path

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
