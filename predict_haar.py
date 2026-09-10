"""Inference (full-volume sliding-window + mirror TTA) for SegMamba-Haar on BraTS2021.

Mirrors predict.py but loads the Haar variant and a LOCAL Haar checkpoint.
The original upstream predict.py pointed at a machine-specific path and at
the baseline SegMamba, so it could not evaluate Haar runs on this machine.

Run after training:
    python predict_haar.py
It auto-discovers the best_model_*.pt with the highest Dice under
./logs/segmamba_haar_brats2021_so8_dicece/  (override with HAAR_CKPT env var).
"""

import glob
import os
import re

import numpy as np
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.prediction import Predictor

set_determinism(123)

data_dir = "./data/fullres/train"
env = "pytorch"
max_epoch = 1000
batch_size = 2
val_every = 2
num_gpus = 1
device = "cuda:0"
patch_size = [128, 128, 128]

# ---- Haar model config: MUST match train_brats2021_haar.py ----
wavelet_mode = "so8"
learnable_wavelet = True
gsc_checkpoint = False

haar_logdir = "./logs/segmamba_haar_brats2021_so8_dicece"


def _find_haar_checkpoint():
    """Return the local Haar best_model_*.pt with the highest Dice score."""
    cands = sorted(
        glob.glob(os.path.join(haar_logdir, "**", "model", "best_model_*.pt"), recursive=True),
        key=lambda p: float(re.search(r"best_model_([0-9.]+)\.pt$", p).group(1))
        if re.search(r"best_model_([0-9.]+)\.pt$", p) else -1.0,
        reverse=True)
    if not cands:
        raise FileNotFoundError(
            f"No Haar checkpoint found under {haar_logdir!r}. "
            "Train with train_brats2021_haar.py first, or set HAAR_CKPT.")
    return cands[0]


class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)

        self.patch_size = patch_size
        self.augmentation = False

    def convert_labels(self, labels):
        ## TC, WT and ET
        result = [(labels == 1) | (labels == 3), (labels == 1) | (labels == 3) | (labels == 2), labels == 3]

        return torch.cat(result, dim=1).float()

    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"]
        properties = batch["properties"]
        label = self.convert_labels(label)

        return image, label, properties

    def define_model_segmamba(self):
        from model_segmamba.segmamba_haar import SegMambaHaar
        model = SegMambaHaar(in_chans=4,
                            out_chans=4,
                            depths=[2, 2, 2, 2],
                            feat_size=[48, 96, 192, 384],
                            learnable_wavelet=learnable_wavelet,
                            gsc_checkpoint=gsc_checkpoint,
                            wavelet_mode=wavelet_mode)

        ckpt = os.environ.get("HAAR_CKPT") or _find_haar_checkpoint()
        print(f"[haar predict] loading checkpoint: {ckpt}")
        new_sd = self.filte_state_dict(torch.load(ckpt, map_location="cpu"))
        model.load_state_dict(new_sd)
        model.eval()
        window_infer = SlidingWindowInferer(roi_size=patch_size,
                                        sw_batch_size=2,
                                        overlap=0.75,
                                        progress=True,
                                        mode="gaussian")

        predictor = Predictor(window_infer=window_infer,
                              mirror_axes=[0, 1, 2])

        save_path = "./prediction_results/segmamba_haar"
        os.makedirs(save_path, exist_ok=True)

        return model, predictor, save_path

    def validation_step(self, batch):
        image, label, properties = self.get_input(batch)
        ddim = False

        model, predictor, save_path = self.define_model_segmamba()

        model_output = predictor.maybe_mirror_and_predict(image, model, device=device)

        model_output = predictor.predict_raw_probability(model_output,
                                                         properties=properties)


        model_output = model_output.argmax(dim=0)[None]
        model_output = self.convert_labels_dim0(model_output)

        label = label[0]
        c = 3
        dices = []
        for i in range(0, c):
            output_i = model_output[i].cpu().numpy()
            label_i = label[i].cpu().numpy()
            d = dice(output_i, label_i)
            dices.append(d)

        print(dices)

        model_output = predictor.predict_noncrop_probability(model_output, properties)
        predictor.save_to_nii(model_output,
                              raw_spacing=[1, 1, 1],
                              case_name=properties['name'][0],
                              save_dir=save_path)

        return 0

    def convert_labels_dim0(self, labels):
        ## TC, WT and ET
        result = [(labels == 1) | (labels == 3), (labels == 1) | (labels == 3) | (labels == 2), labels == 3]

        return torch.cat(result, dim=0).float()

    def filte_state_dict(self, sd):
        if "module" in sd:
            sd = sd["module"]
        new_sd = {}
        for k, v in sd.items():
            k = str(k)
            new_k = k[7:] if k.startswith("module") else k
            new_sd[new_k] = v
        del sd
        return new_sd


if __name__ == "__main__":

    trainer = BraTSTrainer(env_type=env,
                            max_epochs=max_epoch,
                            batch_size=batch_size,
                            device=device,
                            logdir="",
                            val_every=val_every,
                            num_gpus=num_gpus,
                            master_port=17751,
                            training_script=__file__)

    train_ds, val_ds, test_ds = get_train_val_test_loader_from_train(data_dir)

    trainer.validation_single_gpu(test_ds)
