"""
Preprocessing script for BraTS2021 data adapted for SegMamba.

BraTS2021 label mapping:
  0: background
  1: NCR (Necrotic/Non-Enhancing Tumor Core)
  2: ED (Peritumoral Edema)
  4: ET (Enhancing Tumor)

BraTS2021 evaluation regions:
  WT (Whole Tumor): labels 1, 2, 4
  TC (Tumor Core): labels 1, 4
  ET (Enhancing Tumor): label 4

We convert BraTS2021 labels to match the format used by SegMamba:
  0: background
  1: NCR (was 1)
  2: ED (was 2)
  3: ET (was 4)  <- remap 4 -> 3 to match BraTS2023 convention

Modalities (same order as BraTS2023):
  t2.nii.gz     (corresponds to BraTS2023 t2w)
  flair.nii.gz  (corresponds to BraTS2023 t2f)
  t1.nii.gz     (corresponds to BraTS2023 t1n)
  t1ce.nii.gz   (corresponds to BraTS2023 t1c)
"""

import argparse
import os

from light_training.preprocessing.preprocessors.preprocessor_mri import MultiModalityPreprocessor
import numpy as np
import SimpleITK as sitk
from light_training.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from batchgenerators.utilities.file_and_folder_operations import *


class BraTS2021Preprocessor(MultiModalityPreprocessor):
    """Preprocessor adapted for BraTS2021 dataset"""

    def get_iterable_list(self):
        """Override to only return valid case directories"""
        all_entries = os.listdir(os.path.join(self.base_dir, self.image_dir))
        # Filter to only directories that contain the expected files
        valid_cases = []
        for entry in sorted(all_entries):
            case_dir = os.path.join(self.base_dir, self.image_dir, entry)
            if os.path.isdir(case_dir):
                # Check if the directory contains the expected files
                seg_path = os.path.join(case_dir, self.seg_filename)
                if os.path.exists(seg_path):
                    valid_cases.append(entry)
                else:
                    print(f"Skipping {entry}: no {self.seg_filename} found")
            else:
                print(f"Skipping non-directory entry: {entry}")
        print(f"Found {len(valid_cases)} valid cases out of {len(all_entries)} entries")
        return valid_cases

    def read_data(self, case_name):
        data = []
        spacing = None
        for dfname in self.data_filenames:
            filepath = os.path.join(self.base_dir, self.image_dir, case_name, dfname)
            d = sitk.ReadImage(filepath)
            if spacing is None:
                spacing = d.GetSpacing()
            data.append(sitk.GetArrayFromImage(d).astype(np.float32)[None,])

        data = np.concatenate(data, axis=0)

        seg_arr = None
        if self.seg_filename != "":
            seg_path = os.path.join(self.base_dir, self.image_dir, case_name, self.seg_filename)
            seg = sitk.ReadImage(seg_path)
            seg_arr = sitk.GetArrayFromImage(seg).astype(np.float32)
            seg_arr = seg_arr[None]

            # Remap BraTS2021 labels: 4 -> 3 (to match BraTS2023 convention)
            seg_arr[seg_arr == 4] = 3

            intensities_per_channel, intensity_statistics_per_channel = self.collect_foreground_intensities(seg_arr, data)
        else:
            intensities_per_channel = []
            intensity_statistics_per_channel = []

        properties = {
            "spacing": spacing,
            "raw_size": data.shape[1:],
            "name": case_name,
            "intensities_per_channel": intensities_per_channel,
            "intensity_statistics_per_channel": intensity_statistics_per_channel
        }

        return data, seg_arr, properties


# BraTS2021 data paths
base_dir = os.environ.get(
    "BRATS2021_DATA_DIR", "./data/fullres/raw/brats2021_extracted"
)
image_dir = "."  # cases are directly under brats2021_extracted

# Modality filenames (same order as BraTS2023: t2w, t2f, t1n, t1c)
data_filename = [
    "t2.nii.gz",
    "flair.nii.gz",
    "t1.nii.gz",
    "t1ce.nii.gz"
]
seg_filename = "seg.nii.gz"


def process_train():
    preprocessor = BraTS2021Preprocessor(
        base_dir=base_dir,
        image_dir=image_dir,
        data_filenames=data_filename,
        seg_filename=seg_filename
    )

    out_spacing = [1.0, 1.0, 1.0]
    output_dir = "./data/fullres/train/"

    preprocessor.run(
        output_spacing=out_spacing,
        output_dir=output_dir,
        all_labels=[1, 2, 3],  # After remapping: NCR=1, ED=2, ET=3
    )


def plan():
    preprocessor = BraTS2021Preprocessor(
        base_dir=base_dir,
        image_dir=image_dir,
        data_filenames=data_filename,
        seg_filename=seg_filename
    )

    preprocessor.run_plan()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess BraTS2021 data (labels remapped 4->3)."
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("BRATS2021_DATA_DIR", "./data/fullres/raw/brats2021_extracted"),
        help="Root directory containing BraTS2021 cases (overrides BRATS2021_DATA_DIR env var).",
    )
    args = parser.parse_args()
    base_dir = args.data_dir
    plan()
    process_train()
