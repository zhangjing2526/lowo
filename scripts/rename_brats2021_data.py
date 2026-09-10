"""
Rename BraTS2021 data files to match the format expected by SegMamba.
BraTS2021 naming: BraTS2021_XXXXX_t1.nii.gz, BraTS2021_XXXXX_t1ce.nii.gz, etc.
We rename to: t1.nii.gz, t1ce.nii.gz, t2.nii.gz, flair.nii.gz, seg.nii.gz
inside each case directory.
"""

import argparse
import os
import shutil

def main():
    parser = argparse.ArgumentParser(
        description="Rename BraTS2021 files to SegMamba-expected names."
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("BRATS2021_DATA_DIR", "./data/fullres/raw/brats2021_extracted"),
        help="Root directory with BraTS2021 case folders (overrides BRATS2021_DATA_DIR env var).",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    all_cases = sorted(os.listdir(data_dir))

    for case_name in all_cases:
        case_dir = os.path.join(data_dir, case_name)

        if not os.path.isdir(case_dir):
            continue

        # Create a renamed directory with just the case number
        new_case_dir = os.path.join(data_dir, case_name)

        for data_name in os.listdir(case_dir):
            old_path = os.path.join(case_dir, data_name)

            if not os.path.isfile(old_path):
                continue

            # Map BraTS2021 naming to standardized naming
            # BraTS2021_XXXXX_t1.nii.gz -> t1.nii.gz
            # BraTS2021_XXXXx_t1ce.nii.gz -> t1ce.nii.gz
            # BraTS2021_XXXXx_t2.nii.gz -> t2.nii.gz
            # BraTS2021_XXXXx_flair.nii.gz -> flair.nii.gz
            # BraTS2021_XXXXx_seg.nii.gz -> seg.nii.gz

            if "_t1.nii" in data_name and "t1c" not in data_name:
                new_name = "t1.nii.gz"
            elif "_t1ce.nii" in data_name or "_t1c.nii" in data_name:
                new_name = "t1ce.nii.gz"
            elif "_t2.nii" in data_name:
                new_name = "t2.nii.gz"
            elif "_flair.nii" in data_name:
                new_name = "flair.nii.gz"
            elif "_seg.nii" in data_name:
                new_name = "seg.nii.gz"
            else:
                print(f"Skipping unknown file: {data_name}")
                continue

            new_path = os.path.join(case_dir, new_name)

            if old_path != new_path:
                # Copy instead of rename to preserve originals
                if not os.path.exists(new_path):
                    shutil.copy2(old_path, new_path)
                    print(f"Copied: {data_name} -> {new_name} in {case_name}")
                else:
                    print(f"Already exists: {new_name} in {case_name}")

    print("Rename complete!")


if __name__ == "__main__":
    main()
