
# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from sklearn.model_selection import KFold  ## K折交叉验证
import pickle
import os
import json
import math
import numpy as np
import torch
from monai import transforms
import SimpleITK as sitk
from tqdm import tqdm 
from torch.utils.data import Dataset 
import glob 
from light_training.dataloading.utils import unpack_dataset
import random 

class MedicalDataset(Dataset):
    def __init__(self, datalist, test=False) -> None:
        super().__init__()
        
        self.datalist = datalist
        self.test = test 

        self.data_cached = []
        for p in tqdm(self.datalist, total=len(self.datalist)):
            info = self.load_pkl(p)

            self.data_cached.append(info)

        ## unpacking
        print(f"unpacking data ....")
        # for 
        folder = []
        for p in self.datalist:
            f = os.path.dirname(p)
            if f not in folder:
                folder.append(f)
        for f in folder:
            unpack_dataset(f, 
                        unpack_segmentation=True,
                        overwrite_existing=False,
                        num_processes=8)


        print(f"data length is {len(self.datalist)}")
        
    def load_pkl(self, data_path):
        pass 
        properties_path = f"{data_path[:-4]}.pkl"
        df = open(properties_path, "rb")
        info = pickle.load(df)

        return info 
    
    def post(self, batch_data):
        return batch_data
    
    def read_data(self, data_path):
        
        image_path = data_path.replace(".npz", ".npy")
        seg_path = data_path.replace(".npz", "_seg.npy")
        image_data = np.load(image_path, "r+")
      
        seg_data = None 
        if not self.test:
            seg_data = np.load(seg_path, "r+")
        return image_data, seg_data

    def __getitem__(self, i):
        
        image, seg = self.read_data(self.datalist[i])

        properties = self.data_cached[i]

        if seg is None:
            return {
                "data": image,
                "properties": properties
            }
        else :
            return {
                "data": image,
                "seg": seg,
                "properties": properties
            }

    def __len__(self):
        return len(self.datalist)

def get_train_test_loader_from_test_list(data_dir, test_list):
    all_paths = glob.glob(f"{data_dir}/*.npz")

    test_datalist = []
    train_datalist = []

    test_list_1 = []
    for t in test_list:
        test_list_1.append(t.replace(".nii.gz", ""))

    test_list = test_list_1
    for p in all_paths:
        p2 = p.split("/")[-1].split(".")[0]
        if p2 in test_list:
            test_datalist.append(p)
        else :
            train_datalist.append(p)

    print(f"training data is {len(train_datalist)}")
    print(f"test data is {len(test_datalist)}", test_datalist)

    train_ds = MedicalDataset(train_datalist)
    test_ds = MedicalDataset(test_datalist)

    loader = [train_ds, test_ds]

    return loader

def get_kfold_data(data_paths, n_splits, shuffle=False):
    X = np.arange(len(data_paths))
    kfold = KFold(n_splits=n_splits, shuffle=shuffle)  ## kfold为KFolf类的一个对象
    return_res = []
    for a, b in kfold.split(X):
        fold_train = []
        fold_val = []
        for i in a:
            fold_train.append(data_paths[i])
        for j in b:
            fold_val.append(data_paths[j])
        return_res.append({"train_data": fold_train, "val_data": fold_val})

    return return_res

def get_kfold_loader(data_dir, fold=0, test_dir=None):

    all_paths = glob.glob(f"{data_dir}/*.npz")
    fold_data = get_kfold_data(all_paths, 5)[fold]

    train_datalist = fold_data["train_data"]
    val_datalist = fold_data["val_data"]  

    print(f"training data is {len(train_datalist)}")
    print(f"validation data is {len(val_datalist)}")
    train_ds = MedicalDataset(train_datalist)
    
    val_ds = MedicalDataset(val_datalist)

    if test_dir is not None:
        test_paths = glob.glob(f"{test_dir}/*.npz")
        test_ds = MedicalDataset(test_paths, test=True)
    else:
        test_ds = None 

    loader = [train_ds, val_ds, test_ds]

    return loader

def get_all_training_loader(data_dir, fold=0, test_dir=None):
    ## train all labeled data 
    ## fold denote the validation data in training data
    all_paths = glob.glob(f"{data_dir}/*.npz")
    fold_data = get_kfold_data(all_paths, 5)[fold]

    train_datalist = all_paths
    val_datalist = fold_data["val_data"]  

    print(f"training data is {len(train_datalist)}")
    print(f"validation data is {len(val_datalist)}")
    train_ds = MedicalDataset(train_datalist)
    
    val_ds = MedicalDataset(val_datalist)

    if test_dir is not None:
        test_paths = glob.glob(f"{test_dir}/*.npz")
        test_ds = MedicalDataset(test_paths, test=True)
    else:
        test_ds = None 

    loader = [train_ds, val_ds, test_ds]

    return loader

def get_train_val_test_loader_seperate(train_dir, val_dir, test_dir=None):
    train_datalist = glob.glob(f"{train_dir}/*.npz")
    val_datalist = glob.glob(f"{val_dir}/*.npz")

    print(f"training data is {len(train_datalist)}")
    print(f"validation data is {len(val_datalist)}")

    if test_dir is not None:
        test_datalist = glob.glob(f"{test_dir}/*.npz")
        print(f"test data is {len(test_datalist)}")
        test_ds = MedicalDataset(test_datalist, test=True)
    else :
        test_ds = None

    train_ds = MedicalDataset(train_datalist)
    val_ds = MedicalDataset(val_datalist)
    
    loader = [train_ds, val_ds, test_ds]

    return loader

def get_train_val_test_loader_from_split_json(data_dir, split_json_file):
    import json 

    with open(split_json_file, "r") as f:

        datalist = json.loads(f.read())

    train_datalist = datalist["train"]
    val_datalist = datalist["validation"]
    test_datalist = datalist["test"]

    def add_pre(datalist):
        for i in range(len(datalist)):
            datalist[i] = os.path.join(data_dir, datalist[i])
    
    add_pre(train_datalist)
    add_pre(val_datalist)
    add_pre(test_datalist)
    print(f"training data is {len(train_datalist)}")
    print(f"validation data is {len(val_datalist)}")
    print(f"test data is {len(test_datalist)}", sorted(test_datalist))

    train_ds = MedicalDataset(train_datalist)
    val_ds = MedicalDataset(val_datalist)
    test_ds = MedicalDataset(test_datalist)

    loader = [train_ds, val_ds, test_ds]

    return loader


# Paper split lives in lowo/splits/. glob.glob() order is readdir() order and
# is not stable across machines, so shuffle(seed=42) alone does not reproduce
# the 875/125/250 partition used in the paper. Frozen case lists are the
# source of truth; they are an internal held-out subset of the BraTS2021
# training pool, not the official BraTS test set.
_DEFAULT_SPLIT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "splits")
)


def _case_stem(path_or_name):
    return os.path.splitext(os.path.basename(str(path_or_name)))[0]


def read_case_list(list_path):
    """One CaseID per line; '#' comments and blank lines ignored."""
    names = []
    with open(list_path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                names.append(_case_stem(s))
    return names


def write_case_list(list_path, names, header_lines=None):
    os.makedirs(os.path.dirname(os.path.abspath(list_path)) or ".", exist_ok=True)
    with open(list_path, "w", encoding="utf-8") as f:
        for h in header_lines or []:
            f.write(f"# {h}\n")
        for n in names:
            f.write(f"{_case_stem(n)}\n")


def _npz_index(data_dir):
    return {_case_stem(p): p for p in glob.glob(os.path.join(data_dir, "*.npz"))}


def _paths_from_names(index, names, tag):
    missing = [n for n in names if n not in index]
    if missing:
        raise FileNotFoundError(
            f"{tag}: {len(missing)}/{len(names)} case(s) not found under the "
            f"npz directory. e.g. {missing[:5]}")
    return [index[n] for n in names]


def _dump_split(dump_dir, train_names, val_names, test_names):
    if not dump_dir:
        return
    header = [
        "Internal held-out subset of the BraTS2021 training pool.",
        "Not the official BraTS test set. One CaseID per line.",
    ]
    write_case_list(os.path.join(dump_dir, "train_875.txt"), train_names,
                    header + [f"{len(train_names)} training cases"])
    write_case_list(os.path.join(dump_dir, "val_125.txt"), val_names,
                    header + [f"{len(val_names)} validation cases"])
    write_case_list(os.path.join(dump_dir, "test_250.txt"), test_names,
                    header + [f"{len(test_names)} held-out cases"])
    print(f"[split] wrote train/val/test lists under {dump_dir}")


def get_train_val_test_loader_from_train(
        data_dir, train_rate=0.7, val_rate=0.1, test_rate=0.2, seed=42,
        split_dir=None, dump_split_dir=None):
    """Load the paper split from frozen case lists when present.

    Fallback (no lists): sort glob, then shuffle with ``seed``. That fallback
    is portable across machines but is *not* the paper partition, because the
    original run shuffled an unsorted glob. Frozen ``test_250.txt`` /
    ``val_125.txt`` / ``train_875.txt`` under ``split_dir`` are required to
    reproduce Table 1.
    """
    index = _npz_index(data_dir)
    if not index:
        raise FileNotFoundError(f"no *.npz cases under {data_dir}")

    split_dir = _DEFAULT_SPLIT_DIR if split_dir is None else split_dir
    list_files = {
        "train": os.path.join(split_dir, "train_875.txt"),
        "val": os.path.join(split_dir, "val_125.txt"),
        "test": os.path.join(split_dir, "test_250.txt"),
    }
    frozen = {k: read_case_list(p) for k, p in list_files.items()
              if os.path.isfile(p)}
    if "test" in frozen:
        missing = [n for n in frozen["test"] if n not in index]
        if missing:
            print(f"[split] frozen test_250.txt does not match {data_dir} "
                  f"({len(missing)}/{len(frozen['test'])} missing); "
                  "ignoring frozen lists for this data directory.")
            frozen = {}

    if "test" in frozen:
        test_names = frozen["test"]
        used_test = set(test_names)
        if "val" in frozen:
            val_names = frozen["val"]
            if used_test.intersection(val_names):
                raise ValueError("frozen val/test lists overlap")
            if "train" in frozen:
                train_names = frozen["train"]
            else:
                held = used_test.union(val_names)
                train_names = sorted(n for n in index if n not in held)
            print(f"[split] frozen lists from {split_dir} "
                  f"(train={len(train_names)}, val={len(val_names)}, "
                  f"test={len(test_names)})")
        else:
            print("[split] WARNING: test_250.txt is frozen but val_125.txt is "
                  "missing. Train/val are the complement of test, re-split "
                  f"with sorted glob and seed={seed}. This avoids test "
                  "contamination but is NOT the paper validation set.")
            rest = sorted(n for n in index if n not in used_test)
            random.seed(seed)
            random.shuffle(rest)
            n_val = int(len(index) * val_rate)
            val_names = rest[:n_val]
            train_names = rest[n_val:]
    else:
        print("[split] WARNING: no frozen test_250.txt; deriving a NEW split "
              f"from sorted glob + shuffle(seed={seed}). This is portable "
              "but is NOT the paper partition (original glob was unsorted).")
        names = sorted(index)
        random.seed(seed)
        random.shuffle(names)
        n_train = int(len(names) * train_rate)
        n_val = int(len(names) * val_rate)
        n_test = int(len(names) * test_rate)
        train_names = names[:n_train]
        val_names = names[n_train:n_train + n_val]
        test_names = names[-n_test:]

    _dump_split(dump_split_dir, train_names, val_names, test_names)

    train_datalist = _paths_from_names(index, train_names, "train")
    val_datalist = _paths_from_names(index, val_names, "val")
    test_datalist = _paths_from_names(index, test_names, "test")

    print(f"training data is {len(train_datalist)}")
    print(f"validation data is {len(val_datalist)}")
    print(f"test data is {len(test_datalist)}", sorted(test_datalist))

    train_ds = MedicalDataset(train_datalist)
    val_ds = MedicalDataset(val_datalist)
    test_ds = MedicalDataset(test_datalist)

    return [train_ds, val_ds, test_ds]

def get_train_loader_from_train(data_dir):
    ## train all labeled data 
    ## fold denote the validation data in training data
    all_paths = glob.glob(f"{data_dir}/*.npz")
    # fold_data = get_kfold_data(all_paths, 5)[fold]

    train_ds = MedicalDataset(all_paths)
  
    return train_ds

def get_test_loader_from_test(data_dir):
    all_paths = glob.glob(f"{data_dir}/*.npz")

    test_ds = MedicalDataset(all_paths)
  
    return test_ds

def get_multi_dir_training_loader(data_dir, fold=0, test_dir=None):
    ## train all labeled data 
    ## fold denote the validation data in training data
    all_paths = []
    for p in data_dir:
        paths = glob.glob(f"{p}/*.npz")
        for pp in paths:
            all_paths.append(pp)

    # print(all_paths)
    fold_data = get_kfold_data(all_paths, 5)[fold]

    train_datalist = all_paths
    val_datalist = fold_data["val_data"]  

    print(f"training data is {len(train_datalist)}")
    print(f"validation data is {len(val_datalist)}")
    train_ds = MedicalDataset(train_datalist)
    
    val_ds = MedicalDataset(val_datalist)

    if test_dir is not None:
        test_paths = glob.glob(f"{test_dir}/*.npz")
        test_ds = MedicalDataset(test_paths, test=True)
    else:
        test_ds = None 

    loader = [train_ds, val_ds, test_ds]

    return loader