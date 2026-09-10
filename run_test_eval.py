"""Config-driven test-set evaluation for SegMamba / Haar(LOWO) / SegAttn variants.

Why this script exists
----------------------
The two existing eval scripts (``analyze_errors_brats2021.py`` and
``evaluate_tta_postproc.py``) hard-code ``SegMamba`` and therefore cannot
evaluate any Haar/LOWO/SegAttn checkpoint.  Training itself was launched with
the model architecture taken from per-experiment config YAMLs (the training
scripts keep the *values* in module-level variables; the YAML is the source of
truth for each run).  This driver builds the model **from the config** so the
architecture (incl. ``grouped_mix``, ``per_channel_*``, ``use_routing``, ...)
exactly matches the checkpoint, then runs the unified protocol:

    * 8-way mirror Test-Time Augmentation (logit averaging)
    * sliding-window inference, overlap = 0.7 (consensus setting)
    * ET -> NCR total-volume-threshold postprocessing (nnU-Net BraTS style)
    * per-case Dice + HD95 + FP/FN + boundary/interior error split
    * writes ``per_case_metrics.csv`` in the exact format of
      ``analyze_errors_brats2021.py`` (so existing aggregate tooling works)

Usage
-----
    python run_test_eval.py --config configs/A6_so8_g8.yaml \
        --ckpt /path/to/best_model_0.9317.pt --dry_run

    python run_test_eval.py --config configs/A6_so8_g8.yaml \
        --ckpt /path/to/best_model_0.9317.pt \
        --force_ethr 50 --test_list splits/test_250.txt --stage test

The dry-run is the important safety net: it builds the model from the config
and does a STRICT ``load_state_dict`` on CPU.  If the config does not match the
checkpoint architecture, it fails loudly *before* you burn GPU time.
"""
import argparse
import csv
import glob
import inspect
import itertools
import json
import os
import re
import sys
import time

import numpy as np
import yaml

# ---- repo imports (run from the segMamba repo root) -------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from monai.inferers import SlidingWindowInferer
from monai.metrics import compute_hausdorff_distance
from monai.utils import set_determinism
from scipy import ndimage
from tqdm import tqdm
set_determinism(123)

from light_training.dataloading.dataset import get_train_val_test_loader_from_train


# ============================================================================
# Model construction (config-driven)
# ============================================================================
MODEL_REGISTRY = {
    # key: (module, class, forced-arg-family)
    "segmamba":      ("model_segmamba.segmamba",      "SegMamba",     "brats"),
    "segmamba_haar": ("model_segmamba.segmamba_haar", "SegMambaHaar", "brats"),
    "segattn_haar":  ("model_segmamba.segmamba_attn", "SegAttnHaar",  "brats"),
    "swin_unetr":    ("monai.networks.nets",          "SwinUNETR",    "monai"),
}

# Architecture fields forced for every BraTS run (identical across variants).
# Two families: the SegMamba family takes in_chans/out_chans/feat_size, while
# MONAI nets take in_channels/out_channels/feature_size. Forcing the wrong
# spelling would raise an unexpected-keyword error at construction, so each
# family gets its own dict and we additionally filter to the class's real
# signature below.
FORCED = {
    "brats": dict(in_chans=4, out_chans=4,
                  depths=[2, 2, 2, 2], feat_size=[48, 96, 192, 384]),
    "monai": dict(in_channels=4, out_channels=4, spatial_dims=3),
}


def build_model_from_config(cfg, device, strict=True):
    """Instantiate the model class named by cfg['model'] using only the config
    keys that are valid __init__ parameters, plus the forced BraTS fields."""
    key = cfg.get("model", "segmamba")
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown cfg['model']={key!r}; "
                         f"known: {list(MODEL_REGISTRY)}")
    mod_name, cls_name, family = MODEL_REGISTRY[key]
    mod = __import__(mod_name, fromlist=[cls_name])
    cls = getattr(mod, cls_name)

    sig = inspect.signature(cls.__init__)
    valid = set(sig.parameters) - {"self"}
    kwargs = {}
    for k, v in cfg.items():
        if k in valid:
            kwargs[k] = v
    # forced fields win, but keep only what this class actually accepts
    forced = {k: v for k, v in FORCED[family].items() if k in valid}
    kwargs.update(forced)
    model = cls(**kwargs)
    return model.to(device).eval()


def filte_state_dict(sd):
    if "module" in sd:
        sd = sd["module"]
    if "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    return {(k[7:] if k.startswith("module") else k): v for k, v in sd.items()}


# Parameter-name prefixes that may be ABSENT from a legacy checkpoint without
# changing inference output.  Deep-supervision heads (ds_out1/2/3) only add
# auxiliary losses during training; they are not executed at inference, so a
# checkpoint trained before deep supervision existed can still be evaluated
# faithfully by leaving them at their random init.
#
# NOTE: `*.wave.skew` (the SO(8) generator) is deliberately NOT ignorable --
# a missing skew means the wavelet basis itself is undefined, so such a
# checkpoint would produce meaningless output and must still be SKIPped.
IGNORABLE_MISSING = ("ds_out",)


def load_or_skip(model, sd, out_dir, ckpt, dry, allow_partial=False,
                 ignorable=IGNORABLE_MISSING):
    """Load sd into model. Default behaviour is unchanged: STRICT load, and on
    any architecture drift write a SKIP marker and return False so the caller
    moves on to the next variant instead of aborting the loop.

    With allow_partial=True, keys missing from the checkpoint are tolerated
    *only* if they fall under one of the `ignorable` prefixes (default
    `ds_out*`, the deep-supervision heads). Unexpected keys (present in the
    checkpoint, absent from the model) are dropped. Any *other* missing key is
    still fatal and falls through to the normal SKIP path -- e.g. a missing
    `.wave.skew` means the wavelet basis is undefined and must not be guessed.

    drift = old training code / old monai version whose parameter names no
    longer match the current model definition (e.g. legacy `.wave.skew`,
    `ds_out*.conv.conv.*` keys).
    """
    model_sd = model.state_dict()
    missing = [k for k in model_sd if k not in sd]
    unexpected = [k for k in sd if k not in model_sd]

    # --- exact match: plain strict load (fast path, identical to before) ---
    if not missing and not unexpected:
        model.load_state_dict(sd, strict=True)
        return True

    # --- opt-in partial load ------------------------------------------------
    if allow_partial:
        # Missing *runtime-cache* buffers (LOWO `*_k_cache` FFT/kernel
        # caches) are safe to tolerate: they are recomputed lazily on the
        # first forward pass and were never persisted by older training
        # code, so their absence in a checkpoint is expected and harmless.
        # A missing *parameter* (e.g. `.wave.skew`) or a *running-stat*
        # buffer (BN running_mean/var) WOULD change inference output and
        # stays fatal -- only cache-named buffers are reclassified here.
        cache_buffer_names = {
            name for name, _ in model.named_buffers()
            if name.endswith("_k_cache")
        }

        tolerated = [k for k in missing if k.startswith(ignorable)]
        tolerated += [k for k in missing if k in cache_buffer_names]
        fatal = [k for k in missing if k not in tolerated]

        if not fatal:
            sd = {k: v for k, v in sd.items() if k in model_sd}
            model.load_state_dict(sd, strict=False)
            roots = sorted({k.split(".")[0] for k in tolerated})
            n_cache = sum(1 for k in tolerated if k in cache_buffer_names)
            print(f"[partial-load OK] {ckpt}\n"
                  f"  tolerated missing ({len(tolerated)}): {roots} "
                  f"-- not executed at inference\n"
                  f"    of which runtime-cache buffers ({n_cache}) "
                  f"-- recomputed on first forward\n"
                  f"  dropped unexpected ({len(unexpected)}): "
                  f"{unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")
            return True
        # non-ignorable drift: refuse, but explain precisely why
        head = (f"[SKIP] partial_load refused for:\n  {ckpt}\n"
                f"  ignorable prefixes: {ignorable}\n"
                f"  missing, tolerated ({len(tolerated)}): {tolerated[:5]}"
                f"{' ...' if len(tolerated) > 5 else ''}\n"
                f"  missing, FATAL     ({len(fatal)}): {fatal[:5]}"
                f"{' ...' if len(fatal) > 5 else ''}\n")
        note = ("  -> these keys affect inference output; leaving them at init "
                "would give meaningless metrics. Eval this checkpoint with the "
                "matching legacy model definition instead.\n")
        print(head + note)
        try:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "SKIP_arch_mismatch.txt"), "w") as f:
                f.write(head + note + f"\n(dry_run={dry})\n")
        except OSError:
            pass
        return False

    # --- default strict path (original behaviour) ---------------------------
    if missing or unexpected:
        head = (f"[SKIP] config<->ckpt architecture mismatch for:\n  {ckpt}\n"
                f"  missing in ckpt   ({len(missing)}):   {missing[:10]}"
                f"{' ...' if len(missing) > 10 else ''}\n"
                f"  unexpected in ckpt({len(unexpected)}): {unexpected[:10]}"
                f"{' ...' if len(unexpected) > 10 else ''}\n")
        note = ("  -> architecture drift (likely trained with an older model "
                "code / monai version). To eval this checkpoint faithfully you "
                "must build the model with the matching legacy definition.\n"
                "     Tip: if the missing keys are only deep-supervision heads "
                "(ds_out*), re-run with --allow_partial_load.\n")
        print(head + note)
        try:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "SKIP_arch_mismatch.txt"), "w") as f:
                f.write(head + note + f"\n(dry_run={dry})\n")
        except OSError:
            pass
        return False
    model.load_state_dict(sd, strict=True)
    return True



def find_best_ckpt(ckpt_dir):
    """Pick the best_model_<dice>.pt with the highest dice in the dir tree."""
    cands = sorted(
        glob.glob(os.path.join(ckpt_dir, "**", "best_model_*.pt"), recursive=True),
        key=lambda p: float(re.search(r"best_model_([0-9.]+)\.pt$", p).group(1))
        if re.search(r"best_model_([0-9.]+)\.pt$", p) else -1.0,
        reverse=True)
    if not cands:
        raise FileNotFoundError(f"No best_model_*.pt under {ckpt_dir}")
    return cands[0]


# ============================================================================
# Inference + metrics (GPU-accelerated HD95 + progress bars)
# ============================================================================
MIRROR_COMBOS = [c for n in range(4) for c in itertools.combinations((2, 3, 4), n)]


def tta_predict(img, model, inferer, use_tta=True):
    """img: (1,4,D,H,W) on device -> argmax label volume (D,H,W) int8."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        acc = None
        combos = MIRROR_COMBOS if use_tta else [()]
        for axes in combos:
            x = torch.flip(img, dims=axes) if axes else img
            y = inferer(x, model)
            if axes:
                y = torch.flip(y, dims=axes)
            if acc is None:
                acc = y.float()
            else:
                acc += y.float()       # fp32 accumulation (safe for argmax)
            del y
        acc /= len(combos)
    return acc.argmax(dim=1)[0].cpu().numpy().astype(np.int8)


def apply_et_to_ncr(pred, thr):
    """If total predicted ET volume < thr, relabel ET voxels to NCR (1)."""
    et = pred == 3
    if 0 < et.sum() < thr:
        pred = pred.copy()
        pred[et] = 1
    return pred


def region_masks(lbl):
    tc = (lbl == 1) | (lbl == 3)
    wt = tc | (lbl == 2)
    et = lbl == 3
    return {"TC": tc, "WT": wt, "ET": et}


def surface_dist_hd95_gpu(gt_mask, pred_mask, device="cuda"):
    """GPU-accelerated HD95 using MONAI's compute_hausdorff_distance.

    Falls back to CPU scipy if GPU fails (e.g. empty masks on GPU).
    """
    if gt_mask.sum() == 0 or pred_mask.sum() == 0:
        return float("nan")
    try:
        gt_t = torch.from_numpy(gt_mask.astype(np.float32)[None, None]).to(device)
        pr_t = torch.from_numpy(pred_mask.astype(np.float32)[None, None]).to(device)
        hd95 = compute_hausdorff_distance(pr_t, gt_t, include_background=True, percentile=95)
        result = hd95.item()
        del gt_t, pr_t
        return result
    except Exception:
        # Fallback to CPU scipy
        return surface_dist_hd95_cpu(gt_mask, pred_mask)


def surface_dist_hd95_cpu(gt, pred):
    """CPU fallback for HD95 using scipy."""
    if gt.sum() == 0 or pred.sum() == 0:
        return float("nan")
    st = ndimage.generate_binary_structure(3, 1)
    gs = gt ^ ndimage.binary_erosion(gt, st)
    ps = pred ^ ndimage.binary_erosion(pred, st)
    dg = ndimage.distance_transform_edt(~gs)
    dp = ndimage.distance_transform_edt(~ps)
    d1 = dp[gs]; d2 = dg[ps]
    return float(np.percentile(np.hstack([d1, d2]), 95))


def boundary_distance_gpu(gt_mask, device="cuda"):
    """GPU-accelerated boundary + interior distance maps using torch.

    Returns (dg, dgi) where:
      dg  = distance from each voxel to the nearest boundary voxel (outside mask)
      dgi = distance from each voxel to the nearest background voxel (inside mask)

    Used for boundary-error computation. Falls back to scipy CPU on failure.
    """
    try:
        # Distance from boundary (erode mask, compute distance from complement of eroded)
        mask_t = torch.from_numpy(gt_mask.astype(np.float32)[None, None]).to(device)
        # Use MONAI's distance transform via a simple approach:
        # boundary = mask XOR eroded_mask
        # We can compute this on CPU with scipy since it's only 2 EDT calls
        # (much faster than HD95 which needs surface extraction + 2 EDT per region)
        dg = ndimage.distance_transform_edt(~gt_mask)
        dgi = ndimage.distance_transform_edt(gt_mask)
        return dg, dgi
    except Exception:
        dg = ndimage.distance_transform_edt(~gt_mask)
        dgi = ndimage.distance_transform_edt(gt_mask)
        return dg, dgi


def analyze_case(gt_lbl, pr_lbl, device="cuda"):
    out = {}
    g = region_masks(gt_lbl); p = region_masks(pr_lbl)
    for cls in ("TC", "WT", "ET"):
        gm, pm = g[cls], p[cls]
        inter = (gm & pm).sum()
        gv, pv = int(gm.sum()), int(pm.sum())
        dice = 1.0 if gv + pv == 0 else 2 * inter / (gv + pv)
        fn = int((gm & ~pm).sum()); fp = int((~gm & pm).sum())
        err = (gm ^ pm)
        if gv:
            dg, dgi = boundary_distance_gpu(gm, device=device)
            band = ((dg <= 2) & (dgi <= 2)) | ((dg > 0) & (dg <= 2)) | ((dgi > 0) & (dgi <= 2))
            bnd_err = int((err & band).sum())
        else:
            bnd_err = 0
        out[cls] = dict(dice=dice, gv=gv, pv=pv, fn=fn, fp=fp,
                        bnd_err=bnd_err, int_err=int(err.sum()) - bnd_err,
                        hd95=surface_dist_hd95_gpu(gm, pm, device=device))
    return out


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2 * (a & b).sum() / s)


def region_dices(gt, pred):
    return {c: dice(region_masks(gt)[c], region_masks(pred)[c]) for c in ("TC", "WT", "ET")}


# ============================================================================
# Frozen-split override
# ============================================================================
# get_train_val_test_loader_from_train() derives the split from
# glob.glob(), whose order is whatever readdir() returns -- it is NOT stable
# across machines, filesystems, or even re-copies of the same data. The seed is
# fixed (42) but the *input order* is not, so the same code yields a different
# train/val/test partition on a different host. Evaluating on a re-derived
# split silently scores cases the model was trained on (measured contamination
# on this project: 201/250 = 80.4%), which inflates Dice and makes arms
# mutually incomparable.
#
# Passing --test_list (and optionally --val_list) rebuilds the split from an
# explicit case list so results are reproducible on ANY host.
def load_frozen_split(ds, list_path, tag, data_dir):
    """Rebuild `ds` from a frozen case list. Returns ds unchanged if no list."""
    if not list_path:
        return ds
    if not os.path.exists(list_path):
        raise FileNotFoundError(f"{tag}_list not found: {list_path}")
    names = [l.strip() for l in open(list_path, encoding="utf-8")
             if l.strip() and not l.startswith("#")]
    index = {os.path.splitext(os.path.basename(p))[0]: p
             for p in glob.glob(os.path.join(data_dir, "*.npz"))}
    missing = [n for n in names if n not in index]
    if missing:
        raise FileNotFoundError(
            f"{tag}_list: {len(missing)}/{len(names)} case(s) not found under "
            f"{data_dir}. e.g. {missing[:5]}. Check --data_dir and that case "
            f"names are <CaseID>.npz stems.")
    from light_training.dataloading.dataset import MedicalDataset
    paths = [index[n] for n in names]
    print(f"[frozen split] {tag}: rebuilt from {os.path.basename(list_path)} "
          f"({len(paths)} cases) -- comparable across hosts", flush=True)
    return MedicalDataset(paths)


# ============================================================================
# Split runner + threshold search + final report
# ============================================================================
def run_split(ds, model, inferer, out_dir, tag, use_tta, max_cases, roi=128, memory_safe=True):
    os.makedirs(out_dir, exist_ok=True)
    n = len(ds) if max_cases == 0 else min(max_cases, len(ds))
    names, t0 = [], time.time()
    # Collect names first for progress bar
    existing = 0
    for idx in range(n):
        item = ds[idx]
        name = item["properties"]["name"]
        names.append(name)
        f = os.path.join(out_dir, f"tta_{tag}_{name}.npz")
        if os.path.exists(f):
            existing += 1
    if existing == n:
        print(f"[{tag}] All {n} cases already inferred, skipping inference.", flush=True)
        return names

    # Run inference with progress bar
    for idx in tqdm(range(n), desc=f"[{tag}] inference", unit="case",
                    total=n, initial=existing):
        item = ds[idx]
        name = item["properties"]["name"]
        f = os.path.join(out_dir, f"tta_{tag}_{name}.npz")
        if os.path.exists(f):
            continue
        img = torch.from_numpy(np.ascontiguousarray(item["data"])).float()[None].cuda()
        gt = np.asarray(item["seg"][0]).astype(np.int8); gt[gt < 0] = 0
        pred = tta_predict(img, model, inferer, use_tta)
        del img
        if memory_safe:
            torch.cuda.empty_cache()
        np.savez_compressed(f, pred=pred, gt=gt)
        del pred, gt
        if memory_safe:
            torch.cuda.empty_cache()
    return names


def evaluate(names, out_dir, tag, thr, device="cuda"):
    rows = []
    for name in tqdm(names, desc=f"  thr={thr:5d}", unit="case", leave=True):
        z = np.load(os.path.join(out_dir, f"tta_{tag}_{name}.npz"))
        pred, gt = z["pred"], z["gt"]
        pred = apply_et_to_ncr(pred, thr)
        r = analyze_case(gt.astype(np.int16), pred.astype(np.int16), device=device)
        row = {"name": name}
        for c in ("TC", "WT", "ET"):
            for k, v in r[c].items():
                row[f"{c}_{k}"] = v
        row["mean_dice"] = float(np.mean([r[c]["dice"] for c in ("TC", "WT", "ET")]))
        rows.append(row)
    return rows


def aggregate_report(rows, title):
    print(f"\n================ {title} ================")
    for cls in ("TC", "WT", "ET"):
        d = np.array([r[f"{cls}_dice"] for r in rows])
        h = np.array([r[f"{cls}_hd95"] for r in rows], dtype=float)
        fn = np.array([r[f"{cls}_fn"] for r in rows]); fp = np.array([r[f"{cls}_fp"] for r in rows])
        be = np.array([r[f"{cls}_bnd_err"] for r in rows]); ie = np.array([r[f"{cls}_int_err"] for r in rows])
        print(f"{cls}: dice mean={d.mean():.4f} median={np.median(d):.4f} "
              f"<0.8: {(d<0.8).sum()} cases | HD95 median={np.nanmedian(h):.2f} | "
              f"FN/FP voxel ratio={fn.sum()/max(fp.sum(),1):.2f} | "
              f"boundary-err share={be.sum()/(be.sum()+ie.sum()):.2%}")
    md = float(np.mean([r["mean_dice"] for r in rows]))
    print(f"OVERALL mean(Dice TC,WT,ET) = {md:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="")
    ap.add_argument("--ckpt", default="", help="explicit .pt; else auto-pick best in --ckpt_dir")
    ap.add_argument("--ckpt_dir", default="")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--stage", default="all", choices=["val", "test", "all"])
    ap.add_argument("--overlap", type=float, default=0.7)
    ap.add_argument("--no_tta", action="store_true")
    ap.add_argument("--max_cases", type=int, default=0)
    ap.add_argument("--thresholds", default="0,50,100,200,300,500,750,1000,1500,2000")
    ap.add_argument("--dry_run", action="store_true",
                    help="build model + strict load ckpt on CPU, then exit (no GPU/data)")
    ap.add_argument("--manifest", default="", help="test_eval_manifest.py")
    ap.add_argument("--index", type=int, default=0)
    # portable single data root (override on the Ubuntu box with the Linux mount path)
    ap.add_argument("--data_root", default=".",
                    help="root that contains ./logs (ckpts) and ./analysis_results (outputs)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no_batch_trim", action="store_true",
                    help="run at full speed (no per-case torch.cuda.empty_cache, "
                         "keep sw_batch_size=4). Use only when GPU memory is ample; "
                         "omit to enable the memory-saving trim.")
    ap.add_argument("--cpu_metrics", action="store_true",
                    help="fall back to CPU scipy for HD95/boundary metrics "
                         "(slower but uses no GPU memory for metrics)")
    ap.add_argument("--allow_partial_load", action="store_true",
                    help="tolerate checkpoint keys that are missing under an "
                         "'ignorable' prefix (default: ds_out, the "
                         "deep-supervision heads, which are not executed at "
                         "inference). Lets legacy checkpoints trained before "
                         "deep supervision be evaluated. DEFAULT OFF = strict "
                         "load, i.e. the original behaviour.")
    ap.add_argument("--ignore_missing", default="ds_out",
                    help="comma-separated key prefixes tolerated by "
                         "--allow_partial_load (default: ds_out)")
    ap.add_argument("--data_dir", default="./data/fullres/train",
                    help="directory of *.npz cases used to build the split")
    ap.add_argument("--test_list", default="",
                    help="frozen test case list (one CaseID per line, '#' "
                         "comments allowed). Strongly recommended: without it "
                         "the split is re-derived from glob.glob() order and "
                         "differs between hosts, which silently contaminates "
                         "the test set.")
    ap.add_argument("--val_list", default="",
                    help="frozen val case list, same format as --test_list")
    ap.add_argument("--force_ethr", type=int, default=None,
                    help="skip per-arm val threshold scan; force ALL arms to use "
                         "the same fixed ET->NCR threshold (e.g. 50). "
                         "Makes cross-arm comparison immune to threshold-fitting noise.")
    args = ap.parse_args()

    metrics_device = "cpu" if args.cpu_metrics else args.device
    ignorable = tuple(p.strip() for p in args.ignore_missing.split(",") if p.strip())

    # ---- resolve config / ckpt from manifest if given -----------------------
    if args.manifest:
        mdir = os.path.dirname(os.path.abspath(args.manifest))
        if mdir not in sys.path:
            sys.path.insert(0, mdir)
        import test_eval_manifest as M
        entry = M.MANIFEST[args.index]
        cfg_path = entry["config"]
        # A manifest entry may use an ABSOLUTE ckpt_rel (e.g. a Linux path for a
        # checkpoint that lives outside --data_root); otherwise ckpt_rel is
        # RELATIVE to --data_root. os.path.isabs is platform-correct: a "G:/..."
        # path is absolute on Windows but NOT on Linux, so it is only honored as
        # an override where it is genuinely absolute.
        ckpt_dir = (entry["ckpt_rel"] if os.path.isabs(entry["ckpt_rel"])
                    else os.path.join(args.data_root, entry["ckpt_rel"]))
        out_name = entry.get("out", entry["name"])
        out_dir = os.path.join(args.data_root, "analysis_results", out_name)
        name = out_name
        print(f"[manifest #{args.index}] {name}\n  config={cfg_path}\n  ckpt_dir={ckpt_dir}\n  out={out_dir}")
    else:
        cfg_path = args.config
        ckpt_dir = args.ckpt_dir
        out_dir = args.out_dir or os.path.join(args.data_root, "analysis_results",
                                               os.path.basename(ckpt_dir))
        name = os.path.basename(ckpt_dir)

    if not cfg_path or not ckpt_dir:
        ap.error("provide --config + --ckpt_dir, or --manifest + --index")

    cfg = yaml.safe_load(open(cfg_path))
    model_key = cfg.get("model")
    print(f"[build] config={cfg_path}  model={model_key}")

    device = torch.device("cpu") if args.dry_run else torch.device(args.device)

    model = build_model_from_config(cfg, device, strict=True)
    ckpt = args.ckpt or find_best_ckpt(ckpt_dir)
    sd = filte_state_dict(torch.load(ckpt, map_location="cpu"))

    # ---- architecture match gate (dry & real share this) -------------------
    # On drift, writes <out_dir>/SKIP_arch_mismatch.txt and returns so the
    # per-variant loop keeps going instead of aborting on the first bad ckpt.
    if not load_or_skip(model, sd, out_dir, ckpt, args.dry_run,
                        allow_partial=args.allow_partial_load,
                        ignorable=ignorable):
        return

    if args.dry_run:
        nparams = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[dry_run OK] ckpt={ckpt}\n  params={nparams:.1f}M")
        print("  -> checkpoint loads under the current config. Safe to run for real.")
        return

    print(f"[load] {ckpt}  ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

    memory_safe = not args.no_batch_trim
    sw_batch = 2 if memory_safe else 4
    # Per-model override: heavy nets (e.g. SwinUNETR, which trains at
    # sw_batch_size=1) may still OOM at 2. Set `eval_sw_batch` in the config.
    if cfg.get("eval_sw_batch"):
        sw_batch = int(cfg["eval_sw_batch"])
    inferer = SlidingWindowInferer(roi_size=[128, 128, 128], sw_batch_size=sw_batch,
                                   overlap=args.overlap, mode="gaussian")
    _, val_ds, test_ds = get_train_val_test_loader_from_train(args.data_dir)
    # Override with frozen lists when given -- otherwise the split is whatever
    # this host's readdir() produced, which is NOT reproducible (see
    # load_frozen_split docstring). Always pass --test_list for comparable
    # numbers, especially when moving to a different machine.
    test_ds = load_frozen_split(test_ds, args.test_list, "test", args.data_dir)
    val_ds = load_frozen_split(val_ds, args.val_list, "val", args.data_dir)
    thr_file = os.path.join(out_dir, "best_thr.json")
    thresholds = [int(t) for t in args.thresholds.split(",")]

    if args.stage in ("val", "all"):
        names = run_split(val_ds, model, inferer, out_dir, "val",
                          not args.no_tta, args.max_cases, memory_safe=memory_safe)
        if args.force_ethr is not None:
            # Unified protocol: skip the per-arm threshold scan entirely. Each arm
            # would otherwise fit its own val-threshold, which breaks cross-arm and
            # cross-seed comparability. Use the SAME threshold for every arm.
            best = args.force_ethr
            json.dump({"best_thr": best, "forced": True}, open(thr_file, "w"))
            print(f"[val] forced ET->NCR thr = {best} (uniform protocol, no scan) -> {thr_file}")
        else:
            results = []
            print(f"[val] Computing metrics for {len(thresholds)} thresholds "
                  f"on {len(names)} cases (GPU-accelerated HD95)...")
            for thr in thresholds:
                rows = evaluate(names, out_dir, "val", thr, device=metrics_device)
                m = {c: float(np.mean([r[f"{c}_dice"] for r in rows])) for c in ("TC", "WT", "ET")}
                results.append((thr, m["ET"]))
                print(f"  thr={thr:5d}: TC={m['TC']:.4f} WT={m['WT']:.4f} ET={m['ET']:.4f}")
            best_et = max(r[1] for r in results)
            best = min((r for r in results if r[1] >= best_et - 1e-9), key=lambda r: r[0])
            json.dump({"best_thr": best[0], "val_et_dice": best[1]}, open(thr_file, "w"))
            print(f"[val] best thr = {best[0]} (val ET dice {best[1]:.4f}) -> {thr_file}")

    if args.stage in ("test", "all"):
        # force_ethr wins over the stored file -- guarantees the same threshold
        # is applied regardless of what an earlier per-arm scan wrote.
        if args.force_ethr is not None:
            thr = args.force_ethr
        else:
            thr = json.load(open(thr_file))["best_thr"] if os.path.exists(thr_file) else 500
        names = run_split(test_ds, model, inferer, out_dir, "test",
                          not args.no_tta, args.max_cases, memory_safe=memory_safe)
        print(f"[test] Computing metrics for {len(names)} cases (GPU-accelerated HD95)...")
        rows = evaluate(names, out_dir, "test", thr, device=metrics_device)
        keys = list(rows[0].keys())
        with open(os.path.join(out_dir, "per_case_metrics.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        aggregate_report(rows, f"TEST  {name}  (TTA={'off' if args.no_tta else 'on'}, ET->NCR thr={thr})")
        print(f"[test] per_case_metrics.csv -> {os.path.join(out_dir, 'per_case_metrics.csv')}")


if __name__ == "__main__":
    main()
