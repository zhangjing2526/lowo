#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
measure_latency.py — Conditional-compute latency harness for SegAttn-Haar (routed).

Purpose
-------
Measure wall-clock latency (ms/volume) and peak VRAM for a *trained* routed
SegAttn-Haar under two deployment modes:

  soft : both branches computed, gate blends  (supernet, training semantics)
  hard : set_hard_routing(True) -> per-block branch decision; ONLY the
         selected branch is computed  (real conditional-compute saving)

It also reports the attention-skip fraction (how many routed blocks dropped
their attention branch under hard routing) — the bridge between the
skip% claim and the measured latency gain.

Optional
--------
  --fixed-ckpt  a separately-trained fixed-prior / plain checkpoint, measured
                in soft mode as the "fixed-subnet / full-attn baseline" row.
  --sliding-window  run full-volume SlidingWindowInferer instead of a single
                    patch forward, to get clinical ms/volume (batch=1).

Design notes (the "harness" part = fixed, reproducible protocol)
---------------------------------------------------------------
  * fixed input size, batch=1
  * CUDA synchronize around every timed iteration
  * warmup (untimed) then median / mean / p95 over --repeats runs
  * peak VRAM reset between modes
  * fp32 by default; --amp for mixed-precision comparison

Dice / HD95 are deliberately NOT measured here — they come from the
auto-logged validation_end (soft/hard mean_dice) in the probe run. This
harness measures ONLY what only it can: wall-clock + VRAM + attn-skip.

Usage
-----
  # patch-level delta (cleanest isolation of conditional-compute saving)
  python measure_latency.py \
      --config configs/exp3_segattn_haar_routed.yaml \
      --ckpt    logs/segattn_haar_so8_dicece_ds/segattn_haar_routed/model/final_model_0.8967.pt \
      --device  cuda:0 --warmup 10 --repeats 30

  # clinical full-volume latency (BraTS 240x240x155, roi 128^3, overlap 0.5)
  python measure_latency.py --config ... --ckpt ... \
      --sliding-window --roi-size 128 128 128 --full-shape 240 240 155

  # also compare against a fixed-prior baseline checkpoint
  python measure_latency.py --config ... --ckpt ... \
      --fixed-ckpt logs/.../segattn_haar_so8_dicece_ds/segattn_haar/model/final_model_*.pt
"""
import argparse
import csv
import importlib.util
import os
import sys
import time

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────
# model construction (reuse the exact training builder)
# ─────────────────────────────────────────────────────────────────────────
def build_model(cfg):
    """Build SegAttnHaar with the EXACT hyperparams used in training.

    Loads _build_model from the training script via importlib (the script's
    `if __name__ == "__main__"` guard prevents training from running on import).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "train_module", os.path.join(here, "3_train_brats2021.py"))
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    return train_module._build_model(cfg)


def load_weights(model, ckpt_path, verbose=True):
    """Load weights handling both formats:
      * final_model_*.pt / best_model_*.pt / tmp_model_*.pt : state_dict
      * resume_latest.pt                              : dict with 'model' key
    Strips a possible DDP 'module.' prefix.
    """
    obj = torch.load(ckpt_path, map_location="cpu")
    sd = obj["model"] if (isinstance(obj, dict) and "model" in obj) else obj
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=True)
    if verbose:
        print(f"[weights] loaded {os.path.basename(ckpt_path)}: {len(sd)} tensors")
        if missing:
            print(f"[weights]   missing   ({len(missing)}): {missing[:5]} ...")
        if unexpected:
            print(f"[weights]   unexpected ({len(unexpected)}): {unexpected[:5]} ...")
    return model


# ─────────────────────────────────────────────────────────────────────────
# timing core
# ─────────────────────────────────────────────────────────────────────────
def _count_skip(stats):
    if not stats:
        return None
    n = len(stats)
    used = sum(s.get("use_attn", 0) for s in stats.values())
    return (1.0 - used / n) if n else None


def _forward(model, x, mode, amp, device):
    """Run one forward; return (logits, stats-or-None)."""
    if mode == "hard":
        if amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out, _, stats = model(x, return_gate_stats=True)
        else:
            out, _, stats = model(x, return_gate_stats=True)
        return out, stats
    else:
        if amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(x)
        else:
            out = model(x)
        return out, None


def time_mode(model, x, mode, device, warmup, repeats, amp, threshold):
    """Single-patch timing. Returns a result dict."""
    model.eval()
    if mode == "hard":
        model.set_hard_routing(True, threshold)
    else:
        model.set_hard_routing(False)

    # ── warmup (untimed) ──
    with torch.no_grad():
        for _ in range(warmup):
            _forward(model, x, mode, amp, device)
            torch.cuda.synchronize(device)

    # ── timed ──
    torch.cuda.reset_peak_memory_stats(device)
    times, stats = [], None
    with torch.no_grad():
        for _ in range(repeats):
            t0 = time.perf_counter()
            out, stats = _forward(model, x, mode, amp, device)
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    model.set_hard_routing(False)  # restore soft semantics
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    times.sort()
    n = len(times)
    return {
        "mode": mode,
        "median_ms": times[n // 2],
        "mean_ms": sum(times) / n,
        "p95_ms": times[min(int(0.95 * n), n - 1)],
        "min_ms": times[0],
        "peak_vram_gb": peak_vram,
        "attn_skip": _count_skip(stats),
        "out_shape": tuple(out.shape),
    }


def time_mode_sw(model, x, mode, device, warmup, repeats, amp, threshold,
                 roi_size, overlap):
    """Full-volume SlidingWindowInferer timing (clinical ms/volume)."""
    from monai.inferers import SlidingWindowInferer

    model.eval()
    if mode == "hard":
        model.set_hard_routing(True, threshold)
    else:
        model.set_hard_routing(False)

    inferer = SlidingWindowInferer(
        roi_size=list(roi_size), sw_batch_size=1, overlap=overlap)

    def predictor(inp):
        # distinct patch per call → must not reuse a cached block decision
        # from a previous window (would freeze routing to the first patch).
        if mode == "hard":
            model.clear_hard_cache()
            return model(inp, return_gate_stats=True)[0]
        return model(inp)

    with torch.no_grad():
        for _ in range(warmup):
            inferer(x, predictor)
            torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    times = []
    with torch.no_grad():
        for _ in range(repeats):
            t0 = time.perf_counter()
            out = inferer(x, predictor)
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    model.set_hard_routing(False)
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    times.sort()
    n = len(times)
    return {
        "mode": mode,
        "median_ms": times[n // 2],
        "mean_ms": sum(times) / n,
        "p95_ms": times[min(int(0.95 * n), n - 1)],
        "min_ms": times[0],
        "peak_vram_gb": peak_vram,
        "attn_skip": None,  # per-window decision; read from patch-level run
        "out_shape": tuple(out.shape),
    }


# ─────────────────────────────────────────────────────────────────────────
# reporting
# ─────────────────────────────────────────────────────────────────────────
def print_table(rows):
    hdr = ["mode", "median_ms", "mean_ms", "p95_ms", "peak_VRAM_GB", "attn_skip%"]
    print("\n" + "=" * 78)
    print(" CONDITIONAL-COMPUTE LATENCY HARNESS — RESULTS")
    print("=" * 78)
    print(f"  {'mode':<14}{'median':>10}{'mean':>10}{'p95':>10}"
          f"{'VRAM_GB':>12}{'skip%':>10}")
    print("  " + "-" * 66)
    for r in rows:
        skip = "" if r["attn_skip"] is None else f"{r['attn_skip']*100:7.1f}"
        print(f"  {r['mode']:<14}{r['median_ms']:>10.2f}{r['mean_ms']:>10.2f}"
              f"{r['p95_ms']:>10.2f}{r['peak_vram_gb']:>12.2f}{skip:>10}")
    # conditional-compute speedup (hard vs soft)
    soft = next((r for r in rows if r["mode"] == "soft"), None)
    hard = next((r for r in rows if r["mode"] == "hard"), None)
    if soft and hard and soft["median_ms"] > 0:
        sp = (1 - hard["median_ms"] / soft["median_ms"]) * 100
        print("  " + "-" * 66)
        print(f"  hard vs soft median: {sp:+.1f}% latency "
              f"({'faster' if sp > 0 else 'slower'})")
    print("=" * 78 + "\n")


def write_csv(rows, path):
    keys = ["mode", "median_ms", "mean_ms", "p95_ms", "min_ms",
            "peak_vram_gb", "attn_skip", "out_shape"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            row = dict(r)
            row["attn_skip"] = "" if r["attn_skip"] is None else f"{r['attn_skip']:.4f}"
            row["out_shape"] = str(r["out_shape"])
            w.writerow(row)
    print(f"[csv] wrote {path}")


# ─────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Conditional-compute latency harness")
    ap.add_argument("--config", default=None,
                    help="YAML config used for training (exact hyperparams). "
                         "If omitted, routed segattn_haar defaults are used.")
    ap.add_argument("--ckpt", required=True,
                    help="Primary checkpoint (routed SegAttn-Haar). "
                         "final_model_*.pt or resume_latest.pt.")
    ap.add_argument("--fixed-ckpt", default=None,
                    help="Optional baseline checkpoint (plain/fixed-prior), "
                         "measured in soft mode as 'fixed-subnet' row.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--input-shape", type=int, nargs=5, default=None,
                    metavar=("B", "C", "D", "H", "W"),
                    help="Single-patch input shape (default: from roi_size in "
                         "config, or 1 4 128 128 128).")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="block-level gate threshold for hard routing")
    ap.add_argument("--amp", action="store_true", help="use fp16 autocast")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=None, help="optional CSV output path")
    # sliding-window (clinical) mode
    ap.add_argument("--sliding-window", action="store_true",
                    help="time full-volume SlidingWindowInferer (ms/volume)")
    ap.add_argument("--roi-size", type=int, nargs=3, default=[128, 128, 128])
    ap.add_argument("--full-shape", type=int, nargs=3, default=[240, 240, 155])
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--grouped-mix", action="store_true",
                    help="use grouped (groups=8) HaarDownsample.mix "
                         "(load-bearing so8 Q); default is legacy groups=1 "
                         "which matches the routed checkpoints trained before "
                         "the lowo patch.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # NOTE: if CUDA_VISIBLE_DEVICES is set, CUDA re-indexes the visible GPUs
        # to 0..k-1. Passing the *physical* id (e.g. `--device cuda:2` while
        # `CUDA_VISIBLE_DEVICES=2`) raises "invalid device ordinal" because only
        # ordinal 0 exists in that masked view. Either pass the re-indexed
        # ordinal (cuda:0) or drop CUDA_VISIBLE_DEVICES entirely.
        try:
            torch.cuda.set_device(device)
        except RuntimeError as e:
            if "invalid device ordinal" in str(e):
                vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                raise RuntimeError(
                    f"invalid device ordinal for {device}. CUDA_VISIBLE_DEVICES="
                    f"'{vis}' masks the physical id — pass the re-indexed "
                    f"ordinal (e.g. cuda:0), or run without CUDA_VISIBLE_DEVICES."
                ) from e
            raise
        torch.cuda.init()
    print(f"[device] {device}  amp={args.amp}  sliding_window={args.sliding_window}")

    # ── config + model ──
    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        # merge CLI override (CLI flag beats YAML, but YAML is canonical)
        if getattr(args, "grouped_mix", False) and not cfg.get("grouped_mix"):
            cfg["grouped_mix"] = True
        print(f"[cfg] {args.config}  model={cfg.get('model')}  "
              f"use_routing={cfg.get('use_routing')}  "
              f"grouped_mix={cfg.get('grouped_mix','not set→False')}")
    else:
        cfg = {"model": "segattn_haar", "use_routing": True,
               "wavelet_mode": "so8", "learnable_wavelet": True,
               "num_heads": None, "window_size": [7, 7, 7],
               "attn_checkpoint": False, "rf_prior_scale": 1.5,
               "router_hidden": None,
               "grouped_mix": args.grouped_mix}

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {type(model).__name__}  params={n_params/1e6:.2f}M  "
          f"routing={'on' if getattr(model.vit, 'use_routing', False) else 'off'}")

    load_weights(model, args.ckpt)

    # ── input tensor ──
    if args.sliding_window:
        shape = [1, 4] + list(args.full_shape)
    else:
        if args.input_shape:
            shape = list(args.input_shape)
        elif cfg.get("roi_size"):
            shape = [1, 4] + list(cfg["roi_size"])
        else:
            shape = [1, 4, 128, 128, 128]
    x = torch.randn(*shape, device=device)
    print(f"[input] shape={tuple(shape)}  ({'sliding-window' if args.sliding_window else 'single patch'})")

    rows = []
    # primary checkpoint: soft + hard
    if args.sliding_window:
        rows.append(time_mode_sw(model, x, "soft", device, args.warmup,
                                 args.repeats, args.amp, args.threshold,
                                 args.roi_size, args.overlap))
        rows.append(time_mode_sw(model, x, "hard", device, args.warmup,
                                 args.repeats, args.amp, args.threshold,
                                 args.roi_size, args.overlap))
    else:
        rows.append(time_mode(model, x, "soft", device, args.warmup,
                              args.repeats, args.amp, args.threshold))
        rows.append(time_mode(model, x, "hard", device, args.warmup,
                              args.repeats, args.amp, args.threshold))

    # optional fixed-subnet / full-attn baseline (soft only)
    if args.fixed_ckpt:
        fixed_cfg = dict(cfg)
        fixed_cfg["use_routing"] = False  # a plain checkpoint has no gates
        fixed_model = build_model(fixed_cfg).to(device)
        load_weights(fixed_model, args.fixed_ckpt)
        fx = torch.randn(*shape, device=device)
        if args.sliding_window:
            rows.append(time_mode_sw(fixed_model, fx, "soft", device, args.warmup,
                                     args.repeats, args.amp, args.threshold,
                                     args.roi_size, args.overlap))
        else:
            rows.append(time_mode(fixed_model, fx, "soft", device, args.warmup,
                                  args.repeats, args.amp, args.threshold))
        rows[-1]["mode"] = "fixed-subnet"

    print_table(rows)
    if args.csv:
        write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
