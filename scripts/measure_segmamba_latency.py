#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
measure_segmamba_latency.py — 针对 SegMamba-Haar 各臂 + SegMamba 基线 + SwinUNETR
的墙钟延迟 / 峰值显存 / FLOPs 测量 harness。

为什么和 measure_latency.py 分开:
  measure_latency.py 是为 SegAttn-Haar (routed) 写的, 依赖 set_hard_routing()。
  SegMamba-Haar 是「可学习小波下采样替换」, 没有 routing 门, 逻辑不同,
  但两类模型共享同一套「测量协议」: 固定输入、batch=1、同步、预热、多次取中位数。

本脚本只做测试-时间测量, 不做 Dice/HD95 (那些来自训练日志的 auto-logged 值)。

    python measure_segmamba_latency.py --arm fixed_g8
    python measure_segmamba_latency.py --arm b0 --device cuda:0 --repeats 50
    python measure_segmamba_latency.py --arm swin --sliding-window
"""

import argparse
import csv
import os
import time

import torch


# ──────────────────────────────────────────────────────────────────────────
# 模型构建 (严格对齐训练时的构造参数)
# ──────────────────────────────────────────────────────────────────────────
def build_segmamba_model(out_chans=4):
    """SegMamba 基线 B0 (learned strided conv)。对齐 configs/exp1_segmamba.yaml。"""
    from model_segmamba.segmamba import SegMamba
    return SegMamba(
        in_chans=4, out_chans=out_chans,
        depths=[2, 2, 2, 2], feat_size=[48, 96, 192, 384],
    )


def build_segmamba_haar_model(cfg: dict, out_chans=4):
    """SegMamba-Haar (fixed / so8 / theta / per_channel / wb 等)。"""
    from model_segmamba.segmamba_haar import SegMambaHaar
    return SegMambaHaar(
        in_chans=4, out_chans=out_chans,
        depths=[2, 2, 2, 2], feat_size=[48, 96, 192, 384],
        learnable_wavelet=cfg.get("learnable_wavelet", True),
        gsc_checkpoint=cfg.get("gsc_checkpoint", False),
        wavelet_mode=cfg.get("wavelet_mode", "theta"),
        per_channel_stem=cfg.get("per_channel_stem", False),
        per_channel_wavelet=cfg.get("per_channel_wavelet", False),
        grouped_mix=cfg.get("grouped_mix", False),
    )


def build_segattn_haar_model(cfg: dict, out_chans=4):
    """SegAttn-Haar (segattn / routed 臂)。

    构建 ``SegAttnHaar``（swin-style window attention + Haar wavelet，
    与 SegMambaHaar 共享 wavelet/GSC/decoder 结构，仅序列混合器替换为注意力）。
    参数来自 cfg dict，对齐训练时的构造签名。
    """
    from model_segmamba.segmamba_attn import SegAttnHaar
    return SegAttnHaar(
        in_chans=4, out_chans=out_chans,
        depths=[2, 2, 2, 2], feat_size=[48, 96, 192, 384],
        learnable_wavelet=cfg.get("learnable_wavelet", True),
        gsc_checkpoint=cfg.get("gsc_checkpoint", False),
        wavelet_mode=cfg.get("wavelet_mode", "theta"),
        per_channel_stem=cfg.get("per_channel_stem", False),
        per_channel_wavelet=cfg.get("per_channel_wavelet", False),
        num_heads=cfg.get("num_heads", None),
        window_size=cfg.get("window_size", (7, 7, 7)),
        attn_checkpoint=False,
        use_routing=cfg.get("use_routing", False),
        rf_prior_scale=cfg.get("rf_prior_scale", 1.5),
        router_hidden=cfg.get("router_hidden", None),
        use_shift=True,
        grouped_mix=cfg.get("grouped_mix", False),
    )


def build_swin_unetr_model():
    """SwinUNETR 外部基线。对齐 3_train_swin_unetr_brats2021.py:110-124。"""
    from monai.networks.nets import SwinUNETR
    return SwinUNETR(
        img_size=(128, 128, 128),
        in_channels=4,
        out_channels=4,
        feature_size=48,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        norm_name="instance",
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        normalize=True,
        use_checkpoint=False,
        spatial_dims=3,
    )


# ── 各臂配置 / checkpoint 映射 (10 臂主表) ────────────────────────────────
ARM_SPECS = {
    # ── SegMamba-Haar (LOWO fzg groups=1) ──
    "fixed": {
        "desc": "A1 fixed Haar, groups=1 mixer",
        "cfg": {"wavelet_mode": "so8", "learnable_wavelet": False, "grouped_mix": False},
        "ckpt": "logs/segmamba_haar_so8_dicece_ds/segmamba_haar_fixed/model/best_model_0.9268.pt",
    },
    "wb": {
        "desc": "A2 wb 边界 loss, groups=1 mixer",
        "cfg": {"wavelet_mode": "so8", "learnable_wavelet": True, "grouped_mix": False},
        "ckpt": "logs/segmamba_haar_so8_dicece_ds/segmamba_haar_wb/model/best_model_0.9181.pt",
    },
    "theta": {
        "desc": "A5 theta, groups=1 mixer (3-DOF separable)",
        "cfg": {"wavelet_mode": "theta", "learnable_wavelet": True, "grouped_mix": False},
        "ckpt": "logs/segmamba_haar_dicece_ds/segmamba_haar_theta/model/best_model_0.9251.pt",
    },
    "so8": {
        "desc": "A3 so8, groups=1 mixer (28-DOF nonseparable)",
        "cfg": {"wavelet_mode": "so8", "learnable_wavelet": True, "grouped_mix": False},
        "ckpt": "logs/segmamba_haar_so8_dicece_ds/segmamba_haar/model/best_model_0.9225.pt",
    },
    # ── fixed groups=8 (A4) & learnable groups=8 (A6) 主对比 ──
    "fixed_g8": {
        "desc": "A4 fixed Haar, groups=8 mixer (主对比)",
        "cfg": {"wavelet_mode": "so8", "learnable_wavelet": False, "grouped_mix": True},
        "ckpt": "logs/segmamba_haar_so8_dicece_ds/segmamba_haar_fixed_g8/model/best_model_0.9285.pt",
    },
    "so8_live": {
        "desc": "A6 so8 live, groups=8 mixer (可学性最佳对消臂)",
        "cfg": {"wavelet_mode": "so8", "learnable_wavelet": True, "grouped_mix": True},
        "ckpt": "logs/segmamba_haar_so8_dicece_ds/segmamba_haar_so8live/model/best_model_0.9317.pt",
    },
    "perch": {
        "desc": "A8 per-channel stem + per-channel wavelet",
        "cfg": {"wavelet_mode": "so8", "learnable_wavelet": True,
                "per_channel_stem": True, "per_channel_wavelet": True,
                "grouped_mix": False},
        "ckpt": "logs/segmamba_haar_so8_dicece_ds/segmamba_haar_perch/model/best_model_0.9098.pt",
    },
    # ── 注意办基线 (SegAttn-Haar) ──
    "segattn": {
        "desc": "C1 SegAttn-Haar, groups=1 mixer",
        "cfg": {"wavelet_mode": "so8", "learnable_wavelet": False, "grouped_mix": False,
                "num_heads": None, "window_size": (7, 7, 7)},
        "ckpt": "logs/segattn_haar_so8_dicece_ds/segattn_haar/model/best_model_0.9190.pt",
    },
    "routed": {
        "desc": "C2 SegAttn-Haar + routing",
        "cfg": {"wavelet_mode": "so8", "learnable_wavelet": False, "grouped_mix": False,
                "num_heads": None, "window_size": (7, 7, 7), "use_routing": True},
        "ckpt": "logs/segattn_haar_so8_dicece_ds/segattn_haar_routed/model/best_model_0.9152.pt",
    },
    # ── 外部基线 ──
    "b0": {
        "desc": "B0 SegMamba learned strided conv 基线",
        "cfg": None,
        "ckpt": "logs/segmamba_dicece_ds/segmamba/model/best_model_0.9212.pt",
    },
    "swin": {
        "desc": "S1 SwinUNETR 外部基线",
        "cfg": None,
        "ckpt": "logs/swin_unetr_brats2021/model/best_model_0.9277.pt",
    },
}


def resolve_ckpt(spec: dict, logs_dir: str = None) -> str:
    """把 manifest 相对路径解析到实际存在的 checkpoint。

    日志/权重根目录优先级: 显式 --logs 参数 > $SEGMAMBA_LOGS > 相对当前
    repo 的 ``./logs``。这样同一份 manifest 在 Windows(edit-only) 与
    Linux(training/eval) 上都无需改写路径即可定位。

    兼容性设计: manifest 里的 ckpt 路径自带 ``logs/`` 前缀, 而 base 可能
    指向仓库根或 ``logs/`` 子目录; 这里对重复的 ``logs/logs`` 段自动去重,
    base 指向哪种都不影响结果。
    """
    base = logs_dir or os.environ.get("SEGMAMBA_LOGS") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "logs")
    rel = spec["ckpt"]

    # 去重 ``logs/logs`` 重复段: 只允许 base 尾再带 *一个* logs
    cand = os.path.normpath(os.path.join(base, rel))
    # 若 base 本身是 `<root>/logs` (默认), 而 rel 又带 logs/ 前缀, normpath 后
    # 会变成 `<root>/logs/logs/...`。把中间的重复 `logs/logs` 折叠成 `logs`。
    if os.sep + "logs" + os.sep + "logs" + os.sep in cand:
        cand = cand.replace(os.sep + "logs" + os.sep + "logs" + os.sep,
                            os.sep + "logs" + os.sep)

    if os.path.isfile(cand):
        return cand

    # manifest 里写的是「某一次训练」的最好值; 若不存在, 在父目录搜索同 basename
    parent = os.path.dirname(cand)
    name = os.path.basename(rel)
    if os.path.isdir(parent):
        matches = sorted(
            os.path.join(parent, f)
            for f in os.listdir(parent)
            if f.startswith(name.split("_0.")[0] if "_0." in name else name.split(".")[0])
            and f.endswith(".pt")
        )
        if matches:
            # 取数值最大者 (best highest 应含最高 val)
            return sorted(matches, key=lambda p: float(
                p.split("_")[-1].removesuffix(".pt")))[-1]
    raise FileNotFoundError(
        f"checkpoint not found for {spec['desc']}: tried {cand}\n"
        f"  pass --logs <dir> or set SEGMAMBA_LOGS to the training host's "
        f"logs/ root."
    )


def load_weights(model, ckpt_path, verbose=True):
    """加载权重, 兼容 state_dict 与 {'model': {...}} 两种格式, 剥离 DDP 前缀。"""
    obj = torch.load(ckpt_path, map_location="cpu")
    sd = obj["model"] if (isinstance(obj, dict) and "model" in obj) else obj
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    # strict=False：兼容 checkpoint 版本差异（旧版无 _k_cache 等缓存 buffer）。
    # checkpoint 加载后用 torch.compile 或手动预热即可重建缓存，不影响测量结果。
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if verbose:
        print(f"[weights] {os.path.basename(ckpt_path)}: {len(sd)} tensors loaded")
        if missing:
            print(f"[weights]   missing ({len(missing)}): {missing[:5]}")
        if unexpected:
            print(f"[weights]   unexpected ({len(unexpected)}): {unexpected[:5]}")
    return model


# ──────────────────────────────────────────────────────────────────────────
# 测量核
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def measure_forward(model, x, warmup, repeats, device):
    """单次 forward 计时, 返回 median ms / mean ms / p95 ms / 峰值显存 GB。"""
    model.eval()

    def run():
        return model(x)

    # 预热 (不计时)
    for _ in range(warmup):
        run()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    # 计时
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        run()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append((time.perf_counter() - t0) * 1000.0)

    times.sort()
    n = len(times)
    peak = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        if device.type == "cuda" else float("nan")
    )
    return {
        "median_ms": times[n // 2],
        "mean_ms": sum(times) / n,
        "p95_ms": times[min(int(0.95 * n), n - 1)],
        "min_ms": times[0],
        "max_ms": times[-1],
        "peak_vram_gb": peak,
    }


@torch.no_grad()
def measure_sliding_window(model, x, roi_size, overlap, warmup, repeats, device):
    """全卷 sliding-window 推理计时 (临床 ms/volume)。"""
    from monai.inferers import SlidingWindowInferer

    model.eval()
    inferer = SlidingWindowInferer(
        roi_size=list(roi_size), sw_batch_size=1, overlap=overlap)

    def run():
        return inferer(x, model)

    for _ in range(max(1, warmup // 2)):
        run()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        run()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    n = len(times)
    return {
        "median_ms": times[n // 2],
        "mean_ms": sum(times) / n,
        "p95_ms": times[min(int(0.95 * n), n - 1)],
        "min_ms": times[0],
        "max_ms": times[-1],
        "peak_vram_gb": (
            torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            if device.type == "cuda" else float("nan")),
    }


# ──────────────────────────────────────────────────────────────────────────
# FLOPs 估算 (GPU 硬件路线可选; 无法取到举手)
# ──────────────────────────────────────────────────────────────────────────
def estimate_flops(model, shape):
    from fvcore.nn import FlopCountAnalysis
    x = torch.randn(1, *shape, device=next(model.parameters()).device)
    return FlopCountAnalysis(model, x).total()


# ──────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="SegMamba-Haar latency harness")
    ap.add_argument("--arm", required=True,
                    choices=list(ARM_SPECS.keys()),
                    help="arm to measure")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--input-shape", type=int, nargs=4, default=[4, 128, 128, 128],
                    metavar=("C", "D", "H", "W"))
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--sliding-window", action="store_true",
                    help="full-volume SlidingWindowInferer (除去 patch 调度开销)")
    ap.add_argument("--full-shape", type=int, nargs=3, default=[240, 240, 155])
    ap.add_argument("--roi-size", type=int, nargs=3, default=[128, 128, 128])
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--csv", default=None,
                    help="CSV output path. Default: auto-write to "
                         "latency_logs/<arm>_<timestamp>.csv so every run "
                         "is archived (pass a path to override).")
    ap.add_argument("--flops", action="store_true", help="try to report FLOPs")
    ap.add_argument("--logs", default=None,
                    help="checkpoint base dir (defaults: $SEGMAMBA_LOGS, "
                         "else ./logs relative to repo root). "
                         "On the training host this is typically "
                         "/data1/zhangjing/seg/SegMamba/logs.")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            torch.cuda.set_device(device)
        except RuntimeError as e:
            vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            raise RuntimeError(
                f"invalid device {device} (CUDA_VISIBLE_DEVICES='{vis}' "
                f"masks the ordinal) — pass the re-indexed id (e.g. cuda:0)"
            ) from e
        torch.cuda.init()
        print(f"[device] {device} amp={args.amp} sliding={args.sliding_window}")
    else:
        print(f"[device] cpu (fallback)")

    spec = ARM_SPECS[args.arm]
    print(f"[arm] {args.arm}: {spec['desc']}")

    # 构建 + 加载
    if args.arm == "swin":
        model = build_swin_unetr_model()
    elif args.arm == "b0":
        model = build_segmamba_model()
    elif args.arm in ("segattn", "routed"):
        model = build_segattn_haar_model(spec["cfg"])
    else:
        model = build_segmamba_haar_model(spec["cfg"])

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {type(model).__name__}  params={n_params/1e6:.2f}M")

    ckpt = resolve_ckpt(spec, args.logs)
    load_weights(model, ckpt)

    # 输入
    if args.sliding_window:
        shape = (1, args.input_shape[0]) + tuple(args.full_shape)
    else:
        shape = (1,) + tuple(args.input_shape)
    x = torch.randn(*shape, device=device)
    print(f"[input] shape={tuple(shape)}")

    rows = []
    if args.sliding_window:
        r = measure_sliding_window(model, x, args.roi_size, args.overlap,
                                   args.warmup, args.repeats, device)
    else:
        r = measure_forward(model, x, args.warmup, args.repeats, device)

    r["arm"] = args.arm
    r["params_M"] = n_params / 1e6
    r["input"] = "x".join(str(s) for s in shape[1:])
    rows.append(r)

    if args.flops:
        try:
            if not args.sliding_window:
                fl = estimate_flops(model, tuple(shape[1:]))
                print(f"[flops] {fl/1e9:.3f} GFLOPs")
                r["gflops"] = fl / 1e9
        except Exception as e:
            print(f"[flops] skipped: {e}")

    # 打印
    for r in rows:
        print(f"\n  median  {r['median_ms']:8.2f} ms")
        print(f"  mean    {r['mean_ms']:8.2f} ms")
        print(f"  p95     {r['p95_ms']:8.2f} ms")
        print(f"  min     {r['min_ms']:8.2f} ms")
        print(f"  max     {r['max_ms']:8.2f} ms")
        print(f"  peakVRAM {r['peak_vram_gb']:6.3f} GB")

    # 默认自动落盘: latency_logs/<arm>_<timestamp>.csv, 显式 --csv 可覆盖
    path = args.csv or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "latency_logs",
        f"{args.arm}_{time.strftime('%Y%m%d_%H%M%S')}.csv",
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[csv] wrote {path}")


if __name__ == "__main__":
    main()