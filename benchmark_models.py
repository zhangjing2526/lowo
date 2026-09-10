"""
Unified efficiency benchmark for the three compared models:
  SegMamba, SegMamba-Haar, SwinUNETR

Reports, per model (single 128^3 patch, batch=1 unless noted):
  - Params (M)
  - FLOPs (G)          via torch.utils.flop_counter (conv/matmul; custom CUDA
                        ops like the Mamba selective scan are NOT counted --
                        state this caveat in the paper)
  - Inference latency  median of 30 timed runs after 10 warmup, batch=1
  - Train step time    forward+backward+SGD step, batch=2 (training config)
  - Peak GPU memory    for both settings

Usage:
  python benchmark_models.py [--device cuda:1]
"""

import argparse
import statistics
import time

import torch
import torch.nn as nn

ROI = (128, 128, 128)
IN_CH, OUT_CH = 4, 4


def build_models():
    from model_segmamba.segmamba import SegMamba
    from model_segmamba.segmamba_haar import SegMambaHaar
    from monai.networks.nets import SwinUNETR

    def segmamba():
        return SegMamba(in_chans=IN_CH, out_chans=OUT_CH,
                        depths=[2, 2, 2, 2], feat_size=[48, 96, 192, 384])

    def segmamba_haar():
        return SegMambaHaar(in_chans=IN_CH, out_chans=OUT_CH,
                            depths=[2, 2, 2, 2], feat_size=[48, 96, 192, 384])

    def swin_unetr():
        try:
            return SwinUNETR(img_size=ROI, in_channels=IN_CH,
                             out_channels=OUT_CH, feature_size=48)
        except TypeError:  # newer monai dropped img_size
            return SwinUNETR(in_channels=IN_CH, out_channels=OUT_CH,
                             feature_size=48)

    return {"SegMamba": segmamba, "SegMamba-Haar": segmamba_haar,
            "SwinUNETR": swin_unetr}


def count_flops(model, device):
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None
    x = torch.randn(1, IN_CH, *ROI, device=device)
    model.eval()
    try:
        with torch.no_grad():
            flop_counter = FlopCounterMode(display=False)
            with flop_counter:
                model(x)
        return flop_counter.get_total_flops()
    except Exception as e:
        print(f"    [flops failed: {e}]")
        return None


def bench_inference(model, device, warmup=10, runs=30):
    model.eval()
    x = torch.randn(1, IN_CH, *ROI, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize(device)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            torch.cuda.synchronize(device)
            times.append((time.perf_counter() - t0) * 1000)
    mem = torch.cuda.max_memory_allocated(device) / 2**30
    return statistics.median(times), mem


def bench_train_step(model, device, warmup=5, runs=15):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=3e-5,
                          momentum=0.99, nesterov=True)
    ce = nn.CrossEntropyLoss()
    x = torch.randn(2, IN_CH, *ROI, device=device)
    y = torch.randint(0, OUT_CH, (2, *ROI), device=device)
    torch.cuda.reset_peak_memory_stats(device)

    def step():
        opt.zero_grad(set_to_none=True)
        loss = ce(model(x), y)
        loss.backward()
        opt.step()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize(device)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize(device)
        times.append((time.perf_counter() - t0) * 1000)
    mem = torch.cuda.max_memory_allocated(device) / 2**30
    return statistics.median(times), mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--skip-train", action="store_true",
                    help="skip train-step benchmark (saves memory)")
    args = ap.parse_args()
    device = args.device
    torch.backends.cudnn.benchmark = True

    rows = []
    for name, ctor in build_models().items():
        print(f"== {name} ==")
        model = ctor().to(device)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"    params: {params:.2f} M")

        flops = count_flops(model, device)
        gflops = flops / 1e9 if flops else float("nan")
        print(f"    flops : {gflops:.1f} G")

        lat, infer_mem = bench_inference(model, device)
        print(f"    infer : {lat:.1f} ms/patch   peak mem {infer_mem:.2f} GiB")

        if args.skip_train:
            tstep, train_mem = float("nan"), float("nan")
        else:
            tstep, train_mem = bench_train_step(model, device)
            print(f"    train : {tstep:.1f} ms/step (bs=2)   peak mem {train_mem:.2f} GiB")

        rows.append((name, params, gflops, lat, infer_mem, tstep, train_mem))
        del model
        torch.cuda.empty_cache()

    print("\n| Model | Params (M) | FLOPs (G) | Infer (ms) | Infer mem (GiB) | Train step (ms) | Train mem (GiB) |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print("| {} | {:.2f} | {:.1f} | {:.1f} | {:.2f} | {:.1f} | {:.2f} |".format(*r))

    print("\nNote: FLOPs counted by torch.utils.flop_counter (conv/matmul);")
    print("custom CUDA kernels (Mamba selective scan) are not included.")


if __name__ == "__main__":
    main()
