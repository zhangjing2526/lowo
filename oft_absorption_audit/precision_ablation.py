"""
精度消融审计 (Precision Ablation)
=================================
回答审稿建议 3 的最后一组关键检查:
  "关闭混合精度和 TF32, 比较 FP32 与从合并前原始权重重新计算的 FP64 结果。
   若 FP64 下误差显著下降, 才更有依据把差异归因于有限精度计算。"

本脚本测量同一条量: 吸收误差 = |未合并前向 - 合并前向|
在三种配置下分别测量, 看它是否随精度下降:

  (A) FP32 + TF32 开   err_tf32_on   —— 复现原始 audit 的执行配置
  (B) FP32 + TF32 关   err_fp32      —— 干净 FP32 (关 TF32 / cudnn benchmark)
  (C) FP64 (从原权重重算合并)  err_fp64
        - 未合并 peft 模型整体 .double()  => 权重与 R 全部升双精度
        - 合并模型 = 从未合并副本整体 .double() 后**重新** peft.merge_and_unload()
          (在 FP64 下重新计算合并, 不继承 FP32 合并时的舍入)
        - 两者均以双精度前向, 吸收误差应趋近 double 机器精度 (~1e-15 * 尺度)

判断逻辑:
  - err_fp64 << err_fp32   => 差异主要来自有限精度 (FP32 舍入), 而非实现 bug
  - err_tf32_off < err_tf32_on => TF32 也贡献了一部分误差
  - 另附每注入层的局部吸收误差 FP32 vs FP64 (用固定输入重放, 复用方法学)
    => "局部吸收在 FP64 下趋近 0" 才是严格支持 "吸收代数等价" 的直接证据

用法:
  python precision_ablation.py
  python precision_ablation.py --checkpoint-dir outputs/oft_adapter_checkpoint
  python precision_ablation.py --output-dir outputs/oft_audit
  python precision_ablation.py --no-global-fp64   # 跳过整网 FP64 (省显存/时间)
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

_sdir = os.path.dirname(os.path.abspath(__file__))
if _sdir not in sys.path:
    sys.path.insert(0, _sdir)

from config import OFTAuditConfig
from provenance import capture_versions, build_oft_config

try:
    from audit import _canonical_name
except Exception:
    def _canonical_name(name: str) -> str:
        for p in ("base_model.model.", "base_model."):
            if name.startswith(p):
                return name[len(p):]
        return name


# ----------------------------------------------------------------------------
# 精度开关
# ----------------------------------------------------------------------------

def _set_precision(tf32: bool):
    """全局设置 TF32 / cudnn benchmark。

    allow_tf32 影响 matmul/cuda 卷积的 TensorFloat-32 路径, 是 FP32 下额外精度损失来源。
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# 模型加载
# ----------------------------------------------------------------------------

def _load_base(cfg):
    from transformers import AutoModelForImageClassification
    base = AutoModelForImageClassification.from_pretrained(
        cfg.base_model,
        num_labels=cfg.num_labels,
        ignore_mismatched_sizes=True,
        trust_remote_code=True,
    )
    return base


def _make_peft(base, cfg, ckpt):
    """在 base 上注入/加载 OFT, 返回 float32 peft 模型 (强制精确 Cayley)。"""
    from peft import OFTConfig, get_peft_model
    from provenance import load_peft_exact_cayley

    with_ckpt = bool(ckpt) and os.path.isfile(os.path.join(ckpt, "adapter_config.json"))
    if with_ckpt:
        model, _ = load_peft_exact_cayley(base, ckpt)
    else:
        model = get_peft_model(base, OFTConfig(
            r=cfg.oft_block_size,
            module_dropout=0.0,
            target_modules=cfg.target_modules,
            use_cayley_neumann=False,
        ))
    model.eval()
    return model


def _get_injected(model):
    """返回 {canonical_name: module} 对合并前持有 base_layer 的 OFT 包裹层。"""
    out = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "base_layer") and ".base_layer" not in name:
            out[_canonical_name(name)] = mod
    return out


def _collect_merged_linears(merged_model, keys):
    """从 merge_and_unload() 之后的模型取出合并后的 nn.Linear, 按 canonical name 索引。

    合并后 OFT 层被替换为普通 nn.Linear (base_layer 提升), 不再有 oft_r,
    因此不能用 _get_injected (它要求 hasattr base_layer); 这里按名字提取合并后的线性层。
    """
    out = {}
    for name, mod in merged_model.named_modules():
        if ".base_layer" in name:
            continue
        c = _canonical_name(name)
        if c in keys and isinstance(mod, torch.nn.Linear):
            out[c] = mod
    return out


def _capture_inputs(model, dummy, keys):
    """前向一次, 捕获 keys 中每层的输入 (CPU)。跳过 base_layer。"""
    inputs = {}
    hooks = []

    def make(name):
        def fn(_, inp, out):
            t = inp[0] if isinstance(inp, (tuple, list)) else inp
            inputs[name] = t.detach().to("cpu")
        return fn

    for name, mod in model.named_modules():
        if ".base_layer" in name:
            continue
        c = _canonical_name(name)
        if c in keys:
            hooks.append(mod.register_forward_hook(make(c)))

    model.eval()
    with torch.no_grad():
        model(pixel_values=dummy, return_dict=True)
    for h in hooks:
        h.remove()
    return inputs


# ----------------------------------------------------------------------------
# 全局吸收误差 (整网未合并 vs 合并) — 严格对照: 底座与输入由调用方固定共享
# ----------------------------------------------------------------------------

def _global_absorption(cfg, ckpt, base, dummy, tf32: bool, high_prec: bool, device):
    """整网未合并 vs 合并吸收误差, 三档共用同一底座与同一输入, 做严格变量对照。

    调用方负责固定传入 base 与 dummy (见 run_precision_ablation); 本函数内部
    deepcopy base, 避免 merge_and_unload() 原地修改污染后续档的测量。

    - fp32 档 (high_prec=False): 在 fp32 下执行, tf32 控制 TF32 是否开启。
    - fp64 档 (high_prec=True): 在 double 下执行, 且**合并运算本身也在 FP64 下进行**
      (先 .double() 再 merge_and_unload), 不再继承 FP32 合并舍入
      (若先 fp32 合并再 .double(), 残差仍含 fp32 合并舍入, 是 T11 修掉的旧 bug)。
    """
    _set_precision(tf32)
    import copy as _copy
    base_c = _copy.deepcopy(base).to(device)
    unmerged = _make_peft(base_c, cfg, ckpt).to(device)
    if high_prec:
        unmerged = unmerged.double()
        dummy_x = dummy.double().to(device)
    else:
        dummy_x = dummy.to(device)

    with torch.inference_mode():
        logits_u = unmerged(pixel_values=dummy_x, return_dict=True).logits

    merged = unmerged.merge_and_unload().to(device)
    merged.eval()
    with torch.inference_mode():
        logits_m = merged(pixel_values=dummy_x, return_dict=True).logits

    err = (logits_u - logits_m).abs()
    res = {
        "max": float(err.max()),
        "mse": float((err ** 2).mean()),
    }
    flat = err.flatten().cpu().numpy()
    res["q50"] = float(np.quantile(flat, 0.5))
    res["q99"] = float(np.quantile(flat, 0.99))
    res["frac_above_1e-6"] = float((flat > 1e-6).mean())
    del base_c, unmerged, merged, dummy_x, logits_u, logits_m, err
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


# ----------------------------------------------------------------------------
# 局部吸收误差 (固定输入, 每注入层 FP32 vs FP64)
# ----------------------------------------------------------------------------

def _local_precision(cfg, ckpt, device):
    """对每个注入层, 用同一固定输入 x 重放, 分别在 FP32 与 FP64 下测吸收误差。

    未合并模块 = peft OFT 包裹层; 合并模块 = merge_and_unload 后的 nn.Linear。
    局部吸收误差 = |f_unmerged(x) - f_merged(x)| (同一 x, 同一精度)。
    """
    base = _load_base(cfg).to(device)
    unmerged = _make_peft(base, cfg, ckpt).to(device)

    inject = _get_injected(unmerged)
    keys = set(inject.keys())
    if not keys:
        print("[ERROR] 未找到任何 OFT 注入层")
        sys.exit(1)
    print(f"  被注入层数: {len(keys)}")

    dummy = torch.randn(min(cfg.num_test_samples, 8), 3, 224, 224, device=device)
    inputs = _capture_inputs(unmerged, dummy, keys)   # fp32 固定输入 (CPU)

    # 先深拷贝未合并模型: merge_and_unload 会原地修改 unmerged (吸收 R 并卸载 adapter),
    # 之后 inject[c] 持有的模块引用会失效 (走已合并状态), 使局部吸收误差假归零。
    import copy as _copy
    unmerged_copy = _copy.deepcopy(unmerged)
    merged = unmerged.merge_and_unload().to(device)
    merged_modules = {}
    for name, mod in merged.named_modules():
        if ".base_layer" in name:
            continue
        c = _canonical_name(name)
        if c in keys:
            merged_modules[c] = mod

    # FP64 合并: 从未合并副本升双精度后**重新**在 FP64 下执行 merge_and_unload,
    # 这样合并运算本身也在 FP64 下进行, 不再继承 FP32 合并时的舍入
    # (若直接对已 fp32 合并的 mm 调 .double(), 只是把已舍入的权重升精度, 残差仍含 fp32 合并误差)。
    unmerged_copy64 = _copy.deepcopy(unmerged_copy).double().to(device)
    merged64 = unmerged_copy64.merge_and_unload().to(device)
    merged_modules64 = _collect_merged_linears(merged64, keys)

    rows = []
    for c in sorted(keys):
        if c not in inputs or c not in merged_modules or c not in merged_modules64:
            rows.append({"layer": c, "fp32_max": float("nan"), "fp64_max": float("nan"),
                         "ratio": float("nan"), "note": "missing"})
            continue
        x = inputs[c]
        um = _get_injected(unmerged_copy).get(c)
        mm = merged_modules[c]

        # FP32: 直接在 fp32 下比较 unmerged vs 已 fp32 合并
        um32 = um.float(); mm32 = mm.float()
        with torch.no_grad():
            yu = um32(x.to(device)).detach().cpu()
            ym = mm32(x.to(device)).detach().cpu()
        d32 = (yu - ym).abs().max().item()

        # FP64: 用从未合并副本升双精度后**重合并**的 FP64 模型, 与未合并副本的 FP64 前向对比。
        # 合并运算本身也在 FP64 下进行, 残差只剩双精度机器精度, 不被 fp32 合并舍入污染。
        um64 = um.double().to(device)
        mm64 = merged_modules64[c]
        with torch.no_grad():
            yu64 = um64(x.double().to(device)).detach().cpu()
            ym64 = mm64(x.double().to(device)).detach().cpu()
        d64 = (yu64 - ym64).abs().max().item()

        ratio = (d32 / d64) if (d64 and d64 == d64 and d64 > 0) else float("nan")
        rows.append({"layer": c, "fp32_max": d32, "fp64_max": d64, "ratio": ratio})

    del base, unmerged, merged, inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------

def run_precision_ablation(cfg: OFTAuditConfig, ckpt=None, global_fp64=True):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")
    _set_seed(cfg.seed)

    prov = capture_versions(ckpt) if ckpt else {}
    _, use_cayley_forced = build_oft_config(ckpt) if ckpt else (None, True)
    print(f"  强制 use_cayley_neumann=False (精确 Cayley): {use_cayley_forced}")

    # 严格对照: 三档 (TF32开 / TF32关 / FP64) 共用同一底座与同一输入。
    # 底座只加载一次, 各档内部 deepcopy, 避免 merge_and_unload 原地修改污染;
    # 输入用固定种子生成一次, 否则各档随机输入不同会让 TF32 倍率等结论失去对照意义。
    base_once = _load_base(cfg)
    torch.manual_seed(cfg.seed)
    dummy_once = torch.randn(min(cfg.num_test_samples, 32), 3, 224, 224)

    print("\n[1/3] FP32 + TF32 开 (复现原始 audit 配置)")
    r_tf32_on = _global_absorption(cfg, ckpt, base_once, dummy_once,
                                   tf32=True, high_prec=False, device=device)
    print(f"      max_err={r_tf32_on['max']:.3e}  MSE={r_tf32_on['mse']:.3e}  "
          f"q99={r_tf32_on['q99']:.3e}  >1e-6占比={r_tf32_on['frac_above_1e-6']:.4f}")

    print("\n[2/3] FP32 + TF32 关 (干净 FP32)")
    r_fp32 = _global_absorption(cfg, ckpt, base_once, dummy_once,
                                tf32=False, high_prec=False, device=device)
    print(f"      max_err={r_fp32['max']:.3e}  MSE={r_fp32['mse']:.3e}  "
          f"q99={r_fp32['q99']:.3e}  >1e-6占比={r_fp32['frac_above_1e-6']:.4f}")

    r_fp64 = None
    if global_fp64:
        print("\n[3/3] FP64 (未合并与合并均在 double 下执行, 合并运算本身也在 FP64)")
        r_fp64 = _global_absorption(cfg, ckpt, base_once, dummy_once,
                                    tf32=False, high_prec=True, device=device)
        print(f"      max_err={r_fp64['max']:.3e}  MSE={r_fp64['mse']:.3e}  "
              f"q99={r_fp64['q99']:.3e}  >1e-6占比={r_fp64['frac_above_1e-6']:.4f}")
    else:
        print("\n[3/3] FP64 整网测量已跳过 (--no-global-fp64)")

    print("\n[*] 每注入层局部吸收误差 FP32 vs FP64 (固定输入重放)")
    local_rows = _local_precision(cfg, ckpt, device)
    valid = [r for r in local_rows if r["fp32_max"] == r["fp32_max"]
             and r["fp64_max"] == r["fp64_max"]]
    valid.sort(key=lambda r: -r["fp32_max"])
    for r in valid[:10]:
        rs = f"{r['ratio']:.2e}" if r["ratio"] == r["ratio"] else "nan"
        print(f"      {r['layer']:<66} fp32={r['fp32_max']:.3e}  "
              f"fp64={r['fp64_max']:.3e}  ratio={rs}")
    if len(valid) > 10:
        print(f"      ... 共 {len(valid)} 层, 仅列前 10 (见 CSV)")

    # 判决
    verdict = "INCONCLUSIVE"
    notes = []
    if r_fp64 is not None:
        if r_fp64["max"] < r_fp32["max"] * 1e-2:
            verdict = "FINITE_PRECISION"
            notes.append("整网吸收误差在 FP64 下较干净 FP32 下降 >100x, "
                         "差异主要由 FP32 有限精度造成。")
        else:
            notes.append("FP64 相对干净 FP32 未显著下降, 差异可能含结构性/实现因素, "
                         "不能仅归因于有限精度。")
    if r_tf32_on["max"] > r_fp32["max"] * 1.05:
        notes.append(f"TF32 开比 TF32 关的误差高 "
                     f"{r_tf32_on['max']/max(r_fp32['max'],1e-30):.1f}x, "
                     f"TF32 也贡献了部分误差。")
    if valid:
        med_fp64 = float(np.median([r["fp64_max"] for r in valid]))
        notes.append(f"局部吸收误差 FP64 中位={med_fp64:.3e} "
                     f"(趋近 0 即支持吸收代数等价)。")

    print(f"\n  精度消融结论: {verdict}")
    for n in notes:
        print(f"    - {n}")

    res = {
        "tf32_on": r_tf32_on,
        "fp32_tf32_off": r_fp32,
        "fp64": r_fp64,
        "local_per_layer": local_rows,
        "verdict": verdict,
        "notes": notes,
        "global_fp64_run": bool(global_fp64),
        "use_cayley_neumann_forced": use_cayley_forced,
    }
    res.update(prov)
    return res


# ----------------------------------------------------------------------------
# 输出
# ----------------------------------------------------------------------------

def _write_csv(local_rows, path):
    fields = ["layer", "fp32_max", "fp64_max", "ratio"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in local_rows:
            row = {k: r.get(k, "nan") for k in fields}
            for k in ("fp32_max", "fp64_max", "ratio"):
                v = row[k]
                row[k] = f"{v:.6e}" if isinstance(v, (int, float)) and v == v else "nan"
            w.writerow(row)


def main():
    p = argparse.ArgumentParser(description="OFT 精度消融审计")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-global-fp64", action="store_true",
                   help="跳过整网 FP64 前向 (省显存/时间)")
    args = p.parse_args()

    cfg = OFTAuditConfig()
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.device:
        cfg.device = args.device
    if args.seed is not None:
        cfg.seed = args.seed

    ckpt = args.checkpoint_dir
    if ckpt is None:
        d = os.path.join(cfg.output_dir, "adapter_checkpoint")
        if os.path.isdir(d):
            ckpt = d

    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 64)
    print("OFT 精度消融审计 (TF32 / FP32 / FP64)")
    print("=" * 64)

    res = run_precision_ablation(cfg, ckpt, global_fp64=not args.no_global_fp64)

    out_json = os.path.join(cfg.output_dir, "precision_ablation_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"\nJSON 已保存: {out_json}")

    out_csv = os.path.join(cfg.output_dir, "precision_ablation_local.csv")
    _write_csv(res["local_per_layer"], out_csv)
    print(f"CSV 已保存: {out_csv}")


if __name__ == "__main__":
    main()
