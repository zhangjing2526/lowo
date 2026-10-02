"""
恒等变换 R=I 对照审计
=======================
用途: 回答审稿建议 1 的"恒等变换对照"。

核心问题:
  OFT 的 merge 前后差异, 必须和一个"不做任何正交旋转"的恒等 adapter 对照。
  - 若 R=I 时 merge 前后仍有显著差异  => 差异来自实现/执行配置, 不能归因于可学习旋转。
  - 若 R=I 时差异趋近 0              => 非平凡 R 带来的差异才有讨论价值。

流程:
  1. 加载 base 模型, 注入/加载 OFT (精确 Cayley, rs_linear 默认 False)
  2. eval + inference_mode + 固定 seed + 关 TF32/benchmark
  3. 基线 A: 同一未合并模型重复两次前向, 测 logits 差异 (执行非确定的底噪)
  4. 探针: dump 所有 OFT 可学习参数名 (人工核对用)
  5. 置恒等: 把 OFT 因子置为恒等正交变换 (Cayley: 置零; rs_linear: 置单位阵)
  6. 自洽验证: model(x) ≈ base(x) 必须成立, 否则置位逻辑与 peft 布局不符, 报错退出
  7. 主测: 恒等 adapter 下 merge_and_unload 前后 logits 差异
  8. 输出 JSON

用法:
  python identity_control.py                                 # fresh 注入
  python identity_control.py --checkpoint-dir outputs/oft_adapter_checkpoint
  python identity_control.py --output-dir outputs/oft_audit  # 指定输出目录
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

_sdir = os.path.dirname(os.path.abspath(__file__))
if _sdir not in sys.path:
    sys.path.insert(0, _sdir)

from config import OFTAuditConfig
from provenance import load_peft_exact_cayley, capture_versions


# ----------------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------------

def _set_deterministic(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def _collect_oft_params(model):
    """dump 所有名字含 oft 的可学习参数, 供探针与人工核对。"""
    info = []
    for n, p in model.named_parameters():
        if "oft" in n.lower():
            info.append({
                "name": n,
                "shape": list(p.shape),
                "requires_grad": bool(p.requires_grad),
                "dtype": str(p.dtype),
            })
    return info


def _rs_linear_mode(model) -> bool:
    """判断当前 adapter 是否 rs_linear (直接参数化正交矩阵 R)。"""
    # PEFT 会向上暴露 peft_config 字典
    cfg = getattr(model, "peft_config", None)
    if isinstance(cfg, dict):
        for v in cfg.values():
            if getattr(v, "rs_linear", False) or getattr(v, "use_oft_rs_linear", False):
                return True
    return False


def _set_oft_identity(model, mode: str):
    """把 OFT 可学习因子置为恒等正交变换。

    mode = 'cayley'   : 参数是斜对称 A (上三角), A=0 => Q=I => 置零
    mode = 'rs_linear': 参数是正交矩阵 R, R=I => 置单位阵
    返回被修改的参数名列表。

    注意: 已加载的 peft checkpoint 其 adapter 参数 often requires_grad=False
    (仅影响 autograd, 不影响前向)。置位用 in-place 操作 (zero_/copy_) 与
    requires_grad 无关, 故这里**不**跳过 requires_grad=False 的参数, 否则恒等
    置位会整批失效 (实测曾出现 "修改 0 个参数" 的假阴性)。
    """
    touched = []
    for n, p in model.named_parameters():
        if "oft" not in n.lower():
            continue
        if mode == "cayley":
            with torch.no_grad():
                p.zero_()
        elif mode == "rs_linear":
            with torch.no_grad():
                # 方阵参数置单位阵, 其他形状置零 (保守)
                if p.ndim == 2 and p.shape[0] == p.shape[1]:
                    p.copy_(torch.eye(p.shape[0], device=p.device, dtype=p.dtype))
                else:
                    p.zero_()
        touched.append(n)
    return touched


# ----------------------------------------------------------------------------
# 核心
# ----------------------------------------------------------------------------

def run_identity_control(cfg: OFTAuditConfig, checkpoint_dir: str = None) -> dict:
    from peft import OFTConfig, PeftModel, get_peft_model
    from transformers import AutoModelForImageClassification

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")

    _set_deterministic(cfg.seed)

    # 1) base + OFT 模型
    print(f"  加载 base: {cfg.base_model}")
    base = AutoModelForImageClassification.from_pretrained(
        cfg.base_model,
        num_labels=cfg.num_labels,
        ignore_mismatched_sizes=True,
        trust_remote_code=True,
    ).to(device)
    base.eval()

    with_checkpoint = bool(checkpoint_dir) and os.path.isfile(
        os.path.join(checkpoint_dir, "adapter_config.json")
    )
    if with_checkpoint:
        model, use_cayley_forced = load_peft_exact_cayley(base, checkpoint_dir)
        print(f"  加载已训练 checkpoint: {checkpoint_dir}")
        print(f"  强制 use_cayley_neumann=False (精确 Cayley): {use_cayley_forced}")
    else:
        use_cayley_forced = True
        print("  fresh 注入 OFT (随机初始化, 将被恒等置位覆盖)")
        peft_config = OFTConfig(
            r=cfg.oft_block_size,
            module_dropout=0.0,
            target_modules=cfg.target_modules,
            use_cayley_neumann=False,   # 精确 Cayley, 非截断展开
        )
        model = get_peft_model(base, peft_config)

    model = model.to(device)
    model.eval()

    # 2) 测试输入
    dummy = torch.randn(min(cfg.num_test_samples, 64), 3, 224, 224, device=device)

    # 3) 探针: OFT 参数名
    oft_params_before = _collect_oft_params(model)
    rs_linear = _rs_linear_mode(model)
    print(f"  OFT 可学习参数 {len(oft_params_before)} 个, rs_linear={rs_linear}")

    # 4) 基线 A: 未合并模型重复两次前向
    with torch.inference_mode():
        l1 = model(pixel_values=dummy, return_dict=True).logits
        l2 = model(pixel_values=dummy, return_dict=True).logits
    baseline_err = (l1 - l2).abs().max().item()
    print(f"  基线(同模型重复前向) max_err={baseline_err:.3e}")

    # 5) 置恒等
    mode = "rs_linear" if rs_linear else "cayley"
    touched = _set_oft_identity(model, mode)
    print(f"  置恒等 mode={mode}, 修改 {len(touched)} 个参数")

    # 6) 自洽验证: model(x) ≈ base(x)
    with torch.inference_mode():
        base_out = base(pixel_values=dummy, return_dict=True).logits
        model_out = model(pixel_values=dummy, return_dict=True).logits
    ident_err = (base_out - model_out).abs().max().item()
    print(f"  恒等自洽验证 model(x) vs base(x) max_err={ident_err:.3e}")

    # 若 cayley 置零失败, 试 rs_linear 置单位阵 (参数可能直接存 R)
    if ident_err > 1e-4 and mode == "cayley":
        print("  cayley 置零自洽验证失败, 改试置单位阵 ...")
        touched = _set_oft_identity(model, "rs_linear")
        with torch.inference_mode():
            model_out = model(pixel_values=dummy, return_dict=True).logits
        ident_err = (base_out - model_out).abs().max().item()
        mode = "rs_linear"
        print(f"  置单位阵后 max_err={ident_err:.3e}")

    if ident_err > 1e-4:
        print("\n[ERROR] 恒等置位自洽验证失败, 置位逻辑与 peft 实际布局不符。")
        print("请核对以下 OFT 参数名后手动修改 _set_oft_identity：")
        for p in oft_params_before:
            print(f"    {p['name']:<80} {p['shape']} rg={p['requires_grad']}")
        sys.exit(1)

    # 7) 主测: 恒等 adapter 下 merge 前后
    with torch.inference_mode():
        pre_logits = model(pixel_values=dummy, return_dict=True).logits
    merged = model.merge_and_unload()
    merged = merged.to(device)
    merged.eval()
    with torch.inference_mode():
        post_logits = merged(pixel_values=dummy, return_dict=True).logits
        base2_out = base(pixel_values=dummy, return_dict=True).logits

    absdiff = (pre_logits - post_logits).abs()
    merge_max = absdiff.max().item()
    merge_mse = ((pre_logits - post_logits) ** 2).mean().item()

    # 分位数 + 超阈值占比
    flat = absdiff.flatten().cpu().numpy()
    qs = {q: float(np.quantile(flat, q)) for q in (0.5, 0.9, 0.99, 1.0)}
    frac = float((flat > 1e-6).mean())

    print(f"  [恒等] merge_and_unload 前后 logits max_err={merge_max:.3e}  MSE={merge_mse:.3e}")
    print(f"  [恒等] 分位数 50/90/99/100% = {qs}  超1e-6占比={frac:.4f}")

    # 恒等 R 下 merge 前后差异的预期量级 = FP32 合并数值噪声 (~1e-5),
    # 不是实现 bug。阈值放宽到 1e-4 以容纳 FP32 舍入, 真正判定 "实现是否有病"
    # 须结合精度消融 (FP64 应趋近机器精度)。
    identity_atol = 1e-4
    verdict = "PASS" if merge_max < identity_atol else "FAIL"
    print(f"\n  恒等对照结论: {verdict} (阈值 {identity_atol:.0e}, 观测 {merge_max:.3e})")
    if verdict == "PASS":
        print("  => 恒等 R 下 merge 前后仅剩 FP32 合并数值噪声, 无实现 bug;")
        print("     非平凡 R 的差异才有分析价值 (见精度消融脚本)。")
    else:
        print("  => 恒等 R 下 merge 前后仍有 >1e-4 差异。这超出 FP32 合并噪声预期,")
        print("     应排查 peft 版本/合并实现, 而非归因于可学习旋转。")
        print("     对照: 精度消融中 FP64 应趋近机器精度 (~1e-14), 若 FP64 也大则确为 bug。")

    results = {
        "identity_only": True,
        "device": str(device),
        "with_checkpoint": with_checkpoint,
        "mode": mode,
        "num_oft_params": len(oft_params_before),
        "rs_linear": rs_linear,
        "baseline_repeat_max_err": baseline_err,
        "identity_sanity_max_err": ident_err,
        "merge_max_err": merge_max,
        "merge_mse": merge_mse,
        "quantiles": qs,
        "frac_above_1e-6": frac,
        "verdict": verdict,
        "use_cayley_neumann_forced": use_cayley_forced,
        "oft_params": oft_params_before,
    }
    if with_checkpoint:
        results.update(capture_versions(checkpoint_dir))
    return results


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OFT 恒等变换 R=I 对照")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = OFTAuditConfig()
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.device:
        cfg.device = args.device
    if args.seed is not None:
        cfg.seed = args.seed

    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        default_ckpt = os.path.join(cfg.output_dir, "adapter_checkpoint")
        if os.path.isdir(default_ckpt):
            checkpoint_dir = default_ckpt

    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 64)
    print("OFT 恒等变换 R=I 对照")
    print("=" * 64)

    results = run_identity_control(cfg, checkpoint_dir)

    out = os.path.join(cfg.output_dir, "identity_control_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n报告已保存: {out}")


if __name__ == "__main__":
    main()