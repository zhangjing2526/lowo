"""
固定输入单层重放审计
======================
解决审稿建议 3 的核心问题: 端到端传播误差 vs 单层局部吸收误差分离。

数学上, 合并前 OFT 层:  y = f_u(x)
               合并后层:   y = f_m(x)
真实吸收等价性应测量:   f_u(x) - f_m(x)   (同一固定输入 x)
而端到端对比测到的是:   f_u(x_l) - f_m(x̃_l)   (输入已不同, x_l ≠ x̃_l)

本脚本:
  1. 加载 OFT 模型, eval + no_grad
  2. 前向一次, 用 hook 捕获每个注入层的**输入** (保存到 CPU)
  3. 重放自洽验证: 用保存的输入直接调用 mod(x), 应与 hook 捕获的输出一致
     (这一步验证"输入提取 + 单模块调用"整套重放链路正确, 否则后续无意义)
  4. merge_and_unload
  5. 对合并后每个对应层, 用**同一个保存的输入 x** 直接调用, 得到局部输出
     局部吸收误差 = f_u(x) - f_m(x)  => 这是真正的单层吸收误差
  6. 对照: 合并后完整前向, hook 捕获每层输入, 得到端到端传播误差
  7. 输出 CSV + JSON

用法:
  python fixed_input_layer_replay.py
  python fixed_input_layer_replay.py --checkpoint-dir outputs/oft_adapter_checkpoint
  python fixed_input_layer_replay.py --output-dir outputs/oft_audit
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
from provenance import load_peft_exact_cayley, capture_versions

try:
    from audit import _canonical_name
except Exception:
    def _canonical_name(name: str) -> str:
        for p in ("base_model.model.", "base_model."):
            if name.startswith(p):
                return name[len(p):]
        return name


# ----------------------------------------------------------------------------
# 模块调用封装
# ----------------------------------------------------------------------------

def _as_tensor(x):
    """把 hook 的 inp/out 提取为单个 tensor。"""
    if isinstance(x, (tuple, list)):
        return x[0]
    return x


def _call_module(mod, x):
    """用单个 tensor x 直接调用模块 mod, 返回首个输出 tensor。

    兼容 nn.Linear(nn.Conv2d) 与 PEFT OFT 包裹层 (接收单 tensor)。
    部分模块 forward 签名无位置参数陷阱 (如 nn.MultiheadAttention),
    但这里只针对被 OFT 注入的线性层, 均为单张量 in/out, 直接 mod(x)。
    """
    with torch.no_grad():
        y = mod(x)
    return _as_tensor(y)


# ----------------------------------------------------------------------------
# 捕获输入
# ----------------------------------------------------------------------------

def _capture_inputs(model, dummy, target_keys):
    """前向一次, 捕获 target_keys 中每一层的输入 (转发到 CPU)。

    返回 {layer_name: input_tensor}。
    跳过 base_layer (只抓 OFT 包裹层或合并后的 nn.Linear)。
    """
    inputs = {}
    hooks = []

    def make_hook(name):
        def fn(mod, inp, out):
            t = _as_tensor(inp)
            inputs[name] = t.detach().to("cpu")
        return fn

    for name, mod in model.named_modules():
        if ".base_layer" in name:
            continue
        cname = _canonical_name(name)
        if cname not in target_keys:
            continue
        h = mod.register_forward_hook(make_hook(cname))
        hooks.append(h)

    model.eval()
    with torch.no_grad():
        model(pixel_values=dummy, return_dict=True)

    for h in hooks:
        h.remove()
    return inputs


def _capture_outputs(model, dummy, target_keys):
    """捕获 target_keys 中每一层的输出 (用于重放自洽验证)。"""
    outs = {}
    hooks = []

    def make_hook(name):
        def fn(mod, inp, out):
            t = _as_tensor(out)
            outs[name] = t.detach().to("cpu")
        return fn

    for name, mod in model.named_modules():
        if ".base_layer" in name:
            continue
        cname = _canonical_name(name)
        if cname not in target_keys:
            continue
        h = mod.register_forward_hook(make_hook(cname))
        hooks.append(h)

    model.eval()
    with torch.no_grad():
        model(pixel_values=dummy, return_dict=True)

    for h in hooks:
        h.remove()
    return outs


# ----------------------------------------------------------------------------
# 核心
# ----------------------------------------------------------------------------

def run_fixed_input_replay(cfg: OFTAuditConfig, checkpoint_dir: str = None):
    from peft import OFTConfig, PeftModel, get_peft_model
    from transformers import AutoModelForImageClassification

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # 1) 模型
    base = AutoModelForImageClassification.from_pretrained(
        cfg.base_model,
        num_labels=cfg.num_labels,
        ignore_mismatched_sizes=True,
        trust_remote_code=True,
    ).to(device)
    base.eval()

    with_ckpt = bool(checkpoint_dir) and os.path.isfile(
        os.path.join(checkpoint_dir, "adapter_config.json")
    )
    if with_ckpt:
        model, use_cayley_forced = load_peft_exact_cayley(base, checkpoint_dir)
        print(f"  已加载 checkpoint: {checkpoint_dir}")
        print(f"  强制 use_cayley_neumann=False (精确 Cayley): {use_cayley_forced}")
    else:
        use_cayley_forced = True
        model = get_peft_model(base, OFTConfig(
            r=cfg.oft_block_size,
            module_dropout=0.0,
            target_modules=cfg.target_modules,
            use_cayley_neumann=False,
        ))
        print("  fresh 注入 OFT")

    model = model.to(device)
    model.eval()

    # 2) 找到被注入的层 (合并前持有 base_layer 属性的包裹层)
    inject_keys = set()
    inject_mods_before = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "base_layer") and ".base_layer" not in name:
            cname = _canonical_name(name)
            inject_keys.add(cname)
            inject_mods_before[cname] = mod
    print(f"  被注入层数: {len(inject_keys)}")
    if not inject_keys:
        print("[ERROR] 未找到任何 OFT 注入层")
        sys.exit(1)

    dummy = torch.randn(min(cfg.num_test_samples, 8), 3, 224, 224, device=device)

    # 3) 捕获合并前每层输入 + 输出
    inputs_before = _capture_inputs(model, dummy, inject_keys)
    outputs_before = _capture_outputs(model, dummy, inject_keys)

    # 4) 重放自洽验证: 用捕获输入直接调用, 应与捕获输出一致
    sanity_rows = {}
    for cname in inject_keys:
        if cname not in inputs_before or cname not in outputs_before:
            sanity_rows[cname] = float("nan")
            continue
        x = inputs_before[cname].to(device)
        y_replay = _call_module(inject_mods_before[cname], x).to("cpu")
        y_hook = outputs_before[cname]
        diff = (y_replay - y_hook).abs().max().item()
        sanity_rows[cname] = diff
    worst_sanity = max((v for v in sanity_rows.values() if v == v), default=0.0)
    print(f"  重放自洽验证 worst max_err = {worst_sanity:.3e}")

    # 5) merge_and_unload
    # 先深拷贝未合并模型: merge_and_unload 会原地修改 model (把 R 吸收进 base_layer 并卸载 adapter),
    # 导致 inject_mods_before 持有的模块引用失效 (再调用会走已合并状态, 使局部吸收误差假归零)。
    import copy as _copy
    model_unmerged_copy = _copy.deepcopy(model)
    print("  merge_and_unload ...")
    merged = model.merge_and_unload()
    merged = merged.to(device)
    merged.eval()

    # 6) 合并后对应层
    inject_mods_after = {}
    merged_keys = set()
    for name, mod in merged.named_modules():
        if ".base_layer" in name:
            continue
        cname = _canonical_name(name)
        if cname in inject_keys:
            inject_mods_after[cname] = mod
            merged_keys.add(cname)
    print(f"  合并后找到对应层: {len(merged_keys)} / {len(inject_keys)}")

    # 7) 相同固定输入下的局部吸收误差
    local_rows = {}
    for cname in inject_keys:
        x = inputs_before.get(cname)
        if x is None or cname not in inject_mods_after:
            local_rows[cname] = {"local_max": float("nan"), "note": "missing before-input or after-mod"}
            continue
        x = x.to(device)
        # 从未合并副本取模块做 unmerged replay (原 inject_mods_before 在 merge 后已失效)
        umod = None
        for n, m in model_unmerged_copy.named_modules():
            if _canonical_name(n) == cname:
                umod = m
                break
        if umod is None:
            local_rows[cname] = {"local_max": float("nan"), "note": "unmerged module lost after deepcopy"}
            continue
        y_u = _call_module(umod, x)
        y_m = _call_module(inject_mods_after[cname], x)
        d = (y_u - y_m).abs()
        local_rows[cname] = {
            "local_max": d.max().item(),
            "local_mse": ((y_u - y_m) ** 2).mean().item(),
        }

    # 8) 端到端传播误差对照
    e2e_inputs_after = _capture_inputs(merged, dummy, inject_keys)
    e2e_rows = {}
    for cname in inject_keys:
        if cname not in e2e_inputs_after or cname not in inputs_before:
            e2e_rows[cname] = {"e2e_input_max": float("nan")}
            continue
        x_before = inputs_before[cname].to(device)
        x_after = e2e_inputs_after[cname].to(device)
        e2e_rows[cname] = {
            "e2e_input_max": (x_before - x_after).abs().max().item(),
        }

    # 9) 汇总
    per_layer = []
    for cname in sorted(inject_keys, key=lambda k: k):
        loc = local_rows.get(cname, {})
        e2e = e2e_rows.get(cname, {})
        san = sanity_rows.get(cname, float("nan"))
        per_layer.append({
            "layer": cname,
            "sanity_replay_max": san,
            "local_absorb_max": loc.get("local_max", float("nan")),
            "local_absorb_mse": loc.get("local_mse", float("nan")),
            "e2e_input_max": e2e.get("e2e_input_max", float("nan")),
            "input_shape": list(inputs_before[cname].shape) if cname in inputs_before else None,
        })

    results = {
        "num_inject_layers": len(inject_keys),
        "worst_sanity_replay_max": worst_sanity,
        "use_cayley_neumann_forced": use_cayley_forced,
        "per_layer": per_layer,
    }
    if with_ckpt:
        results.update(capture_versions(checkpoint_dir))
    return results


# ----------------------------------------------------------------------------
# 输出
# ----------------------------------------------------------------------------

def _write_csv(per_layer, path):
    fieldnames = ["layer", "sanity_replay_max", "local_absorb_max",
                  "local_absorb_mse", "e2e_input_max", "input_shape"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_layer:
            row = {k: r.get(k, "nan") for k in fieldnames}
            for k in ("sanity_replay_max", "local_absorb_max",
                      "local_absorb_mse", "e2e_input_max"):
                v = row[k]
                row[k] = f"{v:.6e}" if isinstance(v, (int, float)) and v == v else "nan"
            row["input_shape"] = str(row["input_shape"])
            w.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="OFT 固定输入单层重放")
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

    ckpt = args.checkpoint_dir
    if ckpt is None:
        d = os.path.join(cfg.output_dir, "adapter_checkpoint")
        if os.path.isdir(d):
            ckpt = d

    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 64)
    print("OFT 固定输入单层重放审计")
    print("=" * 64)

    results = run_fixed_input_replay(cfg, ckpt)

    json_path = os.path.join(cfg.output_dir, "fixed_input_replay.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"JSON 已保存: {json_path}")

    csv_path = os.path.join(cfg.output_dir, "fixed_input_replay.csv")
    _write_csv(results["per_layer"], csv_path)
    print(f"CSV 已保存: {csv_path}")

    # 控制台摘要
    print("\n摘要 (local_absorb_max 排序前 10):")
    valid = [r for r in results["per_layer"] if r["local_absorb_max"] == r["local_absorb_max"]]
    valid.sort(key=lambda r: -r["local_absorb_max"])
    for r in valid[:10]:
        print(f"  {r['layer']:<70} local={r['local_absorb_max']:.3e}  "
              f"e2e_input={r['e2e_input_max']:.3e}")


if __name__ == "__main__":
    main()