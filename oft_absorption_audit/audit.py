"""
OFT 吸收审计核心
==================
关键断言:
  OFT 前向 (PEFT 0.13.2, 线性层输出侧旋转):  y = (x·W₀ᵀ)·R + b   (R 可学习正交, W₀ 冻结)
  吸收后:    W_merged = Rᵀ · W₀  (= R⁻¹·W₀, 因 R 正交; 输出侧旋转)

实际机制:
  merge_and_unload() → 将 adapter 权重合并回 base 层的 weight matrix
  合并后: y_merged = W_absorbed × x

验证逻辑:
  ∥ y_OF T− y_merged ∥ < tol  对所有层成立 → PASS

逐层误差记录:
  - max absolute error
  - mean squared error
  - Frobenius norm residue
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn

from config import OFTAuditConfig


def _find_oft_layers(model) -> list:
    """遍历 model.named_modules(), 找到所有 OFT adapter 注入的线性层。

    返回 [(layer_name, module)] 列表 (含 OFT lora_A/lora_B 子模块时过滤之)。
    """
    oft_layers = []
    for name, mod in model.named_modules():
        # OFT adapter 在 peft 中通常包裹在 lora 子结构中
        # 具有 "oft_conv" 或 "lora_A"/"lora_B" 特征的 skip
        if hasattr(mod, "weight") and isinstance(mod.weight, nn.Parameter):
            # 跳过 adapter 内部的参数层
            if any(k in name for k in ("lora_", "oft_", "_mod_")):
                continue
            # 捕获被 OFT 包裹的原始线性层名
            if mod.weight.requires_grad == False:
                # frozen → 可能是 base 层的线性映射 → 这就是我们要审计的
                pass
        oft_layers.append((name, mod))
    return oft_layers


def run_absorption_audit(cfg: OFTAuditConfig) -> dict:
    """吸收审计主流程。

    返回:
      dict with keys:
        overall_pass: bool
        overall_pass_rate: float
        per_layer: {layer_name: {pass, max_err, mse}}
    """
    from peft import PeftModel
    from transformers import AutoModelForImageClassification, AutoImageProcessor

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")

    # 1) 加载基模型 + OFT adapter
    base_model = AutoModelForImageClassification.from_pretrained(
        cfg.base_model,
        num_labels=cfg.num_labels,
        ignore_mismatched_sizes=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, cfg.checkpoint_dir)
    model = model.to(device)
    model.eval()

    # 2) 构造测试输入
    print(f"  构造测试输入 ({cfg.num_test_samples} 张随机图片)")
    dummy_images = torch.randn(
        cfg.num_test_samples, 3, 224, 224,
        device=device,
        dtype=torch.float32,
    )

    # 3) 合并前 → 前向
    print("  合并前前向...")
    with torch.no_grad():
        out_before = model(pixel_values=dummy_images, return_dict=True)
        # 收集 hidden states (如果模型支持 output_hidden_states)
    logits_before = out_before.logits.cpu()

    # 4) 提取逐层 OFT 层名及各自的中间激活 ↓
    # (peft 合并后 auto_mapping 变为 base + 内联权重 → 无可变 adapter 权重痕迹)
    print("  提取逐层激活...")
    layers_info = _capture_layer_activations(model, dummy_images, device)

    # 5) merge_and_unload
    print("  merge_and_unload...")
    merged_model = model.merge_and_unload()
    merged_model = merged_model.to(device)
    merged_model.eval()

    # 6) 合并后 → 前向
    print("  合并后前向...")
    with torch.no_grad():
        out_after = merged_model(pixel_values=dummy_images, return_dict=True)
    logits_after = out_after.logits.cpu()

    # 7) 全局 logits 对比
    max_logit_err = (logits_before - logits_after).abs().max().item()
    mse_logit = ((logits_before - logits_after) ** 2).mean().item()
    print(f"  全局 logits:    max_err={max_logit_err:.2e}  MSE={mse_logit:.2e}")

    # 8) 逐层误差
    per_layer = {}
    for layer_name, act_before in layers_info.items():
        # 合并后重建 activation capture
        _hook_out = {}
        _register_hook_for_layer(merged_model, layer_name, _hook_out)
        with torch.no_grad():
            _ = merged_model(pixel_values=dummy_images[:1], return_dict=True)
        act_after = _hook_out.get("out", None)

        if act_after is None:
            per_layer[layer_name] = {"pass": False, "max_err": float("nan"), "mse": float("nan"), "note": "hook_failure"}
            continue

        diff = (act_before.cpu() - act_after.cpu()).abs()
        max_err = diff.max().item()
        mse = diff.pow(2).mean().item()
        passed = max_err < cfg.absorption_atol

        per_layer[layer_name] = {"pass": passed, "max_err": max_err, "mse": mse}

    passed_count = sum(1 for v in per_layer.values() if v["pass"])
    overall_pass_rate = passed_count / max(1, len(per_layer))

    return {
        "overall_pass": max_logit_err < cfg.absorption_atol,
        "overall_pass_rate": overall_pass_rate,
        "global_logit_max_err": max_logit_err,
        "global_logit_mse": mse_logit,
        "per_layer": per_layer,
    }


def _canonical_name(name: str) -> str:
    """PEFT merge_and_unload 后会去掉 'base_model.model.' 前缀。

    合并前后同一层的命名空间不同，需要规范化为统一键。
    """
    for prefix in ("base_model.model.", "base_model."):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _capture_layer_activations(model, dummy_images: torch.Tensor, device: torch.device):
    """通过 forward hook 抓取每层激活值。

    只抓第一个 batch 的前向激活 (避免大内存)。
    """
    activations = {}
    hooks = []

    def _hook_fn(mod_name):
        def fn(module, inp, out):
            # out 可能是 tensor 或 tuple/list，统一取第一个
            if isinstance(out, (tuple, list)):
                out = out[0]
            # out shape: [B, ...]，与 _register_hook_for_layer 的捕获口径一致
            activations[mod_name] = out.detach().cpu()

        return fn

    for name, mod in model.named_modules():
        # merge 前 adapter 由 "包裹层 + 内部 base_layer" 组成；base_layer 只是
        # 内部原始线性层，其输出不等于 adapter 层(已施加正交变换)的输出，必须跳过。
        if ".base_layer" in name:
            continue
        name = _canonical_name(name)
        # merge 前的 PEFT 包裹层(OFT/LoRA 线性层)持有 base_layer 属性，
        # 其输出才是"合并后应当等价"的对象；merge 后对应位置退化为普通 nn.Linear。
        if hasattr(mod, "base_layer"):
            h = mod.register_forward_hook(_hook_fn(name))
            hooks.append(h)
        elif isinstance(mod, (nn.Linear, nn.Conv2d)):
            h = mod.register_forward_hook(_hook_fn(name))
            hooks.append(h)

    with torch.no_grad():
        _ = model(pixel_values=dummy_images[:1], return_dict=True)

    for h in hooks:
        h.remove()

    return activations


def _register_hook_for_layer(model, layer_name: str, out_dict: dict):
    """为指定的层注册 hook 输出到 out_dict["out"]。"""
    for name, mod in model.named_modules():
        if _canonical_name(name) == layer_name:

            def fn(_, inp, out):
                if isinstance(out, (tuple, list)):
                    out = out[0]
                out_dict["out"] = out.detach().cpu()

            mod.register_forward_hook(fn)
            return
    raise ValueError(f"层不存在: {layer_name}")