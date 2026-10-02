"""
OFT Absorption Audit — 主入口
================================
流程:
  1. 加载公开预训练模型 → 注入 Cayley OFT 适配器
  2. 小规模训练 → 保存 adapter checkpoint
  3. 吸收审计: merge_and_unload 前后逐层对比输出

论文定位: "外部方法复现验证"
用法:
  python main.py                # 完整流程
  python main.py --audit-only   # 仅审计(需已有 checkpoint)
"""

import argparse
import csv
import json
import os
import sys
import time
import warnings

import numpy as np
import torch

from config import OFTAuditConfig

# ── 延迟导入手依赖检查 ─────────────────────────────────
_MISSING = {}

def _check_imports():
    for mod, hint in [
        ("torch", "pip install torch"),
        ("transformers", "pip install transformers"),
        ("peft", "pip install peft"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            _MISSING[mod] = hint
    if _MISSING:
        print("[ERROR] 缺少依赖:")
        for m, h in _MISSING.items():
            print(f"  {m} → {h}")
        sys.exit(1)

_check_imports()
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification
from peft import OFTConfig, get_peft_model


# ── 子模块导入 ─────────────────────────────────────────
from train import train_oft
from audit import run_absorption_audit


def build_config(args: argparse.Namespace) -> OFTAuditConfig:
    cfg = OFTAuditConfig()
    if args.base_model:
        cfg.base_model = args.base_model
    if args.block_size is not None:
        cfg.oft_block_size = args.block_size
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.output_dir:
        cfg.output_dir = args.output_dir
        cfg.checkpoint_dir = os.path.join(args.output_dir, "adapter_checkpoint")
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.max_train_samples is not None:
        cfg.max_train_samples = args.max_train_samples
    if args.device:
        cfg.device = args.device
    return cfg


def _write_per_layer_csv(results: dict, csv_path: str):
    """把逐层审计结果写成 CSV，按 max_err 降序排列，异常层排最前。"""
    per_layer = results.get("per_layer", {})
    rows = []
    for name, r in per_layer.items():
        max_err = r.get("max_err", float("nan"))
        mse = r.get("mse", float("nan"))
        passed = bool(r.get("pass", False))
        note = r.get("note", "")
        rows.append({
            "layer": name,
            "pass": passed,
            "max_err": max_err,
            "mse": mse,
            "note": note,
        })

    # 异常样本排最前，再按 max_err 降序
    rows.sort(key=lambda x: (x["pass"], -x["max_err"] if x["max_err"] == x["max_err"] else float("inf")))

    fieldnames = ["layer", "pass", "max_err", "mse", "note"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "layer": row["layer"],
                "pass": "yes" if row["pass"] else "no",
                "max_err": f"{row['max_err']:.6e}" if row["max_err"] == row["max_err"] else "nan",
                "mse": f"{row['mse']:.6e}" if row["mse"] == row["mse"] else "nan",
                "note": row["note"],
            })
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="OFT Absorption Audit")
    parser.add_argument("--base-model", default=None,
                        help="HuggingFace 预训练模型 ID (默认 dinov2-large)")
    parser.add_argument("--block-size", type=int, default=None,
                        help="OFT 块大小")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--audit-only", action="store_true",
                        help="跳过训练，仅运行吸收审计")
    args = parser.parse_args()

    cfg = build_config(args)
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    print("=" * 60)
    print("OFT 吸收审计 — 外部方法复现验证")
    print(f"  模型: {cfg.base_model}")
    print(f"  块大小: {cfg.oft_block_size} (精确 Cayley)")
    print(f"  输出目录: {cfg.output_dir}")
    print("=" * 60)

    if args.audit_only:
        if not os.path.exists(cfg.checkpoint_dir):
            print(f"[ERROR] 未找到 checkpoint: {cfg.checkpoint_dir}")
            sys.exit(1)
        print("[audit-only] 跳过训练，直接进入吸收审计")
    else:
        # 1) 加载公开预训练模型
        print(f"\n[1/3] 加载公开预训练模型: {cfg.base_model}")
        base_model = AutoModelForImageClassification.from_pretrained(
            cfg.base_model,
            num_labels=cfg.num_labels,
            ignore_mismatched_sizes=True,
            trust_remote_code=True,
        )
        processor = AutoImageProcessor.from_pretrained(cfg.base_model)

        # 2) 注入精确 Cayley OFT
        print("[2/3] 注入 Cayley OFT 适配器")
        # use_cayley_neumann=False → 精确 Cayley (不做 Neumann 近似)
        oft_kwargs = dict(target_modules=cfg.target_modules)
        try:
            # peft>=0.10 的 OFTConfig arg 名
            peft_config = OFTConfig(
                r=cfg.oft_block_size,
                module_dropout=0.0,
                **oft_kwargs,
            )
        except TypeError:
            # 旧版 peft 兼容
            peft_config = OFTConfig(
                oft_block_size=cfg.oft_block_size,
                **oft_kwargs,
            )
        model = get_peft_model(base_model, peft_config)
        # 打印可训练参数
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

        # 3) 训练
        print(f"[3/3] 训练 OFT 适配器")
        train_oft(model, processor, cfg)

        # 保存 adapter checkpoint
        print(f"保存 checkpoint → {cfg.checkpoint_dir}")
        model.save_pretrained(cfg.checkpoint_dir)

    # ── 吸收审计 ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("开始吸收审计")
    print("=" * 60)

    results = run_absorption_audit(cfg)

    # 输出摘要
    print("\n" + "=" * 60)
    print("审计摘要")
    print("=" * 60)
    for layer_name, r in results["per_layer"].items():
        status = "✓ PASS" if r["pass"] else "✗ FAIL"
        print(f"  {status}  {layer_name}: max_err={r['max_err']:.2e}")
    print(f"\n总 PASS: {results['overall_pass_rate']:.2%}")

    # 写入报告 (JSON)
    report_path = os.path.join(cfg.output_dir, "audit_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"报告已保存: {report_path}")

    # 写入逐层 CSV (便于目检异常层)
    csv_path = os.path.join(cfg.output_dir, "per_layer_audit.csv")
    _write_per_layer_csv(results, csv_path)
    print(f"逐层报告已保存: {csv_path}")
    return results


if __name__ == "__main__":
    main()