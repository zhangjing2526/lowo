"""
OFT 训练模块
=============
小规模训练精确 Cayley OFT 适配器，用于后续吸收审计。

训练对象:
  - base model 冻结
  - 仅 OFT 的正交因子 R 可学习 (严格 Cayley 参数化)
"""

import time

import numpy as np
import torch
import torch.nn as nn

from config import OFTAuditConfig


def _load_cifar10_subset(cfg: OFTAuditConfig, processor):
    """加载 CIFAR-10 子集并映射为模型输入。

    返回 (inputs, labels) 的 tensor 形式。
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("需要 datasets 库: pip install datasets")

    print("  加载 CIFAR-10 ...")
    ds = load_dataset("cifar10", split="train")
    if cfg.max_train_samples:
        ds = ds.select(range(min(cfg.max_train_samples, len(ds))))

    import io
    from PIL import Image

    pixel_values, labels = [], []
    # CIFAR-10 是 PIL Image + label
    for i in range(len(ds)):
        img = ds[i]["img"]
        lab = ds[i]["label"]
        labels.append(lab)
        pixel_values.append(img)

    # 批量处理
    print(f"  处理 {len(pixel_values)} 张图片 ...")
    inputs = processor(images=pixel_values, return_tensors="pt")
    inputs["labels"] = torch.tensor(labels)
    return inputs


def train_oft(model, processor, cfg: OFTAuditConfig):
    """训练 OFT adapter (base 冻结)。"""
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=0.01,
    )

    inputs = _load_cifar10_subset(cfg, processor)
    pixel_values = inputs["pixel_values"].to(device)
    labels = inputs["labels"].to(device)

    n = len(labels)
    n_batches = max(1, n // cfg.batch_size)

    print(f"  设备: {device}, 样本数: {n}, 批次数: {n_batches}")
    t0 = time.time()

    for epoch in range(cfg.num_epochs):
        epoch_loss = 0.0
        # shuffle
        perm = torch.randperm(n)
        for bidx in range(n_batches):
            idx = perm[bidx * cfg.batch_size: (bidx + 1) * cfg.batch_size]
            if len(idx) == 0:
                continue
            xb = pixel_values[idx]
            yb = labels[idx]

            optimizer.zero_grad()
            out = model(pixel_values=xb, labels=yb, return_dict=True)
            loss = out.loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg = epoch_loss / max(1, n_batches)
        print(f"  epoch {epoch+1}/{cfg.num_epochs}  loss={avg:.4f}  "
              f"elapsed={time.time()-t0:.1f}s")

    print("  训练完成")
    return model