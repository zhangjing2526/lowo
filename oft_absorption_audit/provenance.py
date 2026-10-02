"""
provenance.py — OFT 审计实验的可追溯性辅助
============================================
集中管理两件保证可证伪性的事:

1. 强制精确 Cayley 加载
   - PEFT 0.13.2 的 OFT 只有精确 Cayley 一种模式 (无 use_cayley_neumann 字段);
     但更新版 PEFT 引入该 flag 且默认值可能不同。训练用的是精确 Cayley,
     审计必须强制对齐, 否则 "有效 R" 与训练时不一致。
   - load_peft_exact_cayley() 始终尝试传 use_cayley_neumann=False; 老版本报 TypeError 时回退
     (该版本精确 Cayley 由构造保证, 等效)。

2. 实验报告 provenance 字段
   - adapter 权重文件 SHA256 / adapter_config SHA256 (把报告绑定到确切产物)
   - peft / transformers / torch 版本 (审计时环境)
   - use_cayley_neumann_forced (是否成功强制精确 Cayley)
"""

import hashlib
import json
import os


def _file_sha256(path: str) -> str:
    if not os.path.isfile(path):
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_oft_config(ckpt_dir: str):
    """读保存的 adapter_config, 强制精确 Cayley, 返回 (OFTConfig, forced: bool)。"""
    import json as _json
    from peft import OFTConfig

    with open(os.path.join(ckpt_dir, "adapter_config.json")) as f:
        raw = _json.load(f)
    r = int(raw.get("r", 8))
    target = raw.get("target_modules")
    if isinstance(target, (set, list)):
        target = list(target)
    kwargs = dict(r=r, target_modules=target, module_dropout=float(raw.get("module_dropout", 0.0)))
    try:
        kwargs["use_cayley_neumann"] = False
        peft_config = OFTConfig(**kwargs)
        forced = True
    except TypeError:
        # 老版本 (如 0.13.2) 无该字段 -> 精确 Cayley 由构造保证
        peft_config = OFTConfig(**{k: v for k, v in kwargs.items() if k != "use_cayley_neumann"})
        forced = False
    return peft_config, forced


def load_peft_exact_cayley(base, ckpt_dir: str):
    """在 base 上加载已训练 OFT, 强制精确 Cayley。返回 (model, forced: bool)。"""
    from peft import PeftModel

    peft_config, forced = build_oft_config(ckpt_dir)
    model = PeftModel.from_pretrained(base, ckpt_dir, peft_config=peft_config)
    return model, forced


def capture_versions(ckpt_dir: str) -> dict:
    """收集审计时环境与产物指纹, 合并进实验报告。"""
    import torch
    import transformers
    import peft

    adapter_path = os.path.join(ckpt_dir, "adapter_model.safetensors")
    cfg_path = os.path.join(ckpt_dir, "adapter_config.json")
    _, forced = build_oft_config(ckpt_dir) if ckpt_dir and os.path.isfile(cfg_path) else (None, True)
    return {
        "adapter_sha256": _file_sha256(adapter_path),
        "adapter_config_sha256": _file_sha256(cfg_path),
        "peft_version": peft.__version__,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "use_cayley_neumann_forced": forced,
    }
