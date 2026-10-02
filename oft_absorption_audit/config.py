"""
OFT Absorption Audit — 配置文件
用途: 精确 Cayley OFT 训练 + 吸收审计实验的统一参数入口
论文定位: "外部方法复现验证" —— 验证可学习正交因子可被相邻 dense 层吸收
"""

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class OFTAuditConfig:
    # ── 模型 ──────────────────────────────────────────
    base_model: str = "facebook/dinov2-large"
    """公开第三方预训练模型 ID (HuggingFace hub)"""

    num_labels: int = 10
    """分类头类别数，默认 CIFAR-10 的 10 类
    dinov2-large 是无分类头的 backbone，必须显式指定 num_labels，
    否则 AutoModelForImageClassification 会按 config 默认值建分类头，
    与 CIFAR-10 标签(0-9)不匹配 → nll_loss device-side assert"""

    # ── OFT 参数 ──────────────────────────────────────
    ## 严格精确 Cayley 参数化 (use_cayley_neumann=False)
    ## 目的: 确保因子是严格正交矩阵，对应 LOWO 的 Q ∈ SO(8) 假设
    oft_block_size: int = 8
    """OFT 块大小，对应正交矩阵的划分粒度"""
    target_modules: List[str] = field(default_factory=lambda: [
        "query", "value", "key", "dense"
    ])
    """OFT 注入的目标线性层"""

    # ── 训练 ──────────────────────────────────────────
    ## 使用小型公开数据集 (CIFAR-10 / ImageNet-1k subset)，确保可复现
    dataset_name: str = "cifar10"
    """训练数据集"""
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 3
    max_train_samples: int = 5000
    """训练样本上限 —— 小规模快速训练，仅用于验证吸收机制"""

    # ── 审计 ──────────────────────────────────────────
    absorption_atol: float = 1e-5
    """吸收前后前向输出误差阈值 (float32)"""
    absorption_rtol: float = 1e-4
    """相对误差阈值"""
    num_test_samples: int = 256
    """验证集样本数"""

    # ── 输出 ──────────────────────────────────────────
    output_dir: str = "outputs/oft_audit"
    checkpoint_dir: str = "outputs/oft_adapter_checkpoint"
    seed: int = 42
    device: str = "cuda"
    """推理/训练设备: cuda / cpu (CUDA 不可用时会自动回退 cpu)"""