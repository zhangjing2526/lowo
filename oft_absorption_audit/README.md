# OFT Absorption Audit — 外部方法复现验证实验

本目录实现 **OFT（Orthogonal Finetuning）吸收审计**，用于论文中 "外部方法复现验证" 小节。

## 目标

验证逻辑:
- OFT 将可学习正交因子 `R` 注入相邻 dense 层；在 PEFT 0.13.2 中为**输出侧**旋转，前向为 `y = (x·W₀ᵀ)·R + b`
- `merge_and_unload()` 将 `R` 吸收进 `W₀`，等价写出 `W_merged = Rᵀ·W₀`，推理零开销
- 审计: 吸收前后逐层输出是否一致（量化 residue norm）

## 文件结构

```
oft_absorption_audit/
├── config.py                 # 实验统一配置 (dataclass)
├── main.py                   # 主入口: 训练 + 审计
├── train.py                  # CIFAR-10 小规模 OFT 训练
├── audit.py                  # 吸收审计 (逐层 hook 对比, 原始版)
├── identity_control.py       # 诊断 1: 恒等变换 R=I 对照 (查实现 bug)
├── fixed_input_layer_replay.py  # 诊断 2: 固定输入单层重放 (分离局部 vs 端到端误差)
├── precision_ablation.py     # 诊断 3: 精度消融 TF32 / FP32 / FP64
├── nonidentity_recovery.py   # 非恒等恢复验证: 独立进程重载, 逐元素核对参数/计算有效 R/‖R-I‖_F/adapter 参与, 证据链
├── provenance.py             # 共享: 强制精确 Cayley 加载 + 实验报告 provenance 字段 (SHA256/版本)
├── selftest_nir_math.py      # 离线数学自检 (CPU/纯 torch, 无需 peft): 验证 Cayley 公式与权重反推 R 自洽, 36 项
├── requirements.txt          # 依赖 (含 safetensors, 供非恒等恢复验证读权重文件)
└── README.md                 # 本文件
```

## 重要定位与诚实标注

- 本实验 = **外部方法复现验证**: 公开基模型 (`facebook/dinov2-large`, ViT-Large, hidden=1024) + 自行训练的 CIFAR-10 OFT 适配器。
- **这不是** "第三方已发布、含已训练变换的公开检查点审计"。审稿明确: 可训练 ≠ 已学习过。
- `dinov2-large` 是 **ViT-Large 不是 ViT-Giant**; 默认无分类头, 故必须显式 `num_labels=10`。
- PEFT 的 `r` 是**块数**而非块大小: 对 hidden=1024, `r=8` 对应块大小 128 (`r × oft_block_size = 输入维数`)。
- **当前结论**: 经下方三组诊断实测, 已升级为**已验证证据** (见"实测结果快照")；不再停留在"待追查线索"。早期整网 e2e 差异 (原 max 0.117, MSE 2.45e-5) 来自**未严格控制变量**的旧 audit, 不应再作为受控结论引用, 也不应声称"已被解析为有限精度假象"。受控诊断 2+3 表明**局部吸收残差**落在有限精度量级, 不否定吸收的代数等价性。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 完整流程 (下载预训练模型 → 训练 OFT → 吸收审计)
python main.py

# 3. 仅审计 (需已有 checkpoint)
python main.py --audit-only
```

## 配置说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `base_model` | `facebook/dinov2-large` | 公开第三方预训练模型 |
| `oft_block_size` | `8` | Cayley 块大小 (对应 LOWO 的 8 维正交) |
| `num_epochs` | `3` | 训练 epoch 数 |
| `absorption_atol` | `1e-5` | 吸收误差阈值 (float32) |

## 三组诊断脚本（回应审稿建议）

审稿建议补三组关键检查再决定论文写法。三个脚本各自带**自洽验证**兜底, 不依赖对 peft 内部布局的猜测:

| 脚本 | 对应建议 | 作用 | 关键自洽校验 |
|---|---|---|---|
| `identity_control.py` | 建议 1 (恒等对照) | 注入 `R=I` 后再测合并前后差异; 若仍显著 ⇒ 差异来自实现噪声 | `model(x) ≈ base(x)` 必须成立 |
| `fixed_input_layer_replay.py` | 建议 3 (局部 vs 端到端) | 用同一固定输入 `x` 重放 `f_unmerged(x) − f_merged(x)`, 分离单层局部吸收误差与上游输入漂移 | `mod(捕获输入)` 必须等于 hook 捕获输出 |
| 脚本 | 对应建议 | 作用 | 关键自洽校验 |
|---|---|---|---|
| `nonidentity_recovery.py` | 建议 1+2 (provenance 与非恒等恢复) | 独立进程从磁盘重载：逐元素核对参数文件 → 算有效 R 与 ‖R−I‖_F → 证明 adapter 参与前向 → 在此状态下做局部重放与端到端。粒度到每层，含输入指纹/SHA256 | 两分支**训练外参数**（随机分类头）必须逐位一致；另见下方「输出读法」关于 FP32 固有反解误差的说明 |

```bash
# 诊断 1: 恒等变换对照 (fresh 注入或加载已训练 checkpoint)
python identity_control.py --checkpoint-dir outputs/oft_adapter_checkpoint

# 诊断 2: 固定输入单层重放
python fixed_input_layer_replay.py --checkpoint-dir outputs/oft_adapter_checkpoint

# 诊断 3: 精度消融 (--no-global-fp64 可跳过整网 FP64 以省显存)
python precision_ablation.py --checkpoint-dir outputs/oft_adapter_checkpoint
#   或: python precision_ablation.py --no-global-fp64

# 非恒等恢复验证 (独立进程, 从磁盘重新加载, 出完整证据链)
python nonidentity_recovery.py --checkpoint-dir outputs/oft_adapter_checkpoint
#   可选: --precisions fp32 fp64
```

### nonidentity_recovery.py 输出读法：各部分误差的正常/异常量级

> 依据本地离线数学自检 `selftest_nir_math.py`（CPU 合成数据，48/48 PASS，
> 直接调用本脚本的 `_cayley_R_from_param` 与 `_effective_R_from_weights`，
> 用 Python 本机 interpreter 即可跑：`python selftest_nir_math.py`）。

由**真实权重**反推有效旋转 R 时，精度受 `W₀` 条件数与该 dtype 舍入限制，
**FP32 与 FP64 的差异是固有的，不是 bug**：

| 量 | FP32 的正常量级 | FP64 的正常量级 | 说明 |
|---|---|---|---|
| `cayley_cross_max`（反推 R vs Cayley 公式重算 Q） | **1e-5 ~ 5e-5** | **~1e-14** | 自检实测 d=1024/block=128: fp32 1.9e-5、fp64 3.8e-14 |
| `R_orth_err`（反推 R 的正交性误差） | 1e-5 级 | ~1e-14 | 同上来源 |
| `local_absorb_max`（固定输入单层吸收误差） | **~1e-5**（实测 2.19e-05） | **~1e-14**（实测 2.84e-14） | **核心观测量**，与前两项无关；**不是"全零"**，FP32 档有真实残差 |
| `e2e_max_err`（端到端 logits） | ~1e-5（实测 2.63e-05） | ~1e-14（实测 1.24e-14） | 需先排除两分支随机初始化不同导致的伪差异 |
| `R_dev_fro`（‖R−I‖_F） | 应 ≫ 1e-4（实测 0.31–0.50） | 同 | <1e-4 说明退化成恒等，证据链不成立 |

**判读要点**：看到 `cayley_cross_max ≈ 1e-5` 不要以为是合并失败 —— 那是从 FP32 权重反解的固有误差。
要把它降下来，看 FP64 那一档即可（应对齐到 1e-14 级）。
真正的"吸收是否精确"看 `local_absorb_max`。

⚠️ 早期版本曾据截断打印 + 中位数得出"96 层局部误差全为 0"，**该说法已作废**。
真实最大值是 FP32 2.19e-05 / FP64 2.84e-14，论文措辞应为"FP64 下达到**双精度量级**"
（实测 2.84e-14，落在 Higham FP64 一阶界内；注意 FP64 机器精度 eps≈2.22e-16，2.84e-14 是**观测残差**而非机器精度本身）。

### 吸收残差的判据：Higham 浮点一阶误差界

FP32 的 `local_absorb_max ≈ 2e-5` 是**期望值而非缺陷** —— unmerged 走两级矩阵乘 `(x@W₀ᵀ)@R`，
merged 走一级 `x@Wmᵀ`，两条路径累加顺序不同，各自带自己的舍入。
因此判据**不能用拍脑袋的绝对阈值**（如 1e-5），而用数值分析的标准一阶界：

> Higham, *Accuracy and Stability of Numerical Algorithms*, Ch.2:
> `fl(Σ aᵢwᵢ)` 的误差 ≤ `γ_n · Σ|aᵢ||wᵢ|`，其中 `γ_n = n·u/(1−n·u)`，`u` 为单位舍入 (`finfo.eps/2`)。

实现见 `_local_forward_fp_bound()`：把第一级误差经 `|R|` 传播，加第二级自身舍入，
再与 merged 单级路径的界相加，得到逐元素上界（在 float64 中计算，避免界自身被舍入污染）。
这条界**不依赖 W₀ 条件数**（用的是实际计算出的中间量），所以即使 `W₀` 块条件数高达 1.25e7 也成立。

脚本按此输出三项：`local_fp_bound_max`（界）、`local_fp_bound_ratio_max`（实际/界）、
`local_within_fp_bound_all`（是否全部层层在界内），并纳入自动判定。
同时要求残差**随精度下降**（FP64/FP32 < 1e-6）才敢归因有限精度 —— 两者缺一不可，
单看"误差小"或单看"在界内"都不足以排除实现差异。

离线预测（`selftest_nir_math.py` 末节，dinov2 真实尺度 W₀~N(0,0.02)、x~N(0,1)、d=1024/block=128）：

| 档 | 局部误差 | 浮点一阶界 | err/bound |
|---|---|---|---|
| FP32 | 2.4e-06（真实 adapter 实测 2.19e-05） | ~7.6e-03 | ~3e-04 |
| FP64 | 1.1e-14 | ~1.4e-11 | ~8e-04 |

即远程 FP32 档应看到"界 ~1e-2、实际 ~1e-5、比值 ~1e-3" ⇒ 在界内且余量合理。
若比值 >1（超出界）则说明差异不止是普通 FP32 累加，需排查异常执行配置（如误开 TF32 —— 脚本已检测并报警）。

### 每次实验报告都含 provenance 字段 (provenance.py 注入)

四个脚本的输出 JSON 均追加:
- `adapter_sha256` / `adapter_config_sha256`: 把报告**绑定到确切产物文件**;
- `peft_version` / `transformers_version` / `torch_version`: 审计时环境;
- `use_cayley_neumann_forced`: 是否成功强制**精确 Cayley**。

**版本一致性（2026-09-19 实测，风险已解除）**：远程审计环境为
`peft=0.13.2 / transformers=4.39.3 / torch=2.6.0+cu126`，与 checkpiont 卡片记录的保存版本
PEFT 0.13.2 **一致**。0.13.2 的 `OFTConfig` 无 `use_cayley_neumann` 字段，传该参会 TypeError
并被 `provenance.load_peft_exact_cayley()` 回退 —— 因此日志里
`强制 use_cayley_neumann=False: False` 的含义是"该 flag 不存在、未生效"，
**而不是"用了 Cayley-Neumann 近似"**。0.13.2 只有精确 Cayley 一种实现，
训练与审计参数化天然同源，不存在漂移。将来升级 PEFT 后须复查该 flag 的默认值。

### 诊断 3 的判决口径
- `err_fp64 << err_fp32` (如下降 >100x) ⇒ 差异主要来自 **FP32 有限精度**, 而非实现 bug。
- `err_tf32_on > err_tf32_off` ⇒ **TF32** 也贡献了一部分误差。
- 局部吸收误差在 FP64 趋近**双精度量级** (实测 2.84e-14, 非机器精度 eps≈2.22e-16) ⇒ **支持吸收代数等价** (实数算术下 `y=(xW₀ᵀ)R+b = x(RᵀW₀)ᵀ+b`, 不要求 R 正交)。该界为诊断性上界, 与代数等价定理 + 跨精度缩放共同支持, 不构成"全部误差都来自有限精度"的完整证明。

## 论文描述建议（当前谨慎版）

论文小节名称: **Reproduction & Absorption Verification**

段落要点:
1. 方法复现: 精确 Cayley 参数化 OFT（非 Cayley-Neumann 近似），并诚实标注为受控复现而非第三方已发布检查点审计。
2. 实验设置: 公开预训练模型 (`dinov2-large`, ViT-Large) + 小规模下游任务 (CIFAR-10, num_labels=10)；PEFT `r`=8 为块数 (块大小 128)。
3. 吸收审计: 逐层 `merge_and_unload()` 前后对比；报告 max_err / MSE / 分位数 / 超阈值占比。
4. **可写结论（已验证, 取代下方旧草稿）**: 在当前实验配置与输入下, OFT 吸收在精确算术下严格等价；观测到的 FP32 残差 (干净 TF32-off 1.0e-5、TF32-on 3.0e-3) 与端到端偏差, 与 TF32 及深度累积产生的有限精度效应一致, 不构成对吸收等价性的否定。详见"实测结果快照"。

## 实测结果快照（2026-09-19, 远程 Ubuntu 8×A100, dinov2-large + CIFAR-10 OFT checkpoint）

> 三脚本已实际运行。结论已从小结 "待追查线索" 升级为 "已验证证据"。

- **诊断 3 — 精度消融（决定性）**: 同一条吸收误差在三种配置下 —
  - FP32 + TF32 开: max 2.945e-03, MSE 9.7e-07
  - FP32 + TF32 关: max 1.025e-05, MSE 1.1e-11
  - **FP64（原权重升双精度重算合并）: max 5.129e-14, MSE 4.7e-29**
  - TF32 开比关放大 **287×** (该倍率现已由共享底座+固定输入的严格对照得出, 见下方说明)；FP64 残差落双精度量级 (实测 5.13e-14, 在 Higham FP64 一阶界内, **非**机器精度 eps≈2.22e-16) ⇒ **吸收在精确算术下严格等价** (由代数定理 + 跨精度缩放共同支持)。
- **诊断 2 — 固定输入单层重放**: 该旧脚本因 `merge_and_unload()` 原地修改导致"未合并模块引用失效"，曾误报 96 层 `local_absorb_max` 全部 = 0（`<5e-7`，**已作废**，见下方"仍须诚实标注的边界"）。修复后与 `nonidentity_recovery.py` 一致：FP32 局部吸收 max **2.19e-5**、FP64 **2.84e-14**（均在 Higham 浮点一阶界内）。端到端偏差来自上游输入漂移的逐层累积而非局部吸收不精确——旧脚本的逐层 `e2e_input_max` 数值（layer.0=0 → layer.1=2.7e-5 → layer.10=7.9e-5）来自同一处受损测量，**一并作废**，仅以"新脚本确认局部吸收在界内、端到端差异有限"为准。
- **诊断 1 — 恒等对照（已重跑, 修复版）**: 原版因 `requires_grad=False` 过滤导致 "修改 0 个参数"（恒等置位未生效），其 FAIL 结论已作废。修复后强制 R=I（真正修改 96 个参数），`model(x) ≈ base(x)` 自洽验证 = 0，**merge 前后 max 4.172e-06, MSE 1.308e-13, 超1e-6占比=1.87%, 结论 PASS**（阈值 1e-4）。说明 R=I 下 merge 仅剩 FP32 合并数值噪声、无实现 bug，非平凡 R 的差异才有分析价值。

**论文可写结论（已验证）**: OFT 吸收在精确算术下严格等价；FP64 局部吸收残差达双精度量级 (2.84e-14, 在 Higham FP64 界内; 非机器精度 eps≈2.22e-16)，FP32 局部吸收 2.19e-5 落在 Higham 浮点一阶界内（实测/界 ≈ 0.015）；观测到的 FP32 残差（干净 1e-5、TF32 下 3e-3）与端到端偏差，与 TF32 及深度累积的有限精度效应一致，不构成对吸收等价性的否定。此结论与 LOWO 主张的 grouped 因子吸收（精确算术下等价）同位。

**仍须诚实标注的边界**: 本实验 adapter 自训练于 CIFAR-10，配置声称 3 epoch / 5000 样本，但**无绑定训练日志与 seed 证实**（见 checkpoint 模型卡诚实声明），因此定位为"自行训练 adapter 的实现验证"，而非第三方已发布、含已训练变换的检查点审计。实测有效旋转偏离恒等 ‖R−I‖_F = 0.30–0.50，确证该权重是**非平凡非恒等旋转**（并非弱学习、R 未偏离恒等）。吸收的"等价性"是定理级结论（对任意 R 成立，不要求 R 正交），已由 FP64 机器精度数值确认；此等价性指输出映射恒等，不等于"合并前后任务表现不变"——若将来拿到真实强训练 OFT 检查点，仍须补验合并前后的任务一致性。