# CIFAR-10 2D LOWO probe (B0 / A2 / A5 / A6)

2D absorption check for ICASSP 2027 LOWO: `SO(4)`, 6 DOF.  CIFAR ResNet-18
`conv1` is 3×3 stride-1, so there is **no stem DWT**.  Haar sites are
layer2/3/4 × (main-path `conv1` + skip) = **6 sites**.

| 臂 | 下采样 | 对应正文 |
|---|---|---|
| B0 | stride-2 conv | 可学习下采样 |
| A2 | 固定 Haar，`groups=4` | 2D HWD |
| A5 | 可学习 \(Q\in SO(4)\)，`groups=1` | 吸收臂：大漂移、ρ=0 |
| A6 | 可学习 \(Q\in SO(4)\)，`groups=4` | 可辨识的 6 DOF |
| HWD | 官方 HWD mixer：固定 Haar + dense \(1\times1\) | Xu 2023 `Down_wt` |
| HWDQ | 官方 HWD mixer + 可学习 \(SO(4)\) | 与 A5 同类；给第三方 mixer 命名 |

`wave.skew` 不参与 weight decay。指标：CIFAR-10 top-1，train 划 5k 做 val，
**按 best val 报 official test**。种子 123 / 456 / 789。共 **12** 个作业。
配对用 `mcnemar.py`（不是准确率上的 t 检验）。

## 机器 1（8 卡）— 种子 123+456

```bash
python train_cifar.py --selftest
bash run_8gpu.sh
```

Windows（PS 5.1）：`.\run_8gpu.ps1`

## 机器 2 — 种子 789，或补当前缺口

若机器 1 **已经**按旧脚本在跑 B0/A2/A6，不要重开那 8 个，只补：

```bash
# 机器 2：A5×3 + A6 seed 789
bash run_remaining.sh 0 1 2 3
```

干净 12 作业的后 4 个（B0/A2/A5/A6 × 789）：

```bash
bash run_rest.sh 0 1 2 3
```

## 单卡

```bash
python train_cifar.py --arm A5 --seed 123 --device cuda:0
```

## 跑完后

```bash
python summarize.py
python verify_absorption_2d.py runs/a5_s123/best.pt   # ρ=0, drift>0
python verify_absorption_2d.py runs/a6_s123/best.pt   # ρ>0
python verify_absorption_2d.py runs/a2_s123/best.pt   # ρ=0, Q=I
python audit_hwd_official.py                         # 官方 HWD mixer class
python audit_hwd_official.py --ckpt runs/a5_s123/best.pt
python mcnemar.py --all --runs runs
# 旧 best.pt 没有逐样本预测时：
python mcnemar.py --refresh runs/a2_s123/best.pt --device cuda:0
```

日志在 `runs/<arm>_s<seed>/`。CIFAR 缓存在 `data/`（约 170 MB）。
已在跑的 A6 123/456 若在改 skew-wd 之前启动，Q 被 weight decay 拉向 I；
要干净的 A2–A6 对比，那两臂用新脚本重跑。
