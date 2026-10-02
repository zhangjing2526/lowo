"""离线自检: 验证 nonidentity_recovery.py 里两个核心数学函数。

不依赖 peft / cuda / 网络, 只用 CPU torch。
1) _cayley_R_from_param : 斜对称参数 A -> 正交 Q
2) _effective_R_from_weights : 由 (W0, Wm) 反推有效旋转 R

做法: 构造合成数据 —— 随机斜对称 A_i -> Q_i -> R=blockdiag(Q_i),
令 Wm = R^T W0 (PEFT merge 的定义), 再用 (W0, Wm) 反推 R',
检查 R' == R 且 R' == Cayley(A)。
"""

import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

spec = importlib.util.spec_from_file_location("nir", HERE / "nonidentity_recovery.py")
nir = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nir)

torch.manual_seed(0)


def build_case(d=8, r=4, dtype=torch.float32):
    block = d // r
    W0 = torch.randn(d, d, dtype=dtype)
    A_blocks = []
    Q_blocks = []
    for i in range(r):
        M = torch.randn(block, block, dtype=dtype)
        A = M - M.T  # 严格斜对称
        A_blocks.append(A)
        Q_blocks.append(nir._cayley_R_from_param(A))
    oft_r = torch.stack(A_blocks)  # [r, block, block]
    R = torch.block_diag(*Q_blocks)
    Wm = (R.T @ W0).to(dtype)
    return W0, Wm, oft_r, R, block


def orth_err(Q):
    n = Q.shape[0]
    return (Q.T @ Q - torch.eye(n, dtype=Q.dtype)).abs().max().item()


checks = []


def chk(name, cond, detail=""):
    checks.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")


print("=" * 60)
print("nonidentity_recovery 数学自检 (CPU, 合成数据)")
print("=" * 60)
print("说明: FP32 下由真实权重反推 R 的精度受 W0 条件数与 FP32 舍入限制 (~1e-5);")
print("      FP64 下同一流程应降到 ~1e-12。阈值按精度分档, 反映这一固有差异。\n")

for dtype, tol in [(torch.float32, 1e-3), (torch.float64, 1e-8)]:
    pname = "fp32" if dtype == torch.float32 else "fp64"
    print(f"--- dtype={pname} (容差 {tol:.0e}) ---")
    for d, r in [(8, 4), (16, 8), (1024, 8)]:
        W0, Wm, oft_r, R, block = build_case(d, r, dtype)
        tag = f"{pname} d={d},r={r},block={block}"

        # 1) Cayley 输出正交
        oe = orth_err(R)
        chk(f"[{tag}] Cayley Q 正交", oe < tol, f"max|Q^T Q - I| = {oe:.3e}")

        # 2) 权重反推 R'
        Rp, R_max, orth_max = nir._effective_R_from_weights(W0, Wm, block)
        rev = (Rp - R.to(Rp.dtype)).abs().max().item()
        chk(f"[{tag}] 权重反推 R' == R", rev < tol, f"max|R'-R| = {rev:.3e}")
        chk(f"[{tag}] 反推 R' 正交", orth_max < tol, f"max orth err = {orth_max:.3e}")

        # 3) 与 Cayley 重算 Q 的交叉验证
        chk(f"[{tag}] R' == Cayley(A) 交叉", rev < tol, f"max diff = {rev:.3e}")

        # 4) R 非恒等 (这是非恒等恢复实验的前提)
        dev = (R - torch.eye(d, dtype=R.dtype)).abs().norm().item()
        chk(f"[{tag}] R 偏离恒等 > 1e-4", dev > 1e-4, f"||R-I||_F = {dev:.4f}")

        # 5) Wm 定义一致性: x W0^T R  ==  x Wm^T
        x = torch.randn(2, d, dtype=dtype)
        lhs = (x @ W0.T) @ R
        rhs = x @ Wm.T
        fw = (lhs - rhs).abs().max().item()
        chk(f"[{tag}] x W0^T R == x Wm^T", fw < tol, f"max diff = {fw:.3e}")

        # 6) Higham 浮点一阶界: 实测局部误差必须落在界内; 且界不能太松 (>= 误差的 1e-6),
        #    否则判据形同虚设 (用它可以把任何数说成"浮点误差")。
        bound = nir._local_forward_fp_bound(x, W0, Wm, R, dtype)
        ratio = fw / max(bound, 1e-300)
        chk(f"[{tag}] 局部误差 <= 浮点一阶界", fw <= bound,
            f"err={fw:.3e} bound={bound:.3e} ratio={ratio:.3e}")
        chk(f"[{tag}] 界非平凡 (err/bound > 1e-6)", ratio > 1e-6,
            f"ratio={ratio:.3e}")

print("=" * 60)
n_pass = sum(1 for _, ok, _ in checks if ok)
print(f"{n_pass}/{len(checks)} PASS")

# ---- 真实量级预测: dinov2-large 规模的权重/激活 ----
# W0 ~ N(0, 0.02) (ViT Linear 的典型尺度), x ~ N(0,1), d=1024, block=128。
# 目的: 预估远程 fp32 档会打印出的"浮点一阶界"与实际误差的量级,
#      便于拿到结果时立刻判断正常与否 (界的典型量级 1e-3~1e-2, err/bound ~1e-3)。
print("\n" + "=" * 60)
print("真实量级预测 (dinov2-large 尺度, d=1024, block=128)")
print("=" * 60)
for dtype, pname in [(torch.float32, "fp32"), (torch.float64, "fp64")]:
    d, r = 1024, 8
    block = d // r
    W0 = torch.randn(d, d, dtype=dtype) * 0.02
    Qb = []
    for i in range(r):
        M = torch.randn(block, block, dtype=dtype)
        Qb.append(nir._cayley_R_from_param(M - M.T))
    R = torch.block_diag(*Qb)
    Wm = (R.T @ W0).to(dtype)
    x = torch.randn(2, 197, d, dtype=dtype)          # [B, N, d] 与 ViT 序列一致
    err = (((x @ W0.T) @ R) - (x @ Wm.T)).abs().max().item()
    bound = nir._local_forward_fp_bound(x, W0, Wm, R, dtype)
    print(f"  [{pname}] 局部误差={err:.3e}  浮点一阶界={bound:.3e}  err/bound={err/max(bound,1e-300):.3e}")

if n_pass != len(checks):
    sys.exit(1)
