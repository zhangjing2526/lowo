#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LOWO 子带幅频响应测量 —— 生成 ICASSP 论文 Fig.1。

为什么需要这个脚本
------------------
仓库里现有的 measure_*.py 全是"能量 / 耗时 / locality"分析，没有任何一个做
幅频响应。而论文的关键卖点是"学到的 Q 把 8 个子带的通带推向了哪里"，
必须有一张 |H(omega)| 图。

数学
----
lowo.py 里 _kernel() 返回 (8, 1, 2, 2, 2)：
    K[b_D*4 + b_H*2 + b_W] = f_D[b_D] (x) f_H[b_H] (x) f_W[b_W]   (张量积)
即第 k 行就是第 k 个子带的 2x2x2 冲激响应 h_k[n]。对 h_k 做 3D FFT 即得
该子带的频率响应 H_k(omega_D, omega_H, omega_W)。

因为只有 2 个抽头，直接对 (2,2,2) 做 FFT 只能拿到 2 个采样点（DC 和 Nyquist）。
要看清通带形状必须零填充：zero-pad 到 (N,N,N) 再做 FFT，等价于对连续的
DTFT 采样。

输出
----
1) 主图: 沿主对角线 omega_D = omega_H = omega_W = omega 的 |H| 曲线，8 个子带
   8 条线。这条对角线最能体现子带的"低通/高通"属性：LLL 从 DC 单调下降，
   HHH 从 DC 单调上升。
2) 副图: 固定 Haar vs 学到的 so8 的 LLL / HHH 对比，看通带被推了多少。
3) 控制台: 逐子带的 DC 增益、Nyquist 增益、||Q-I||_F、theta 偏离 pi/4 的数值。

用法
----
  # 固定 Haar（基线，不需要任何权重）
  python measure_subband_freqresp.py --out figs/subband_freqresp.pdf

  # 论文 Fig. 2：A6 stem，Haar（灰）叠 learned SO(8)（彩），矢量 PDF
  python measure_subband_freqresp.py \
      --ckpt lowo/checkpoints/a6_so8_g8_audit.pt \
      --key downsample_layers.0.0 \
      --layout oneline \
      --out figs/subband_freqresp_so8_vs_haar_oneline.pdf

注意：lowo.py 只 import torch，不依赖 mamba_ssm / MONAI，所以本脚本可以直接
按文件路径加载它，不需要构造整个 SegMamba 模型。
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path

import torch

REPO = Path(r"G:\material\天枢\mamba\lowo")
LOWO_PATH = REPO / "model_segmamba" / "lowo.py"
DEFAULT_CKPT = REPO / "checkpoints" / "a6_so8_g8_audit.pt"
# ICASSP spconf \textwidth = 178 mm.  Fig. 2 is a figure* at that width.
PAPER_TEXTWIDTH_IN = 178.0 / 25.4
PAPER_FIG2_HEIGHT_IN = 1.48

SUBBAND_NAMES = ["LLL", "LLH", "LHL", "LHH",
                 "HLL", "HLH", "HHL", "HHH"]


# --------------------------------------------------------------------------
# 加载 lowo.py（按文件路径，避开 mamba_ssm / MONAI 依赖）
# --------------------------------------------------------------------------
def load_lowo_module():
    if not LOWO_PATH.exists():
        sys.exit(f"[fatal] 找不到 {LOWO_PATH}")
    spec = importlib.util.spec_from_file_location("lowo_standalone", LOWO_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# 从 ckpt 里挖出 wavelet 参数
# --------------------------------------------------------------------------
def harvest_wavelet_params(ckpt_path):
    """返回 list of dict: {key, mode, theta, skew}

    ckpt 的 state_dict 里 wavelet 相关的键形如
        *.wave.theta      (3,)      theta 模式
        *.wave.skew       (8, 8)    so8 模式
    不同框架（lightning / 自研 trainer）包法不同，这里做保守的多层展开。
    """
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "network", "model_state_dict"):
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break
    if not isinstance(obj, dict):
        return []

    found = {}
    for k, v in obj.items():
        if not torch.is_tensor(v):
            continue
        low = k.lower()
        if low.endswith(".theta") or low == "theta":
            if v.numel() == 3:
                found.setdefault(k.replace(".theta", ""), {})["theta"] = v.detach().float()
        if low.endswith(".skew") or low == "skew":
            if v.shape[-2:] == (8, 8):
                found.setdefault(k.replace(".skew", ""), {})["skew"] = v.detach().float()

    out = []
    for stem, d in sorted(found.items()):
        mode = "so8" if "skew" in d else "theta"
        out.append({"key": stem, "mode": mode,
                    "theta": d.get("theta"), "skew": d.get("skew")})
    return out


# --------------------------------------------------------------------------
# 由参数构造 (8, 8) 分析核
# --------------------------------------------------------------------------
def build_kernel(lowo, mode, theta=None, skew=None):
    """复刻 lowo.py:73-85 的 _kernel()，返回 (8, 8) 矩阵（去掉 conv 的 1 通道维）。"""
    if theta is None:
        theta = torch.full((3,), math.pi / 4)
    c, s = torch.cos(theta), torch.sin(theta)
    f = torch.stack([torch.stack([c, s], dim=-1),
                     torch.stack([-s, c], dim=-1)], dim=-2)          # (3, 2, 2)
    k = torch.einsum("ad,bh,cw->abcdhw", f[0], f[1], f[2]).reshape(8, 8)
    if mode == "so8" and skew is not None:
        q = torch.matrix_exp(skew - skew.transpose(0, 1))            # in SO(8)
        k = q @ k
    return k


def ortho_dev(skew):
    """||Q - I||_F，与 lowo.py:87-93 一致。"""
    if skew is None:
        return 0.0
    q = torch.matrix_exp(skew - skew.transpose(0, 1))
    return (q - torch.eye(8, dtype=q.dtype)).norm().item()


# --------------------------------------------------------------------------
# 幅频响应
# --------------------------------------------------------------------------
def freq_response_3d(k8, n=64):
    """k8: (8, 8) 分析核 -> H: (8, n, n, n) 复数频率响应。

    第 k 行 reshape 成 (2,2,2) 冲激响应，零填充到 (n,n,n) 后做 3D FFT。
    注意 fft 不做 shift，索引 i 对应 omega = 2*pi*i/n。
    """
    h = k8.reshape(8, 2, 2, 2)                       # 8 个子带的 2x2x2 冲激响应
    pad = torch.zeros(8, n, n, n, dtype=torch.float64)
    pad[:, :2, :2, :2] = h.double()
    H = torch.fft.fftn(pad, dim=(1, 2, 3))
    return H                                          # (8, n, n, n) complex


def diag_response(k8, n=256):
    """沿主对角线 omega_D = omega_H = omega_W 的响应。

    用 DTFT 直接算（而不是从 3D FFT 里插值），分辨率可以任意高：
        H_k(omega) = sum_n h_k[n] exp(-j omega (n_D + n_H + n_W))
    """
    # k8 的第 k 行 reshape 成 (2,2,2) 就是第 k 个子带的冲激响应，
    # 展平顺序 d*4+h*2+w 与下面的 idx 完全一致，所以直接用 (8,8) 即可。
    h = k8.reshape(8, 8).to(torch.complex128)                      # (subband, tap)
    idx = torch.tensor([[d, r, c] for d in range(2)
                        for r in range(2)
                        for c in range(2)], dtype=torch.float64)   # (8, 3)
    omega = torch.linspace(0.0, math.pi, n, dtype=torch.float64)   # 0 .. Nyquist
    phase = omega[:, None] * idx.sum(dim=1)[None, :]               # (n, 8)
    E = torch.exp(-1j * phase)                                     # (n, 8) complex128
    H = E @ h.transpose(0, 1)                                      # (n, 8)
    return omega, H.transpose(0, 1)                                # (8, n)


def db(x, eps=1e-12):
    return 20.0 * torch.log10(x.abs().clamp_min(eps))


# --------------------------------------------------------------------------
# 控制台数值表
# --------------------------------------------------------------------------
def report(tag, k8, skew, theta):
    print(f"\n=== {tag} ===")
    if skew is not None:
        print(f"  ||Q - I||_F (ortho_dev) = {ortho_dev(skew):.4f}")
    if theta is not None:
        dev = (theta - math.pi / 4).abs() * 180.0 / math.pi
        print("  theta (deg) = "
              + ", ".join(f"{t * 180 / math.pi:.2f}" for t in theta.tolist())
              + f"   |dev from 45deg| = {dev.mean().item():.2f} deg (mean)")

    _, H = diag_response(k8, n=2)          # 只要 DC(0) 和 Nyquist(pi)
    dc = H[:, 0].abs()
    ny = H[:, 1].abs()
    print(f"  {'subband':<8}{'|H(0)|':>10}{'dB':>9}{'|H(pi)|':>10}{'dB':>9}")
    for i, name in enumerate(SUBBAND_NAMES):
        print(f"  {name:<8}{dc[i].item():>10.4f}{db(dc[i]).item():>9.2f}"
              f"{ny[i].item():>10.4f}{db(ny[i]).item():>9.2f}")

    orth_err = (k8 @ k8.T - torch.eye(8, dtype=torch.float64)).abs().max().item()
    print(f"  max |K K^T - I| = {orth_err:.3e}   (应为 ~0，验证结构正交)")


# --------------------------------------------------------------------------
# 画图
# --------------------------------------------------------------------------
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    # Match Fig.3 (drift_vs_rho, generated by replot_rho_scatter.py): serif
    # family + same base size, so the two paper figures share axis style/size.
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = [
        "Times New Roman", "DejaVu Serif", "Liberation Serif", "serif"]
    matplotlib.rcParams["font.size"] = 9.0
    matplotlib.rcParams["axes.linewidth"] = 0.6
    return plt


def _save(fig, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    # bbox_inches="tight" is REQUIRED: without it the canvas edge is a hard
    # crop and any rotated y-label wider than subplots_adjust(left=...) gets
    # sliced in half (the "$|H|$ (dB)" label was cut exactly this way).
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.06)
    print(f"\n[saved] {out_path}")
    # PNG sidecar for visual checks when the primary output is a vector PDF.
    if str(out_path).lower().endswith(".pdf"):
        png = str(out_path)[:-4] + ".png"
        fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.06)
        print(f"[saved] {png}")


def make_figure(curves, out_path):
    """curves: list of (tag, k8) —— 每个 tag 画一行 8 个子带。"""
    plt = _mpl()
    n_rows = len(curves)
    fig, axes = plt.subplots(n_rows, 8, figsize=(22, 3.0 * n_rows), squeeze=False)
    colors = plt.cm.tab10.colors

    for r, (tag, k8) in enumerate(curves):
        omega, H = diag_response(k8, n=256)
        w = (omega / math.pi).numpy()          # 归一化频率 0..1 (1 = Nyquist)
        for c in range(8):
            ax = axes[r][c]
            ax.plot(w, db(H[c]).numpy(), color=colors[c % 10], lw=1.6)
            ax.set_title(SUBBAND_NAMES[c], fontsize=11)
            ax.grid(alpha=0.3, lw=0.5)
            ax.set_xlim(0, 1)
            ax.set_ylim(-45, 15)
            if c == 0:
                ax.set_ylabel(f"{tag}\n|H| (dB)", fontsize=10)
            if r == n_rows - 1:
                ax.set_xlabel(r"$\omega/\pi$", fontsize=10)
            if r == 0:
                ax.set_title(SUBBAND_NAMES[c], fontsize=11)
    fig.suptitle("LOWO subband magnitude response along the main diagonal "
                 r"$\omega_D=\omega_H=\omega_W$", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out_path)
    plt.close(fig)


def make_oneline_figure(k_haar, k_learned, out_path,
                        width_in=PAPER_TEXTWIDTH_IN,
                        height_in=1.72):
    """Paper freqresp: one row, Haar (gray) under learned SO(8) (color).

    Sized to the ICASSP text width so 8 pt labels stay 8 pt on the page
    when included at ``width=\\textwidth`` in a ``figure*``.  Legend sits
    above the panels (not inside HHH).
    """
    plt = _mpl()
    colors = plt.cm.tab10.colors
    omega, H_h = diag_response(k_haar, n=256)
    _, H_l = diag_response(k_learned, n=256)
    w = (omega / math.pi).numpy()

    fig, axes = plt.subplots(1, 8, figsize=(width_in, height_in),
                             sharey=False)
    h_haar = h_so8 = None
    for c, ax in enumerate(axes):
        h_haar, = ax.plot(w, db(H_h[c]).numpy(), color="0.55", lw=1.35,
                          zorder=1)
        h_so8, = ax.plot(w, db(H_l[c]).numpy(), color=colors[c % 10],
                         lw=1.85, zorder=2)
        ax.set_title(SUBBAND_NAMES[c], fontsize=9, pad=2.0)
        ax.grid(alpha=0.3, lw=0.4)
        ax.set_xlim(0, 1)
        ax.set_ylim(-45, 15)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([-40, -20, 0, 10])
        ax.tick_params(axis="both", labelsize=7, length=2.2, pad=1.2)
        ax.set_xlabel(r"$\omega/\pi$", fontsize=8, labelpad=1.2)
        if c == 0:
            ax.set_ylabel(r"$|H|$ (dB)", fontsize=8, labelpad=2)
        else:
            ax.set_yticklabels([])
    fig.legend(
        [h_haar, plt.Line2D([0], [0], color="0.15", lw=1.85)],
        ["fixed Haar", r"learned $SO(8)$"],
        loc="upper center", ncol=2, fontsize=8, frameon=False,
        handlelength=1.8, handletextpad=0.4, columnspacing=1.4,
        bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(left=0.048, right=0.995, top=0.78, bottom=0.22,
                        wspace=0.14)
    _save(fig, out_path)
    plt.close(fig)


def make_grid_figure(k_haar, k_learned, out_path,
                     width_in=3.386, height_in=2.70, ncols=4):
    """Paper freqresp, single-column fit: 2 rows x 4 cols of panels.

    Printed at ``width=\\columnwidth`` in a one-column ``figure``.  The grid is
    full column width, so the point sizes below ARE (to within 1%) the printed
    sizes: they are set to 9.0 pt, which prints at ~9.6 pt (scale =
    columnwidth / width_in = 3.386 / 3.35 = 1.011) and clears the ICASSP
    "no smaller than nine point" minimum.  Shared y axis, tick labels only on
    the outer panels.
    """
    plt = _mpl()
    colors = plt.cm.tab10.colors
    omega, H_h = diag_response(k_haar, n=256)
    _, H_l = diag_response(k_learned, n=256)
    w = (omega / math.pi).numpy()

    nrows = math.ceil(8 / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width_in, height_in),
                             sharex=True, sharey=True, squeeze=False)
    h_haar = h_so8 = None
    for c in range(8):
        r, col = divmod(c, ncols)
        ax = axes[r][col]
        h_haar, = ax.plot(w, db(H_h[c]).numpy(), color="0.55", lw=1.1,
                          zorder=1)
        h_so8, = ax.plot(w, db(H_l[c]).numpy(), color=colors[c % 10],
                         lw=1.5, zorder=2)
        ax.set_title(SUBBAND_NAMES[c], fontsize=9.0, pad=1.8)
        ax.grid(alpha=0.3, lw=0.35)
        ax.set_xlim(0, 1)
        ax.set_ylim(-45, 15)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_yticks([-40, -30, -20, -10, 0, 10])
        ax.tick_params(axis="both", labelsize=9.0, length=2.2, pad=1.2)
        if r == nrows - 1:
            ax.set_xlabel(r"$\omega/\pi$", fontsize=9.0, labelpad=1.2)
        if col == 0:
            ax.set_ylabel(r"$|H|$ (dB)", fontsize=9.0, labelpad=2.5)
    fig.legend(
        [h_haar, plt.Line2D([0], [0], color="0.15", lw=1.5)],
        ["fixed Haar", r"learned $SO(8)$"],
        loc="upper center", ncol=2, fontsize=9.0, frameon=False,
        handlelength=1.3, handletextpad=0.35, columnspacing=1.0,
        bbox_to_anchor=(0.5, 1.04))
    fig.subplots_adjust(left=0.165, right=0.995, top=0.86, bottom=0.17,
                        wspace=0.14, hspace=0.55)
    _save(fig, out_path)
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="训练好的权重；不给则只画固定 Haar 基线")
    ap.add_argument("--max-banks", type=int, default=1,
                    help="ckpt 里可能有多个 wavelet 实例（每个 HaarDownsample 一个），"
                         "最多画几个，默认只画第一个")
    ap.add_argument("--key", default="",
                    help="只保留 key 含该子串的 wavelet 实例"
                         "（论文 Fig.2 用 downsample_layers.0.0 = A6 stem）")
    ap.add_argument("--layout", choices=("rows", "oneline", "grid"),
                    default="rows",
                    help="rows: 每个核一行；oneline: Haar+learned 叠在一行"
                         "（figure* 全宽，论文 Fig.2 legacy）；"
                         "grid: 2x4 单栏版（论文 Fig.2 现用）")
    ap.add_argument("--out", default="figs/subband_freqresp.pdf")
    args = ap.parse_args()

    if not LOWO_PATH.exists():
        print(f"[warn] {LOWO_PATH} 不存在，跳过导入检查")
    else:
        load_lowo_module()

    curves = []

    # 1) 固定 Haar 基线
    k_haar = build_kernel(None, "theta", theta=torch.full((3,), math.pi / 4))
    report("fixed Haar (theta = pi/4, Q = I)", k_haar.double(), None, None)
    curves.append(("fixed Haar", k_haar.double()))

    # 2) 学到的核
    if args.ckpt:
        banks = harvest_wavelet_params(args.ckpt)
        if args.key:
            banks = [b for b in banks if args.key in b["key"]]
        if not banks:
            print(f"\n[warn] {args.ckpt} 里没找到匹配的 theta/skew 参数，只输出固定 Haar 基线")
        for b in banks[: args.max_banks]:
            tag = f"learned {b['mode']}"
            k = build_kernel(None, b["mode"], theta=b["theta"], skew=b["skew"])
            report(f"{tag}  [{b['key']}]", k.double(), b["skew"], b["theta"])
            curves.append((tag, k.double()))

    if args.layout in ("oneline", "grid"):
        learned = [c for c in curves if c[0] != "fixed Haar"]
        if not learned:
            sys.exit(f"[fatal] --layout {args.layout} 需要 ckpt 里至少一个学到的核")
        if args.layout == "oneline":
            make_oneline_figure(k_haar.double(), learned[0][1], args.out)
        else:
            make_grid_figure(k_haar.double(), learned[0][1], args.out)
    else:
        make_figure(curves, args.out)


if __name__ == "__main__":
    main()
