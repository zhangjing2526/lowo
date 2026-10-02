"""Whole-network simultaneous absorption fold -- end-to-end consistency check.

The paper's audit is layer-local: each site is folded one at a time and the
local composite is compared.  A reviewer will (rightly) ask whether the folds
also COMPOSE: fold every removable site of a checkpoint at once and check that
held-out inference is unchanged.  Local folds are exact equalities, so the
chained result can only differ by floating-point accumulation -- this script
measures exactly that residual.

WHAT IT FOLDS
  Every site whose adjacent mixer is DENSE (groups=1):
    * HaarDownsample with per_channel=False and mix.groups == 1    (analysis)
    * HaarUpBlock (the decoder's real upsample, dense 1x1 expand)  (synthesis)
    * HaarUpsample (the standalone synthesis bandlet, same density)
  Sites that are NOT foldable are left untouched and reported:
    * WaveletConv3d (sandwich) -- its internal mixer is groups=8 in every arm
    * HaarDownsample with grouped_mix=True (groups=8)
    * HaarDownsample with per_channel=True (A8's per-channel rung)

  The general fold uses R8 = K @ K_haar^T built from the checkpoint's own
  analysis kernel, so it is correct for BOTH parameterizations:
    so8   mode: K = Q K_haar          -> R8 = Q
    theta mode: K = (prod F) K_haar   -> R8 = product of per-axis rotations
  A dense mixer absorbs ANY constant orthogonal R on the 8C index, so no
  structure of R8 is assumed.

  Analysis  (y = W dwt_K(x)):  z = R8 z_haar
      W'[o,(t,c)] = sum_s W[o,(s,c)] R8[s,t]                (bias unchanged)
  Synthesis (y = idwt_K(W x)): z'[t] = sum_s R8[s,t] z[s]
      W'[(t,c),i] = sum_s R8[s,t] W[(s,c),i],  b' likewise
  After folding, the wavelet is reset to fixed Haar (skew := 0 / theta := pi/4),
  so the folded network contains NO learned transform at all.

USAGE (8xA100 host, from the lowo repo root)
  python fold_all_sites_check.py \
      --config configs/A5_so8_g1.yaml --ckpt <A5 best_model_*.pt> \
      --data_dir ./data/fullres/train --test_list splits/test_250.txt \
      --n_cases 10 --out fold_all_A5.json

  Smoke tests, run these FIRST -- both were verified on the Windows edit host:
  python fold_all_sites_check.py --module-selftest   # real down->sandwich->up
                                                     # chain, no ckpt, no data
  python fold_all_sites_check.py --algebra-only      # pure-algebra identity

  --selftest: full pipeline with random weights + one random 128^3 input
  (needs the repo, monai and a GPU, but no checkpoint and no data).

EXPECTED RESULT
  Per-site fold residual (float64)  ~1e-15   -- asserts < 1e-10
  Chain-level logit deviation (fp32) ~1e-6   -- float32 accumulation noise
  100% voxel argmax agreement, per-case Dice(orig, folded) = 1.0 on every case.
  Anything far larger means a fold bug, not float noise -- do not paste a big
  number into the paper; the number that goes in the paper is the chain-level
  fp32 deviation, reported as "max relative logit deviation X e-6".

SCOPE NOTE (do not over-claim in the paper)
  A5 is g=1, but its eight SANDWICH mixers are g=8 in every arm, so only 8 of
  A5's 16 sites fold (4 downsample + 4 upsample).  The claim is scoped to the
  dense subset of each arm / to dense arms -- never to "all arms".
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys

import numpy as np
import torch

DT = torch.float64


# --------------------------------------------------------------------------- #
# the Haar 8x8 analysis matrix, transcribed from lowo.py::HaarWavelet3D
# --------------------------------------------------------------------------- #
def haar8() -> torch.Tensor:
    """(8, 8) analysis matrix of the FIXED Haar bank (theta = pi/4, no so8).

    Computed directly in float64: in float32 the pi/4 rotation is only
    orthogonal to ~1e-7, which would leak into R8 and swamp the algebraic
    residual this script is trying to measure.
    """
    t = torch.full((3,), math.pi / 4, dtype=DT)
    c, s = torch.cos(t), torch.sin(t)
    f = torch.stack([torch.stack([c, s], dim=-1),
                     torch.stack([-s, c], dim=-1)], dim=-2)  # (3, 2, 2)
    k = torch.einsum('ad,bh,cw->abcdhw', f[0], f[1], f[2]).reshape(8, 8)
    return k.to(DT)


def kernel_matrix(wave) -> torch.Tensor:
    """(8, 8) analysis matrix of `wave`'s CURRENT kernel."""
    with torch.no_grad():
        return wave._kernel().reshape(8, 8).to(DT)


def reset_to_haar(wave) -> None:
    """Freeze `wave` at fixed Haar (no learned factor left in the kernel).

    theta is always re-filled with pi/4, not only when trainable: in so8 mode
    theta starts as a float32 buffer, so after a ``.double()`` copy its VALUE
    would still be the float32 rounding of pi/4 (~1e-8 off), which would show
    up as a fake residual in the float64 check.
    """
    with torch.no_grad():
        if hasattr(wave, "skew"):
            wave.skew.zero_()
            if wave.skew.requires_grad:
                wave.skew.requires_grad_(False)
        if hasattr(wave, "theta"):
            wave.theta.fill_(math.pi / 4)
            if wave.theta.requires_grad:
                wave.theta.requires_grad_(False)
        if hasattr(wave, "_k_cache"):          # cached non-learnable kernel
            wave._k_cache = wave._kernel()


# --------------------------------------------------------------------------- #
# folding
# --------------------------------------------------------------------------- #
def _fold_analysis_weight(W: torch.Tensor, R8: torch.Tensor) -> torch.Tensor:
    """W: (Cout, 8*Cin, 1,1,1) subband-major -> folded weight, same shape."""
    Cout = W.shape[0]
    Wf = torch.einsum('osc,st->otc', W.reshape(Cout, 8, -1).to(DT), R8)
    return Wf.reshape(W.shape).to(W.dtype)


def _fold_synthesis_weight(W: torch.Tensor, R8: torch.Tensor) -> torch.Tensor:
    """W: (8*Cout, Cin, 1,1,1) subband-major -> folded weight, same shape."""
    n8 = W.shape[0] // 8
    Wf = torch.einsum('sci,st->tci', W.reshape(8, n8, -1).to(DT), R8)
    return Wf.reshape(W.shape).to(W.dtype)


def _fold_synthesis_bias(b: torch.Tensor, R8: torch.Tensor) -> torch.Tensor:
    n8 = b.shape[0] // 8
    bf = torch.einsum('sc,st->tc', b.reshape(8, n8).to(DT), R8)
    return bf.reshape(b.shape).to(b.dtype)


def _local_rel_check(mod, role: str) -> float:
    """max |y_folded - y_orig| / max|y_orig| for one site, all in float64.

    The double copy recomputes its own kernel, so R8 is derived from the SAME
    kernel the reference module uses; the residual measured here is therefore
    the algebraic residual of the fold (~1e-15), not dtype noise.  Float32
    deployment error is measured separately by the chain-level comparison.
    """
    with torch.no_grad():
        ref = copy.deepcopy(mod).double()
        folded = copy.deepcopy(mod).double()
        R8 = ref.wave._kernel().reshape(8, 8) @ haar8().to(ref.wave._kernel().device).T
        if role == "downsample":
            folded.mix.weight.copy_(_fold_analysis_weight(folded.mix.weight.data, R8))
            reset_to_haar(folded.wave)
            c_in = folded.mix.in_channels // 8
            x = torch.randn(1, c_in, 16, 16, 16, dtype=DT, device=ref.mix.weight.device)
            y0, y1 = ref(x), folded(x)
        else:
            folded.expand.weight.copy_(_fold_synthesis_weight(folded.expand.weight.data, R8))
            if folded.expand.bias is not None:
                folded.expand.bias.copy_(_fold_synthesis_bias(folded.expand.bias.data, R8))
            reset_to_haar(folded.wave)
            x = torch.randn(1, ref.expand.in_channels, 8, 8, 8, dtype=DT, device=ref.expand.weight.device)
            # HaarUpBlock.forward(inp, skip) takes a skip input, but the fold only
            # concerns the expansion + IDWT part, which is identical for both
            # HaarUpsample and HaarUpBlock.  Invoke that sub-path explicitly.
            y0 = ref.wave.idwt(ref.expand(x))
            y1 = folded.wave.idwt(folded.expand(x))
    return float(((y0 - y1).abs().max() / y0.abs().max()).item())


def fold_dense_sites(model) -> list:
    """Fold every groups=1 site of `model` IN PLACE.  Returns a per-site report."""
    from model_segmamba.lowo import HaarDownsample, HaarUpsample, WaveletConv3d
    from model_segmamba.haar_blocks import HaarUpBlock

    report = []
    kh = haar8().to(next(model.parameters()).device)
    for name, mod in model.named_modules():
        if isinstance(mod, HaarDownsample):
            if mod.per_channel:
                report.append(dict(site=name, role="downsample", folded=False,
                                   reason="per-channel rung"))
                continue
            if mod.mix.groups != 1:
                report.append(dict(site=name, role="downsample", folded=False,
                                   reason=f"groups={mod.mix.groups}"))
                continue
            R8 = kernel_matrix(mod.wave) @ kh.T
            rel = _local_rel_check(mod, "downsample")
            with torch.no_grad():
                mod.mix.weight.copy_(_fold_analysis_weight(mod.mix.weight.data, R8))
                reset_to_haar(mod.wave)
            report.append(dict(site=name, role="downsample", folded=True,
                               R8_drift=float((R8 - torch.eye(8, dtype=DT, device=R8.device)).norm()),
                               rel_check=rel))
        elif isinstance(mod, (HaarUpsample, HaarUpBlock)):
            R8 = kernel_matrix(mod.wave) @ kh.T
            rel = _local_rel_check(mod, "upsample")
            with torch.no_grad():
                mod.expand.weight.copy_(_fold_synthesis_weight(mod.expand.weight.data, R8))
                if mod.expand.bias is not None:
                    mod.expand.bias.copy_(_fold_synthesis_bias(mod.expand.bias.data, R8))
                reset_to_haar(mod.wave)
            report.append(dict(site=name, role="upsample", folded=True,
                               R8_drift=float((R8 - torch.eye(8, dtype=DT, device=R8.device)).norm()),
                               rel_check=rel))
        elif isinstance(mod, WaveletConv3d):
            report.append(dict(site=name, role="sandwich", folded=False,
                               reason="sandwich mixer is groups=8"))
    return report


# --------------------------------------------------------------------------- #
# algebra-only self test (no repo, no checkpoint)
# --------------------------------------------------------------------------- #
def algebra_only() -> None:
    """Random dense mixer + random R: the single-site fold identity, float64.

    Uses the real subband-major layout (index = s*C + c), i.e. R acts as the
    Kronecker product R8 kron I_C, exactly as in the network.
    """
    g = torch.Generator().manual_seed(0)
    R1 = torch.linalg.qr(torch.randn(8, 8, generator=g, dtype=DT))[0]
    C = 8
    Rk = torch.kron(R1.contiguous(), torch.eye(C, dtype=DT))  # (8C, 8C)
    W1 = torch.randn(48, 8 * C, generator=g, dtype=DT)       # dense: 8C -> Cout
    z_haar = torch.randn(8 * C, 4, generator=g, dtype=DT)    # Haar coefficients
    y_orig = W1 @ (Rk @ z_haar)          # learned bank: z = R8 z_haar, then W1
    W1f = torch.einsum('osc,st->otc', W1.reshape(48, 8, C), R1).reshape(48, 8 * C)
    y_folded = W1f @ z_haar              # folded mixer on the FIXED-Haar coeffs
    err = float((y_folded - y_orig).abs().max() / y_orig.abs().max())
    print(f"[algebra] single dense fold (R kron I_C layout), max rel err = {err:.3e}")
    assert err < 1e-12, "algebra check failed"
    print("[algebra] OK")


# --------------------------------------------------------------------------- #
# module-level self test: a real down -> sandwich -> up chain, no checkpoint
# --------------------------------------------------------------------------- #
def module_selftest(device) -> None:
    """Fold the dense sites of a tiny 3-module chain and compare end to end.

    Needs only torch + model_segmamba/lowo.py (no mamba_ssm, no MONAI, no
    data), so it runs on the Windows edit host as well as on the GPU server.
    The sandwich module in the middle is groups=8 and is intentionally NOT
    folded -- exactly the A5 situation (dense subset of a grouped arm).
    """
    import torch.nn as nn
    from model_segmamba.lowo import HaarDownsample, WaveletConv3d
    from model_segmamba.haar_blocks import HaarUpBlock

    class Chain(nn.Module):
        def __init__(self):
            super().__init__()
            self.down = HaarDownsample(4, 32, learnable=True, mode="so8",
                                       grouped_mix=False)
            self.mid = WaveletConv3d(32, learnable=True, mode="so8")
            self.up = HaarUpBlock(32, 4, learnable=True, mode="so8")

        def forward(self, x):
            y = self.mid(self.down(x))
            return self.up(y, x)

    torch.manual_seed(0)
    net = Chain().to(device).eval()
    with torch.no_grad():                       # give every site a real drift
        for m in net.modules():
            if hasattr(m, "skew") and isinstance(getattr(m, "skew"), torch.Tensor):
                m.skew.copy_(torch.randn_like(m.skew) * 0.7)

    x = torch.randn(1, 4, 32, 32, 32, device=device)
    with torch.no_grad():
        y0 = net(x).clone()
    report = fold_dense_sites(net)
    with torch.no_grad():
        y1 = net(x)

    for r in report:
        tag = (f"R8_drift={r['R8_drift']:.3f} rel={r['rel_check']:.2e}"
               if r["folded"] else r.get("reason", ""))
        print(f"   {'FOLD' if r['folded'] else 'skip':4s} {r['site']:12s} {tag}")
    rel = float((y0 - y1).abs().max() / y0.abs().max())
    print(f"[module-selftest] chain-level max rel err = {rel:.3e}")
    bad = [r for r in report if r["folded"] and r["rel_check"] >= 1e-10]
    assert not bad, f"local fold check failed at {[r['site'] for r in bad]}"
    assert rel < 1e-5, "chain-level output changed beyond float32 noise"
    print("[module-selftest] OK (down+up folded, sandwich untouched)")


# --------------------------------------------------------------------------- #
# end-to-end inference comparison
# --------------------------------------------------------------------------- #
def find_repo(explicit: str) -> str:
    """Locate the repo root holding model_segmamba/lowo.py (and, for the full
    data path, run_test_eval.py)."""
    if explicit:
        return explicit
    here = os.path.dirname(os.path.abspath(__file__))
    starts = [os.getcwd(), here,
              os.path.join(here, os.pardir, "lowo"),          # <paper>/lowo
              os.path.join(here, os.pardir, os.pardir, "lowo")]
    for start in starts:
        p = os.path.abspath(start)
        for _ in range(6):
            if os.path.exists(os.path.join(p, "model_segmamba", "lowo.py")):
                return p
            p = os.path.dirname(p)
    raise SystemExit("could not locate the lowo repo (model_segmamba/lowo.py); "
                     "pass --repo")


def dice(a: np.ndarray, b: np.ndarray, c: int) -> float:
    """Single-label Dice between two labelmaps on channel `c` only.

    Kept for call sites that genuinely need one raw sub-region channel.
    For BraTS region scores use `region_dices`, which applies the correct
    union semantics (ET={3}, TC={1,3}, WT={1,2,3}).
    """
    x, y = a == c, b == c
    s = int(x.sum() + y.sum())
    return 1.0 if s == 0 else float(2.0 * (x & y).sum() / s)


def region_dices(gt: np.ndarray, pred: np.ndarray) -> dict:
    """WT/TC/ET Dice between two int8 labelmaps (0..3).

    BraTS regions are unions of sub-region labels, NOT raw channels:
      ET (enhancing tumor) = {3}
      TC (tumor core)       = {1, 3}          necrotic + enhancing
      WT (whole tumor)      = {1, 2, 3}        all three
    Dice of an empty union (both sides empty) is 1.0.

    Order of the two arguments is ignored by Dice, but by convention the
    first is the reference (ground truth) and the second the prediction.
    When used to compare two model predictions, call it the
    prediction-to-prediction Dice between orig_pred and folded_pred.
    """
    unions = {
        "ET": (3,),
        "TC": (1, 3),
        "WT": (1, 2, 3),
    }
    out = {}
    for name, labels in unions.items():
        g = np.isin(gt, labels)
        p = np.isin(pred, labels)
        s = int(g.sum() + p.sum())
        out[f"dice_{name}"] = 1.0 if s == 0 else float(2.0 * (g & p).sum() / s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="", help="required unless --algebra-only")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--repo", default="", help="lowo repo root containing run_test_eval.py")
    ap.add_argument("--data_dir", default="./data/fullres/train")
    ap.add_argument("--test_list", default="splits/test_250.txt")
    ap.add_argument("--n_cases", type=int, default=10)
    ap.add_argument("--overlap", type=float, default=0.7)
    ap.add_argument("--sw_batch", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="fold_all.json")
    ap.add_argument("--selftest", action="store_true",
                    help="random weights + one random input; no ckpt, no data")
    ap.add_argument("--module-selftest", action="store_true",
                    help="tiny down->sandwich->up chain; needs only torch + lowo.py")
    ap.add_argument("--algebra-only", action="store_true")
    args = ap.parse_args()

    if args.algebra_only:
        algebra_only()
        return
    if args.module_selftest:
        sys.path.insert(0, find_repo(args.repo))
        module_selftest(torch.device(args.device if args.device.startswith("cpu")
                                     else "cpu"))
        return
    if not args.config:
        ap.error("--config is required unless --algebra-only / --module-selftest")

    sys.path.insert(0, find_repo(args.repo))
    import yaml
    import run_test_eval as R
    from monai.inferers import SlidingWindowInferer

    device = torch.device(args.device)
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    model = R.build_model_from_config(cfg, device)
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(R.filte_state_dict(sd), strict=True)
        print(f"[ckpt] loaded {os.path.basename(args.ckpt)} (strict)")

    model_f = R.build_model_from_config(cfg, device)
    model_f.load_state_dict(model.state_dict(), strict=True)
    report = fold_dense_sites(model_f)
    n_fold = sum(1 for r in report if r["folded"])
    print(f"[fold] {n_fold} site(s) folded, "
          f"{len(report) - n_fold} left untouched:")
    for r in report:
        tag = (f"R8_drift={r['R8_drift']:.3f} rel={r['rel_check']:.2e}"
               if r["folded"] else r.get("reason", ""))
        print(f"   {'FOLD' if r['folded'] else 'skip':4s} {r['site']:52s} {tag}")
    bad = [r for r in report if r["folded"] and r["rel_check"] >= 1e-10]
    assert not bad, f"local fold check failed at {[r['site'] for r in bad]}"

    inferer = SlidingWindowInferer(roi_size=[128, 128, 128],
                                   sw_batch_size=args.sw_batch,
                                   overlap=args.overlap, mode="gaussian")

    def inputs():
        if args.selftest:
            yield "random", torch.randn(1, 4, 128, 128, 128, device=device)
            return
        names = [l.strip() for l in open(args.test_list, encoding="utf-8")
                 if l.strip() and not l.startswith("#")][:args.n_cases]
        from light_training.dataloading.dataset import MedicalDataset
        ds = MedicalDataset([os.path.join(args.data_dir, n + ".npz") for n in names])
        for i in range(len(ds)):
            item = ds[i]
            yield item["properties"]["name"], \
                torch.from_numpy(np.ascontiguousarray(item["data"])).float()[None].to(device)

    rows = []
    with torch.no_grad():                       # fp32 on purpose: autocast would
        for name, img in inputs():              # mask the accumulation residual
            y0 = inferer(img, model).float()
            y1 = inferer(img, model_f).float()
            a0, a1 = y0.argmax(1)[0].cpu().numpy(), y1.argmax(1)[0].cpu().numpy()
            rel = float((y0 - y1).abs().max() / y0.abs().max())
            # Prediction-to-prediction Dice: orig_pred vs folded_pred, correct
            # BraTS union semantics (TC={1,3}, WT={1,2,3}, ET={3}).
            rd = region_dices(a0, a1)
            row = dict(case=name,
                       max_abs_diff=float((y0 - y1).abs().max()),
                       max_rel_diff=rel,
                       argmax_agreement=float((a0 == a1).mean()),
                       dice_TC=rd["dice_TC"],
                       dice_WT=rd["dice_WT"],
                       dice_ET=rd["dice_ET"])
            rows.append(row)
            print(f"   {name:24s} rel={rel:.3e} agree={row['argmax_agreement']*100:6.2f}% "
                  f"Dice={np.mean([row['dice_TC'], row['dice_WT'], row['dice_ET']]):.4f}")

    summary = dict(config=args.config, ckpt=args.ckpt,
                   n_sites_folded=n_fold, sites=report, cases=rows,
                   max_rel_diff=max(r["max_rel_diff"] for r in rows),
                   min_argmax_agreement=min(r["argmax_agreement"] for r in rows),
                   all_dice_one=all(r["dice_TC"] == 1.0 and r["dice_WT"] == 1.0
                                    and r["dice_ET"] == 1.0 for r in rows))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[done] wrote {args.out}; max rel diff {summary['max_rel_diff']:.3e}; "
          f"argmax agreement >= {summary['min_argmax_agreement'] * 100:.2f}%; "
          f"all Dice == 1.0: {summary['all_dice_one']}")


if __name__ == "__main__":
    main()
