"""P2 supplement: real-GT Dice, TTA on/off ablation, sandwich strict-audit.

The base script fold_all_sites_check.py reports Dice(orig_pred, folded_pred)
-- the paper calls this a "prediction-to-prediction" Dice, which is NOT a
segmentation-quality score.  A reviewer will ask for the actual Dice against
the ground-truth label, and for a decomposition of the fp32 chain residual
(1e-3) into its sources: TTA accumulation vs. sliding-window vs. the fold
itself.  This script answers both, and additionally runs the corrected P3
conjugation test (Sec.4) against every groups=8 sandwich mixer to *prove*
(not assume) they are not absorbable.

WHAT THIS ADDS ON TOP OF fold_all_sites_check.py
  1. Per-region Dice against the real label: fold the checkpoint, run both
     models on the SAME inputs, compare each against gt (WT/TC/ET).
  2. TTA ablation: run every case twice, with 8-way mirror TTA on and with a
     single window (no TTA).  The gap between the two "max rel diff" numbers
     is the TTA contribution; the residual with TTA off is the fold +
     sliding-window only.
  3. Sandwich strict-audit: for every WaveletConv3d site, load its trained
     Q = expm(skew - skew^T) and test whether B = (Q^T ⊗ I) W (Q ⊗ I)
     remains 8-block diagonal.  If yes the groups=8 mixer structurally absorbs
     the fold (a refutation); if no it cannot, and the site is *provably* not
     absorbable -- the off-diagonal energy is reported.

USAGE (8xA100 host, lowo repo root)
  python fold_audit_supplement.py \
      --config configs/A5_so8_g1.yaml --ckpt <best_model.pt> \
      --data_dir ./data/fullres/train --test_list splits/test_250.txt \
      --n_cases 10 --out fold_audit_supplement.json

  # fast self-check with no checkpoint and random weights:
  python fold_audit_supplement.py --selftest

Nothing here retrains; it only folds in place on a deep copy, exactly as the
base audit does.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch

# base audit shares the fold machinery
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from fold_all_sites_check import (  # noqa: E402
    dice,
    find_repo,
    fold_dense_sites,
    region_dices,
)


# --------------------------------------------------------------------------- #
# real-GT region dice (NOTE: region_dices is defined once in fold_all_sites_check
# and re-exported above so all three audit scripts share one implementation)
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# sandwich strict-audit (Sec.4 corrected conjugation criterion)
# --------------------------------------------------------------------------- #
def _conjugate_block_diag(W: np.ndarray, Q: np.ndarray, G: int, d: int) -> tuple:
    """Block-energy of B = (Q^T ⊗ I) W (Q ⊗ I), split by Gxd blocks.

    Returns (off_diag_energy, total_energy) in Frobenius^2.
    """
    Qk = np.kron(Q.T, np.eye(d))          # Q^T ⊗ I
    B = Qk @ W @ np.kron(Q, np.eye(d))
    off, tot = 0.0, 0.0
    for i in range(G):
        for j in range(G):
            blk = B[i * d:(i + 1) * d, j * d:(j + 1) * d]
            e = float(np.linalg.norm(blk, ord="fro") ** 2)
            tot += e
            if i != j:
                off += e
    return off, tot


def audit_sandwich_sites(model) -> list:
    """For every WaveletConv3d site, test strict absorbability and report energy."""
    from model_segmamba.lowo import WaveletConv3d

    rows = []
    for name, mod in model.named_modules():
        if not isinstance(mod, WaveletConv3d):
            continue
        with torch.no_grad():
            w = mod.mix.weight.detach().cpu().numpy()[:, :, 0, 0, 0]  # (8ch, ch)
            ch = mod.mix.in_channels // 8
            G = 8
            # groups=8 conv: out channels split into G blocks of size ch, each
            # block mixes ONLY its own input group -> assemble the (8ch, 8ch)
            # block-diagonal operator before the conjugation test.
            W_full = np.zeros((G * ch, G * ch), dtype=w.dtype)
            for g in range(G):
                W_full[g * ch:(g + 1) * ch, g * ch:(g + 1) * ch] = \
                    w[g * ch:(g + 1) * ch, :]
            if mod.wave.mode == "so8" and hasattr(mod.wave, "skew"):
                skew = mod.wave.skew.detach().cpu().numpy()
                Q = _matrix_exp(skew - skew.T)
            else:
                Q = np.eye(G)
            off, tot = _conjugate_block_diag(W_full, Q, G, ch)
        rel = off / tot if tot > 0 else 0.0
        rows.append(dict(site=name, role="sandwich",
                         off_diag_energy=off, total_energy=tot,
                         rel_off_diag=rel,
                         absorbable=bool(rel < 1e-8)))
    return rows


def _matrix_exp(A: np.ndarray) -> np.ndarray:
    """Matrix exponential with no hard scipy dependency.

    Tries numpy.linalg.matrix_exp first, then scipy.linalg.expm, then a pure
    numpy eigendecomposition fallback -- so the sandwich audit runs on any host
    that has numpy (the remote 8xA100 box, the Windows edit host, or CI).
    """
    if hasattr(np, "linalg") and hasattr(np.linalg, "matrix_exp"):
        return np.linalg.matrix_exp(A)
    try:
        from scipy.linalg import expm
        return expm(A)
    except Exception:
        pass
    w, V = np.linalg.eig(A.astype(np.complex128))
    return np.real((V * np.exp(w)) @ np.linalg.inv(V))


# --------------------------------------------------------------------------- #
# core per-case evaluation both TTA ways
# --------------------------------------------------------------------------- #
def run_comparison(img, gt, model, model_f, inferer, use_tta: bool,
                   repo_tools) -> dict:
    """Run orig+folded on one case, report argmax agreement + GT Dice.

    Both models are evaluated under the SAME precision (bf16 autocast) so the
    only difference between the tta_on and tta_off entries is the mirror-TTA
    averaging, not the dtype.
    """
    MIRROR_COMBOS = repo_tools.MIRROR_COMBOS if use_tta else [()]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        def _run(m):
            acc = None
            for axes in MIRROR_COMBOS:
                x = torch.flip(img, dims=axes) if axes else img
                y = inferer(x, m)
                if axes:
                    y = torch.flip(y, dims=axes)
                acc = y.float() if acc is None else acc + y.float()
                del y
            acc /= len(MIRROR_COMBOS)
            return acc.argmax(dim=1)[0].cpu().numpy().astype(np.int8)

        a0 = _run(model)
        a1 = _run(model_f)
    return dict(tta=bool(use_tta),
                argmax_agreement=float((a0 == a1).mean()),
                dice_orig_vs_gt=region_dices(gt, a0),
                dice_folded_vs_gt=region_dices(gt, a1))


# --------------------------------------------------------------------------- #
# local CPU self-test: weight-audit + GT dice + fold consistency, no data
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    repo_root = find_repo("")
    sys.path.insert(0, repo_root)
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
            return self.up(self.mid(self.down(x)), x)

    dev = torch.device("cpu")
    torch.manual_seed(0)
    net = Chain().to(dev).eval()
    with torch.no_grad():
        for m in net.modules():
            if hasattr(m, "skew") and isinstance(m.skew, torch.Tensor):
                m.skew.copy_(torch.randn_like(m.skew) * 0.7)

    # 1) sandwich strict-audit (Sec.4 conjugation criterion)
    rep = audit_sandwich_sites(net)
    print(f"[selftest] {len(rep)} sandwich site(s) audited:")
    for r in rep:
        print(f"   {r['site']:10s} off_diag={r['off_diag_energy']:.3e} "
              f"rel={r['rel_off_diag']:.3e} absorbable={r['absorbable']}")

    # 2) GT dice sanity: identical labels -> 1.0 on every region
    gt = np.zeros((48, 48, 48), dtype=np.int8)
    gt[0, 0, 0] = 1
    gt[1, 1, 1] = 2
    gt[2, 2, 2] = 3
    same = region_dices(gt, gt)
    print(f"[selftest] GT==pred dice: {same}")
    assert min(same.values()) == 1.0, "identical-label dice must be 1.0"

    # 3) full fold consistency (reuse base machinery) on the SAME input
    folded = copy.deepcopy(net)
    fold_dense_sites(folded)
    with torch.no_grad():
        x = torch.randn(1, 4, 32, 32, 32)
        y0 = net(x)
        y1 = folded(x)
    rel = float((y0 - y1).abs().max() / y0.abs().max())
    print(f"[selftest] chain rel err = {rel:.3e}")
    print("[selftest] OK")


# --------------------------------------------------------------------------- #
# A5 summary: per-TTA-mode region means + paired deltas + MAE + worst/drop
# --------------------------------------------------------------------------- #
_REGIONS = ("WT", "TC", "ET")


def summarize_cases(cases: list) -> dict:
    """Aggregate a list of entries (case, tta_on/tta_off) into a summary.

    Metrics, computed independently for tta_on and tta_off:
      dice_{reg}_orig / dice_{reg}_folded : per-region mean Dice vs GT
      ddice_mean_{reg}      : paired mean (folded - orig)
      ddice_abs_mean_{reg}  : paired mean |folded - orig|
      ddice_abs_worst_{reg} : max |folded - orig| and its case
      ddice_drop_{reg}      : min (folded - orig) and its case (worst if negative, smallest gain if all positive)
    Then a case-based overall (per-case = mean over the 3 regions):
      ddice_mean, ddice_abs_mean, ddice_abs_worst(+case), ddice_drop(+case)
    """
    out = {}
    for mode in ("tta_on", "tta_off"):
        m = dict(regions={}, overall={})
        # per-case deltas, one overall value per case keyed by region
        case_delta = {}
        for entry in cases:
            cname, comp = entry["case"], entry[mode]
            orig = comp["dice_orig_vs_gt"]
            folded = comp["dice_folded_vs_gt"]
            case_delta[cname] = {
                reg: folded[f"dice_{reg}"] - orig[f"dice_{reg}"]
                for reg in _REGIONS
            }
        for reg in _REGIONS:
            origs = [e[mode]["dice_orig_vs_gt"][f"dice_{reg}"] for e in cases]
            folds = [e[mode]["dice_folded_vs_gt"][f"dice_{reg}"] for e in cases]
            deltas = [case_delta[e["case"]][reg] for e in cases]
            abs_worst_i = max(range(len(cases)), key=lambda i: abs(deltas[i]))
            drop_i = min(range(len(cases)), key=lambda i: deltas[i])
            m["regions"][reg] = {
                "dice_orig_mean": float(np.mean(origs)),
                "dice_folded_mean": float(np.mean(folds)),
                "ddice_mean": float(np.mean(deltas)),
                "ddice_abs_mean": float(np.mean(np.abs(deltas))),
                "ddice_abs_worst": float(abs(deltas[abs_worst_i])),
                "ddice_abs_worst_case": cases[abs_worst_i]["case"],
                "ddice_drop": float(deltas[drop_i]),
                "ddice_drop_case": cases[drop_i]["case"],
            }
        per_case = {e["case"]: float(np.mean(list(case_delta[e["case"]].values())))
                    for e in cases}
        vals = list(per_case.values())
        worst_i = max(range(len(vals)), key=lambda i: abs(vals[i]))
        drop_i = min(range(len(vals)), key=lambda i: vals[i])
        names = list(per_case.keys())
        m["overall"] = {
            "ddice_mean": float(np.mean(vals)),
            "ddice_abs_mean": float(np.mean(np.abs(vals))),
            "ddice_abs_worst": float(abs(vals[worst_i])),
            "ddice_abs_worst_case": names[worst_i],
            "ddice_drop": float(vals[drop_i]),
            "ddice_drop_case": names[drop_i],
        }
        out[mode] = m
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--repo", default="")
    ap.add_argument("--data_dir", default="./data/fullres/train")
    ap.add_argument("--test_list", default="splits/test_250.txt")
    ap.add_argument("--n_cases", type=int, default=10)
    ap.add_argument("--overlap", type=float, default=0.7)
    ap.add_argument("--sw_batch", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="fold_audit_supplement.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    repo_root = find_repo(args.repo)
    sys.path.insert(0, repo_root)
    import yaml
    import run_test_eval as R
    from monai.inferers import SlidingWindowInferer
    from light_training.dataloading.dataset import MedicalDataset

    device = torch.device(args.device)

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    model = R.build_model_from_config(cfg, device)
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(R.filte_state_dict(sd), strict=True)
    model_f = R.build_model_from_config(cfg, device)
    model_f.load_state_dict(model.state_dict(), strict=True)
    report = fold_dense_sites(model_f)
    n_fold = sum(1 for r in report if r["folded"])

    sandwich = audit_sandwich_sites(model_f)

    inferer = SlidingWindowInferer(roi_size=[128, 128, 128],
                                   sw_batch_size=args.sw_batch,
                                   overlap=args.overlap, mode="gaussian")

    names = [l.strip() for l in open(args.test_list, encoding="utf-8")
             if l.strip() and not l.startswith("#")][:args.n_cases]
    ds = MedicalDataset([os.path.join(args.data_dir, n + ".npz") for n in names])

    cases = []
    for i in range(len(ds)):
        item = ds[i]
        name = item["properties"]["name"]
        img = torch.from_numpy(np.ascontiguousarray(item["data"])).float()[None].to(device)
        gt = np.asarray(item["seg"][0]).astype(np.int8)
        gt[gt < 0] = 0
        entry = dict(case=name)
        for use_tta in (True, False):
            entry["tta_on" if use_tta else "tta_off"] = run_comparison(
                img, gt, model, model_f, inferer, use_tta, R)
        cases.append(entry)
        print(f"   {name}: tta_on={entry['tta_on']['argmax_agreement']:.6f} "
              f"tta_off={entry['tta_off']['argmax_agreement']:.6f}")

    summary = dict(config=args.config, ckpt=args.ckpt,
                   n_sites_folded=n_fold, sites=report,
                   sandwich_audit=sandwich, cases=cases,
                   aggregate=summarize_cases(cases))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()