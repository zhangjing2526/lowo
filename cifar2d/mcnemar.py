#!/usr/bin/env python3
"""Paired McNemar test on official CIFAR-10 test predictions.

    python mcnemar.py runs/a2_s123/best.pt runs/a6_s123/best.pt
    python mcnemar.py --refresh runs/a2_s123/best.pt   # fill test_y / test_pred
    python mcnemar.py --all --runs runs                 # every pair, same seed
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from itertools import combinations

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _binom_cdf_le(k: int, n: int) -> float:
    """P(X <= k) for X ~ Bin(n, 1/2)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    logp = -n * math.log(2.0)
    s = 0.0
    for i in range(k + 1):
        s += math.exp(
            math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + logp
        )
    return min(1.0, s)


def mcnemar(y: torch.Tensor, pred_a: torch.Tensor, pred_b: torch.Tensor):
    a_ok = pred_a.eq(y)
    b_ok = pred_b.eq(y)
    n01 = int((a_ok & ~b_ok).sum())  # A right, B wrong
    n10 = int((~a_ok & b_ok).sum())  # A wrong, B right
    n = n01 + n10
    acc_a = float(a_ok.float().mean())
    acc_b = float(b_ok.float().mean())
    if n == 0:
        return dict(n01=n01, n10=n10, p=1.0, acc_a=acc_a, acc_b=acc_b)
    p = 2.0 * _binom_cdf_le(min(n01, n10), n)
    return dict(n01=n01, n10=n10, p=min(1.0, p), acc_a=acc_a, acc_b=acc_b)


def load_preds(path: str):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if "test_y" not in obj or "test_pred" not in obj:
        raise SystemExit(
            f"{path} has no test_y/test_pred; run: python mcnemar.py --refresh {path}"
        )
    return obj["test_y"].long(), obj["test_pred"].long(), obj


def refresh(path: str, data_dir: str, device: str) -> None:
    from train_cifar import collect_preds, make_loaders
    from resnet_lowo import build_model

    obj = torch.load(path, map_location="cpu", weights_only=False)
    arm = obj.get("arm")
    if not arm:
        raise SystemExit(f"{path}: checkpoint has no 'arm'")
    device_t = torch.device(device)
    model = build_model(arm).to(device_t)
    model.load_state_dict(obj["state_dict"])
    _, _, test_ld = make_loaders(data_dir, batch=256, workers=2, seed=0)
    y, pred = collect_preds(model, test_ld, device_t)
    obj["test_y"] = y
    obj["test_pred"] = pred
    obj["test_acc"] = float((pred == y).float().mean())
    torch.save(obj, path)
    print(f"refreshed {path}  test_acc={obj['test_acc']:.4f}  n={len(y)}")


def report(path_a: str, path_b: str) -> None:
    ya, pa, oa = load_preds(path_a)
    yb, pb, ob = load_preds(path_b)
    if not torch.equal(ya, yb):
        raise SystemExit("test_y mismatch; refresh both from the official test set")
    r = mcnemar(ya, pa, pb)
    la = oa.get("arm", os.path.basename(os.path.dirname(path_a)))
    lb = ob.get("arm", os.path.basename(os.path.dirname(path_b)))
    print(
        f"{la} {r['acc_a']:.4f}  vs  {lb} {r['acc_b']:.4f}  "
        f"n01={r['n01']} n10={r['n10']}  McNemar p={r['p']:.4g}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="*")
    ap.add_argument("--refresh", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--runs", default=os.path.join(os.path.dirname(__file__), "runs"))
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "data"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.refresh is not None:
        targets = args.refresh or args.ckpts
        if not targets:
            raise SystemExit("pass checkpoint paths to --refresh")
        for p in targets:
            refresh(p, args.data, args.device)
        return
    if args.all:
        by_seed: dict[int, list[str]] = {}
        for name in sorted(os.listdir(args.runs)):
            if "wdskew" in name:
                continue
            p = os.path.join(args.runs, name, "best.pt")
            if not os.path.isfile(p):
                continue
            try:
                obj = torch.load(p, map_location="cpu", weights_only=False)
            except Exception:
                continue
            seed = obj.get("seed")
            if seed is None:
                continue
            by_seed.setdefault(int(seed), []).append(p)
        for seed, paths in sorted(by_seed.items()):
            print(f"--- seed {seed} ---")
            for a, b in combinations(paths, 2):
                try:
                    report(a, b)
                except SystemExit as e:
                    print(f"  skip {a} vs {b}: {e}")
        return
    if len(args.ckpts) != 2:
        raise SystemExit("usage: mcnemar.py ckpt_a ckpt_b")
    report(args.ckpts[0], args.ckpts[1])


if __name__ == "__main__":
    main()
