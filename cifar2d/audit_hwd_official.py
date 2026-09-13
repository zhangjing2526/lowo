#!/usr/bin/env python3
"""Audit the official HWD mixer class (Xu et al. 2023 / apple1986/HWD).

Published Down_wt is fixed Haar DWT + dense 1x1.  Any Q on that DWT
has rho=0.  No Cityscapes weights needed.

    python audit_hwd_official.py
    python audit_hwd_official.py --ckpt runs/a5_s123/best.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lowo2d import HWDOfficial2D
from verify_absorption_2d import audit, load_state_dict


def synthetic() -> None:
    print("Official HWD mixer + random learned Q  (expect rho=0)")
    m = HWDOfficial2D(16, 32, learnable=True)
    with torch.no_grad():
        m.wave.skew.normal_(0, 0.4)
    audit(m.state_dict())
    print("\nOfficial HWD mixer, Q frozen at I  (expect rho=0, drift=0)")
    audit(HWDOfficial2D(16, 32, learnable=False).state_dict())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="A5 or HWDQ checkpoint")
    args = ap.parse_args()
    synthetic()
    if args.ckpt is None:
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(here, "runs", "a5_s123", "best.pt"),
            r"G:\material\天枢\mamba\cifar2d\runs\a5_s123\best.pt",
        ]
        for default in candidates:
            if os.path.exists(default):
                args.ckpt = default
                break
    if args.ckpt:
        if not os.path.exists(args.ckpt):
            raise SystemExit(f"missing {args.ckpt}")
        print(f"\ncheckpoint {args.ckpt}")
        print("A5 uses the same dense 1x1 mixer class as official HWD.")
        audit(load_state_dict(args.ckpt))


if __name__ == "__main__":
    main()
