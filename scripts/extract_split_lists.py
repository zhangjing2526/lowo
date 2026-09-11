"""Recover frozen case lists from eval artifacts.

The paper split is an internal held-out subset of the BraTS2021 training
pool, not the official BraTS test set. Evaluation writes
``tta_val_<CaseID>.npz`` and ``tta_test_<CaseID>.npz``. If those files still
exist on the training host, this script rebuilds ``val_125.txt`` /
``test_250.txt`` for ``splits/``.

    python scripts/extract_split_lists.py /path/to/analysis_results/<arm> \
        --out splits
"""
from __future__ import annotations

import argparse
import os
import re


def stems_from_dir(eval_dir: str, tag: str) -> list[str]:
    pat = re.compile(rf"^tta_{re.escape(tag)}_(.+)\.npz$")
    names = []
    for fn in os.listdir(eval_dir):
        m = pat.match(fn)
        if m:
            names.append(m.group(1))
    return sorted(names)


def write_list(path: str, names: list[str], kind: str) -> None:
    header = (
        f"# Internal held-out {kind} subset of the BraTS2021 training pool.\n"
        "# NOT the official BraTS test set. One CaseID per line.\n"
        f"# Recovered from tta_{kind}_*.npz ({len(names)} cases).\n"
    )
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(header)
        for n in names:
            f.write(n + "\n")
    print(f"wrote {path} ({len(names)} cases)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir", help="directory with tta_val_*.npz / tta_test_*.npz")
    ap.add_argument("--out", default="splits")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    val = stems_from_dir(args.eval_dir, "val")
    test = stems_from_dir(args.eval_dir, "test")
    if not val and not test:
        raise SystemExit(f"no tta_val_*.npz or tta_test_*.npz under {args.eval_dir}")
    if val:
        write_list(os.path.join(args.out, "val_125.txt"), val, "val")
    if test:
        write_list(os.path.join(args.out, "test_250.txt"), test, "test")


if __name__ == "__main__":
    main()
