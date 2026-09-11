#!/usr/bin/env python3
"""Print best_test for every finished run under ./runs."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs")


def main() -> None:
    if not os.path.isdir(RUNS):
        print("no runs/ yet")
        return
    rows = []
    for name in sorted(os.listdir(RUNS)):
        if "wdskew" in name:
            continue
        p = os.path.join(RUNS, name, "summary.txt")
        if os.path.isfile(p):
            rows.append(open(p, encoding="utf-8").read().strip())
    if not rows:
        print("no summary.txt yet; jobs still running")
        return
    print("\n".join(rows))


if __name__ == "__main__":
    sys.exit(main())
