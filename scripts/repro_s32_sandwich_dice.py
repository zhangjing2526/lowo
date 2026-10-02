"""Reproduce ICASSP-2027 LOWO Sec.3.2 -- grouped-sandwich projection cost.

This is a thin, portable entry point around ``scripts/fold_sandwich_projection.py``
(the script that produced the paper's ``sandwich_projection_final.json``).  It
only pins the repo-relative defaults (repo root, A6 config, frozen 250-list);
every other flag is forwarded verbatim to that script's own argparse, so any
flag it accepts also works here.

What Sec.3.2 reports (verbatim numbers, already reproduced by the shipped JSON):

    Using the A6 checkpoint (so8, learnable Haar, groups=8), three grouped-
    sandwich sites with low / medium / high rho (0.166 / 0.352 / 0.593) were
    evaluated on 250 held-out BraTS cases.  Exact reparameterization changed
    mean Dice by less than 4e-6 in all cases, whereas forced shared-block
    projection increased the worst-case single-case Dice drift from 0.0038 to
    0.344.  Thus rho provides a practical signal for screening unsafe
    structural simplification before deployment.

REQUIREMENTS (NOT in this repo -- supply via CLI)
  --ckpt      full A6 checkpoint, e.g.
              <mamba>/test/segmamba_haar_so8live/model/best_model_0.9317.pt
  --data_dir  BraTS2021 preprocessed .npz dir on the 8xA100 host
              (./data/fullres/train)

USAGE (8xA100 host, from the lowo repo root)
  # 1) rank every sandwich site by rho (no data needed); writes
  #    sandwich_rho_ranking.json -- the audit trail of the site pick.
  python scripts/repro_s32_sandwich_dice.py --ckpt <A6.pt> --rank-only

  # 2) pilot on 10 cases (explore; do NOT pick sites from these numbers).
  python scripts/repro_s32_sandwich_dice.py --ckpt <A6.pt> --data_dir <brats> \\
      --n_cases 10 --purpose pilot --out sandwich_projection_pilot.json

  # 3) freeze the protocol, then run the held-out 250 with purpose=final.
  python scripts/repro_s32_sandwich_dice.py --ckpt <A6.pt> --data_dir <brats> \\
      --purpose final --out sandwich_projection_final.json

  # CPU self-check, no checkpoint / no data (clone sanity).
  python scripts/repro_s32_sandwich_dice.py --selftest

The sites are chosen low/mid/high rho by the underlying script (recomputed from
the checkpoint, never carried over).  To pin them explicitly:
  --sites vit.gscs.1.proj2,vit.gscs.1.proj,vit.gscs.0.proj2

Related audit scripts in this folder (different experiments, not Sec.3.2):
  fold_all_sites_check.py     -- Sec.3.3 dense-fold composition on A5 (chain-level)
  fold_audit_supplement.py    -- GT Dice + TTA ablation + Sec.4 strict sandwich audit
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)            # lowo/  == repo root


def main() -> None:
    sys.path.insert(0, _HERE)
    # Repo-relative defaults.  User-supplied flags (appended) override them
    # because argparse keeps the LAST occurrence of each option.
    defaults = [
        "repro_s32_sandwich_dice",
        "--repo", _REPO,
        "--config", os.path.join(_REPO, "configs", "A6_so8_g8.yaml"),
        "--test_list", os.path.join(_REPO, "splits", "test_250.txt"),
    ]
    sys.argv = defaults + sys.argv[1:]
    import fold_sandwich_projection
    fold_sandwich_projection.main()


if __name__ == "__main__":
    main()
