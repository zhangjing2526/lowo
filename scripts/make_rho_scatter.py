"""Dump every wavelet site of every checkpoint, write ``analysis/absorption_sites.csv``,
then redraw the paper Fig. 2.

The point of the figure: ``||Q-I||_F`` (how far the basis moved) carries no
information about whether the move did anything.  Sites whose mixer is dense sit
exactly on rho = 0 no matter how far right they are; only the grouped mixers rise
off the axis.

Needs the training checkpoints, which are NOT shipped in the repo.  If they are
absent the script falls back to the tracked ``analysis/absorption_sites.csv`` and
only redraws the figure, so it still runs in a clone.  Paths are resolved from the
repo root; pass ``--logs_root`` / ``--test_root`` to point at a local copy of the
training logs.

    python scripts/make_rho_scatter.py
    python scripts/make_rho_scatter.py --logs_root /data/logs --test_root /data/test
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # so `import verify_absorption` works from scripts/
import verify_absorption as va

CSV_OUT = REPO / 'analysis' / 'absorption_sites.csv'
A4_CSV = REPO / 'analysis' / 'a4_sites.csv'

# Highest-Dice checkpoint of each run -- the same ten networks the paper audits.
# These live outside the repo; override with --logs_root / --test_root.
DEFAULT_LOGS = Path(r'G:/material/天枢/mamba/logs')
DEFAULT_TEST = Path(r'G:/material/天枢/mamba/test')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs_root', type=Path, default=DEFAULT_LOGS)
    ap.add_argument('--test_root', type=Path, default=DEFAULT_TEST)
    ap.add_argument('--csv_out', type=Path, default=CSV_OUT)
    ap.add_argument('--out', type=Path, default=REPO / 'figs' / 'drift_vs_rho.pdf')
    ap.add_argument('--dpi', type=int, default=300)
    args = ap.parse_args()

    LOGS, TEST = args.logs_root, args.test_root
    # (arm, label, checkpoint) -- highest-Dice checkpoint of each run.
    CKPTS = [
        ('A1', 'fixed, g1',
         LOGS / 'segmamba_haar_so8_dicece_ds' / 'segmamba_haar_fixed' / 'model' / 'best_model_0.9268.pt'),
        ('A2', 'fixed, g8',
         TEST / 'segmamba_haar_fixed_g8' / 'model' / 'best_model_0.9285.pt'),
        ('A3', 'theta, g1',
         LOGS / 'segmamba_haar_dicece_ds' / 'segmamba_haar_theta' / 'model' / 'best_model_0.9251.pt'),
        ('A4', 'theta, g8',
         LOGS / 'segmamba_haar_dicece_ds' / 'segmamba_haar_theta_g8_s123_s123' / 'model' / 'best_model_0.9224.pt'),
        ('A5', 'so8, g1',
         LOGS / 'segmamba_haar_so8_dicece_ds' / 'segmamba_haar' / 'model' / 'best_model_0.9225.pt'),
        ('A6', 'so8, g8',
         TEST / 'segmamba_haar_so8live' / 'model' / 'best_model_0.9317.pt'),
        ('A7', 'so8, g1, +wb',
         LOGS / 'segmamba_haar_so8_dicece_ds' / 'segmamba_haar_wb' / 'model' / 'best_model_0.9181.pt'),
        ('A8', 'so8, per-channel',
         LOGS / 'segmamba_haar_so8_dicece_ds' / 'segmamba_haar_perch' / 'model' / 'best_model_0.9098.pt'),
        ('C1', 'SegAttn-Haar',
         LOGS / 'segattn_haar_so8_dicece_ds' / 'segattn_haar' / 'model' / 'best_model_0.9190.pt'),
        ('C2', 'SegAttn + routing',
         LOGS / 'segattn_haar_so8_dicece_ds' / 'segattn_haar_routed' / 'model' / 'best_model_0.9152.pt'),
    ]

    rows = []
    for arm, label, path in CKPTS:
        if not path.exists():
            print(f'!! missing {arm}: {path}')
            continue
        sd = va.load_state_dict(str(path))
        sites = va.discover(sd, in_chans=4)
        for i, site in enumerate(sites):
            if site['kind'] == 'orphan' or site['groups'] == 0:
                print(f'!! {arm} unresolved site {site["path"]}')
                continue
            q = va.orthogonal_factor(sd, site['wave'])
            drift = (q - torch.eye(8, dtype=va.DT)).norm().item()
            if site['kind'] == 'synthesis':
                res = va.check_synthesis(site, q, seed=1234 + i)
            elif site['kind'] == 'waveletconv':
                res = va.check_waveletconv(site, q, seed=1234 + i)
            else:
                res = va.check_analysis(site, q, seed=1234 + i)
            rows.append(dict(arm=arm, label=label, site=site['path'],
                             role=site['role'], groups=site['groups'],
                             drift=drift, rho=res['off_block'],
                             fold_err=res['err_fold']))
        print(f'{arm}: {sum(1 for r in rows if r["arm"] == arm)} sites')

    # A4 seed-123 can be merged from analysis/a4_sites.csv when the full
    # checkpoint is not on disk.
    if not any(r['arm'] == 'A4' for r in rows) and A4_CSV.exists():
        with open(A4_CSV, newline='') as fh:
            for r in csv.DictReader(fh):
                if r['arm'] != 'A4-123':
                    continue
                rows.append(dict(arm='A4', label='theta, g8',
                                 site=r['site'], role=r['role'],
                                 groups=int(r['groups']),
                                 drift=float(r['drift']),
                                 rho=float(r['rho']),
                                 fold_err=float(r['fold_err'])))
        print('A4: 16 sites (from analysis/a4_sites.csv, seed 123)')

    found_arms = sorted({r['arm'] for r in rows})
    if rows and len(found_arms) == len(CKPTS):
        with open(args.csv_out, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        n_inert = sum(1 for r in rows if r['rho'] < 1e-12)
        print(f'\ntotal sites {len(rows)}   inert (rho=0) {n_inert}   '
              f'max fold err {max(r["fold_err"] for r in rows):.2e}')
        print(f'max drift among inert sites: '
              f'{max(r["drift"] for r in rows if r["rho"] < 1e-12):.3f}')
    else:
        # Partial checkpoint set (e.g. a machine without every training run):
        # do NOT overwrite the tracked 163-site CSV with a partial table.
        if rows:
            print(f'!! only {len(found_arms)}/{len(CKPTS)} arms found '
                  f'({",".join(found_arms)}); keeping the tracked CSV')
        print('using tracked CSV for the figure')
        if not args.csv_out.exists():
            raise SystemExit(f'[fatal] {args.csv_out} missing and no checkpoints')
        raw = list(csv.DictReader(open(args.csv_out, newline='')))
        rows = []
        for r in raw:
            rows.append(dict(arm=r['arm'], label=r['label'], site=r['site'],
                             role=r['role'], groups=int(r['groups']),
                             drift=float(r['drift']), rho=float(r['rho']),
                             fold_err=float(r['fold_err'])))

    # Redraw the figure with the canonical, paper-accurate renderer.
    sys.path.insert(0, str(REPO / 'scripts'))
    from replot_rho_scatter import plot_drift_vs_rho
    summary = plot_drift_vs_rho(rows, args.out, args.dpi)
    print('identity %d, dense %d, live %d'
          % (summary['identity'], summary['dense'], summary['live']))


if __name__ == '__main__':
    main()
