"""Dump every wavelet site of every checkpoint, then plot drift against rho.

The point of the figure: ||Q-I||_F (how far the basis moved) carries no
information about whether the move did anything.  Sites whose subband mixer
is dense (groups=1) sit exactly on rho=0 no matter how far right they are;
only grouped (groups=8) mixers rise off the axis.

Usage
-----
Run on the nine checkpoints that back the paper (paths under ``LOGS``/``ROOT``),
or supply your own:

    python scripts/make_rho_scatter.py
    python scripts/make_rho_scatter.py --ckpt path/to/best_model.pt [path2 ...]
    python scripts/make_rho_scatter.py \
        --ckpt A2@logs/segmamba_haar_fixed_g8/model/best_model.pt \
        --out-csv /tmp/sites.csv --out-fig /tmp/drift_vs_rho.pdf

A ``tag@path`` prefix labels the arm; without one the file basename is used.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

# make the repo-root module importable regardless of the current working dir
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import verify_absorption as va

LOGS = os.path.join(ROOT, 'logs')

# (arm, label, checkpoint) — highest-Dice checkpoint of each run, the same
# nine networks the paper audits.  Adjust paths to your own log layout.
CKPTS = [
    ('A1', 'fixed, g1',
     rf'{LOGS}\segmamba_haar_so8_dicece_ds\segmamba_haar_fixed\model\best_model_0.9268.pt'),
    ('A2', 'fixed, g8',
     rf'{ROOT}\test\segmamba_haar_fixed_g8\model\best_model_0.9285.pt'),
    ('A3', 'theta, g1',
     rf'{LOGS}\segmamba_haar_dicece_ds\segmamba_haar_theta\model\best_model_0.9251.pt'),
    ('A5', 'so8, g1',
     rf'{LOGS}\segmamba_haar_so8_dicece_ds\segmamba_haar\model\best_model_0.9225.pt'),
    ('A6', 'so8, g8',
     rf'{ROOT}\test\segmamba_haar_so8live\model\best_model_0.9317.pt'),
    ('A7', 'so8, g1, +wb',
     rf'{LOGS}\segmamba_haar_so8_dicece_ds\segmamba_haar_wb\model\best_model_0.9181.pt'),
    ('A8', 'so8, per-channel',
     rf'{LOGS}\segmamba_haar_so8_dicece_ds\segmamba_haar_perch\model\best_model_0.9098.pt'),
    ('C1', 'SegAttn-Haar',
     rf'{LOGS}\segattn_haar_so8_dicece_ds\segattn_haar\model\best_model_0.9190.pt'),
    ('C2', 'SegAttn + routing',
     rf'{LOGS}\segattn_haar_so8_dicece_ds\segattn_haar_routed\model\best_model_0.9152.pt'),
]


def parse_ckpt(arg: str) -> tuple[str, str]:
    """Return (tag, path) for a '[tag@]path' CLI argument."""
    if '@' in arg:
        tag, path = arg.split('@', 1)
        return tag, path
    return os.path.basename(arg), arg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', nargs='*', default=None,
                    help='[tag@]path checkpoints; default = the 9 paper arms')
    ap.add_argument('--in-chans', type=int, default=4,
                    help='input modalities feeding downsample_layers.0')
    ap.add_argument('--tol', type=float, default=1e-10,
                    help='fold error below which a site counts as inert')
    ap.add_argument('--out-csv', default=None,
                    help='output CSV (default analysis/absorption_sites.csv)')
    ap.add_argument('--out-fig', default=None,
                    help='output figure (default figs/drift_vs_rho.pdf)')
    args = ap.parse_args()

    ckpts = [parse_ckpt(c) for c in (args.ckpt or CKPTS)]
    out_csv = args.out_csv or os.path.join(ROOT, 'analysis', 'absorption_sites.csv')
    out_fig = args.out_fig or os.path.join(ROOT, 'figs', 'drift_vs_rho.pdf')

    rows = []
    for arm, label, path in ckpts:
        if not os.path.exists(path):
            print(f'!! missing {arm}: {path}')
            continue
        sd = va.load_state_dict(path)
        sites = va.discover(sd, in_chans=args.in_chans)
        for i, site in enumerate(sites):
            if site['kind'] == 'orphan' or site['groups'] == 0:
                print(f'!! {arm} unresolved site {site["path"]}')
                continue
            q = va.orthogonal_factor(sd, site['wave'])
            drift = (q - torch.eye(8, dtype=va.DT)).norm().item()
            res = (va.check_synthesis if site['kind'] == 'synthesis'
                   else va.check_analysis)(site, q, seed=1234 + i)
            rows.append(dict(arm=arm, label=label, site=site['path'],
                             role=site['role'], groups=site['groups'],
                             drift=drift, rho=res['off_block'],
                             fold_err=res['err_fold']))
        print(f'{arm}: {sum(1 for r in rows if r["arm"] == arm)} sites')

    if not rows:
        raise SystemExit('no checkpoints found')

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n_inert = sum(1 for r in rows if r['rho'] < 1e-12)
    print(f'\ntotal sites {len(rows)}   inert (rho=0) {n_inert}   '
          f'max fold err {max(r["fold_err"] for r in rows):.2e}')
    print(f'max drift among inert sites: '
          f'{max(r["drift"] for r in rows if r["rho"] < 1e-12):.3f}')

    # ------------------------------------------------------------------ figure
    plt.rcParams.update({'font.size': 8, 'font.family': 'serif',
                         'axes.linewidth': 0.7, 'xtick.major.width': 0.7,
                         'ytick.major.width': 0.7})
    fig, ax = plt.subplots(figsize=(6.0, 1.9))
    inert = [r for r in rows if r['rho'] < 1e-12]
    live = [r for r in rows if r['rho'] >= 1e-12]
    ax.axhspan(-3, 3, color='0.90', zorder=0)
    ax.scatter([r['drift'] for r in inert], [0] * len(inert),
               s=22, marker='o', facecolors='none', edgecolors='#1f4e79',
               linewidths=0.9, zorder=3,
               label=fr'$\rho=0$ ($n={len(inert)}$)')
    ax.scatter([r['drift'] for r in live], [r['rho'] * 100 for r in live],
               s=24, marker='^', color='#c0392b', zorder=3,
               label=fr'$\rho>0$ ($n={len(live)}$)')
    ax.annotate('A5 stem, inert', xy=(2.84, 0), xytext=(2.84, 18),
                ha='center', fontsize=7, color='#1f4e79',
                arrowprops=dict(arrowstyle='-', lw=0.5, color='#1f4e79'))
    ax.annotate('C1 stem, inert', xy=(3.53, 0), xytext=(3.65, 60),
                ha='left', fontsize=7, color='#1f4e79',
                arrowprops=dict(arrowstyle='-', lw=0.5, color='#1f4e79'))
    ax.annotate(fr'A6 stem, $\rho={0.88*100:.0f}$%',
                xy=(3.53, 88), xytext=(3.45, 75),
                ha='right', fontsize=7, color='#c0392b',
                arrowprops=dict(arrowstyle='-', lw=0.5, color='#c0392b'))
    ax.set_xlabel(r'basis drift $\|Q-I\|_F$', labelpad=2)
    ax.set_ylabel(r'escape fraction $\rho$ (%)', labelpad=2)
    ax.set_xlim(-0.15, 4.0)
    ax.set_ylim(-6, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False,
              fontsize=8, handletextpad=0.5, borderpad=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout(pad=0.4)
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    plt.rcParams['text.usetex'] = False
    fig.savefig(out_fig, bbox_inches='tight', pad_inches=0.01)
    fig.savefig(out_fig.replace('.pdf', '.png'), dpi=300,
                bbox_inches='tight', pad_inches=0.01)
    print('wrote', out_fig)


if __name__ == '__main__':
    main()