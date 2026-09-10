"""Replot drift_vs_rho from an existing absorption CSV.

Used after ``make_rho_scatter.py`` has produced the per-site table, when you
want the figure at a different size / style without re-reading checkpoints.

    python scripts/replot_rho_scatter.py --csv analysis/absorption_sites.csv \
        --out-fig figs/drift_vs_rho.pdf
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', default='analysis/absorption_sites.csv')
    ap.add_argument('--out-fig', default='figs/drift_vs_rho.pdf')
    args = ap.parse_args()

    rows = []
    with open(args.csv, newline='') as fh:
        for r in csv.DictReader(fh):
            rows.append(dict(drift=float(r['drift']), rho=float(r['rho']),
                             label=r.get('label', r.get('arm', '?'))))
    print(f'read {len(rows)} sites from {args.csv}')

    plt.rcParams.update({'font.size': 6.5, 'font.family': 'serif',
                         'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                         'ytick.major.width': 0.6, 'lines.linewidth': 0.7})
    fig, ax = plt.subplots(figsize=(3.39, 1.55))
    inert = [r for r in rows if r['rho'] < 1e-12]
    live = [r for r in rows if r['rho'] >= 1e-12]
    ax.axhspan(-3, 3, color='0.90', zorder=0)
    ax.scatter([r['drift'] for r in inert], [0] * len(inert),
               s=12, marker='o', facecolors='none', edgecolors='#1f4e79',
               linewidths=0.7, zorder=3,
               label=fr'$\rho=0$ ($n={len(inert)}$)')
    ax.scatter([r['drift'] for r in live], [r['rho'] * 100 for r in live],
               s=14, marker='^', color='#c0392b', zorder=3,
               label=fr'$\rho>0$ ($n={len(live)}$)')
    ax.set_xlabel(r'basis drift $\|Q-I\|_F$', labelpad=1.5)
    ax.set_ylabel(r'escape fraction $\rho$ (%)', labelpad=1.5)
    ax.set_xlim(-0.15, 4.0)
    ax.set_ylim(-6, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False,
              fontsize=6.0, handletextpad=0.4, borderpad=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout(pad=0.3)
    plt.rcParams['text.usetex'] = False
    os.makedirs(os.path.dirname(args.out_fig) or '.', exist_ok=True)
    fig.savefig(args.out_fig, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(args.out_fig.replace('.pdf', '.png'), dpi=300,
                bbox_inches='tight', pad_inches=0.02)
    print('wrote', args.out_fig)


if __name__ == '__main__':
    main()