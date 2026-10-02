"""Reproduce the paper's Fig. 2 -- basis drift ||Q-I_8||_F vs escape fraction
rho over the 163 audited transform sites (ICASSP 2027, single column).

This script needs NEITHER torch NOR the training checkpoints: it reads the
tracked ``analysis/absorption_sites.csv`` (one row per audited site) and redraws
the exact figure shipped in ``figs/drift_vs_rho.pdf``.  It is the single source
of truth for the figure; ``make_rho_scatter.py`` regenerates the CSV and then
calls :func:`plot_drift_vs_rho` from here.

    python scripts/replot_rho_scatter.py                  # writes figs/drift_vs_rho.pdf + .png
    python scripts/replot_rho_scatter.py --out /tmp/x.pdf

The three labelled points name the SITE (the stem / stride-2 downsample mixer),
not the grouping, so the plot reads together with the sentence that contrasts the
A4 stem (grouped mixer, rho=0.839) with the A3 stem (dense mixer, rho=0):

  * A6 stem -- grouped mixer, ||Q-I||_F=3.53, rho=0.884  (the load-bearing case)
  * A4 stem -- grouped mixer, ||Q-I||_F=2.70, rho=0.839
  * C1 stem -- dense mixer,   ||Q-I||_F=3.53, rho=0      (inert: absorbed exactly)

The figure is drawn at the paper's exact size (3.39 x 2.25 in, 244 x 162 pt)
with no ``bbox_inches='tight'`` so a clone reproduces the submitted PDF.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO / 'analysis' / 'absorption_sites.csv'
DEFAULT_OUT = REPO / 'figs' / 'drift_vs_rho.pdf'


def load_rows(csv_path=DEFAULT_CSV):
    rows = []
    with open(csv_path, newline='') as fh:
        for r in csv.DictReader(fh):
            rows.append({
                'arm': r['arm'], 'label': r['label'], 'site': r['site'],
                'role': r['role'], 'groups': int(r['groups']),
                'drift': float(r['drift']),
                'rho': float(r['rho']),
                'fold_err': float(r['fold_err']),
            })
    return rows


def _stem(rows, arm):
    return next(r for r in rows
                if r['arm'] == arm and r['site'].endswith('downsample_layers.0.0'))


def plot_drift_vs_rho(rows, out_pdf=DEFAULT_OUT, dpi=300):
    """Draw Fig. 2 from a list of site dicts; write ``out_pdf`` (.pdf + .png)."""
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif',
                       'Nimbus Roman', 'serif'],
        'mathtext.fontset': 'stix', 'font.size': 9,
    })

    identity = [r for r in rows if r['rho'] < 1e-12 and r['drift'] == 0.0]
    dense = [r for r in rows if r['rho'] < 1e-12 and r['drift'] > 0.0]
    live = [r for r in rows if r['rho'] >= 1e-12]

    fig, ax = plt.subplots(figsize=(3.39, 2.25))
    fig.subplots_adjust(left=.15, right=.98, bottom=.215, top=.71)
    # Draw the three classes separately.  The 59 non-trivial-but-removable
    # sites (dense mixer) are the key set, so the 32 frozen Q==I zeros are
    # reported on their own.
    ax.scatter([r['drift'] for r in identity], [0] * len(identity),
               s=22, c='#777777', zorder=4,
               label=r'Frozen: $Q=I_8$ ($n=%d$)' % len(identity))
    ax.scatter([r['drift'] for r in dense], [0] * len(dense),
               s=20, facecolors='white', edgecolors='#235a87', linewidths=.9,
               zorder=3,
               label=r'Dense: $Q\neq I_8$, $\rho=0$ ($n=%d$)' % len(dense))
    ax.scatter([r['drift'] for r in live], [100 * r['rho'] for r in live],
               s=22, c='#be4435', marker='^', linewidths=.2, zorder=3,
               label=r'Per-subband: $\rho>0$ ($n=%d$)' % len(live))

    handles, labels = ax.get_legend_handles_labels()
    fig.legend([handles[i] for i in [1, 0, 2]], [labels[i] for i in [1, 0, 2]],
               loc='upper center', bbox_to_anchor=(.53, 1.01), frameon=False,
               fontsize=9, handletextpad=.5, labelspacing=.3, borderaxespad=0)

    ax.set(xlim=(-.15, 4), ylim=(-6, 103), xticks=[0, 1, 2, 3, 4],
           yticks=[0, 50, 100])
    ax.set_xlabel(r'Basis drift $\|Q-I_8\|_F$', labelpad=1)
    ax.set_ylabel(r'$\rho$ (%)', labelpad=1)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.28, linestyle='--', linewidth=.6)
    ax.tick_params(length=3, pad=2)

    # Labels name the SITE (stem), not the grouping, so the figure reads with
    # the A4-stem vs A3-stem contrast in the text.
    a6 = _stem(rows, 'A6')
    a4 = _stem(rows, 'A4')
    c1 = _stem(rows, 'C1')
    ax.annotate('A6 stem', xy=(a6['drift'], 100 * a6['rho']), xytext=(1.8, 94),
                fontsize=9, color='#a53a2e',
                arrowprops=dict(arrowstyle='->', color='#a53a2e', lw=.7))
    ax.annotate('A4 stem', xy=(a4['drift'], 100 * a4['rho']), xytext=(0.32, 70),
                fontsize=9, color='#a53a2e',
                arrowprops=dict(arrowstyle='->', color='#a53a2e', lw=.7))
    ax.annotate('C1 stem', xy=(c1['drift'], 0), xytext=(2.05, 16),
                fontsize=9, color='#235a87',
                arrowprops=dict(arrowstyle='->', color='#235a87', lw=.7))

    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    # Exact paper size: do NOT use bbox_inches='tight' (would change the canvas).
    fig.savefig(out_pdf, dpi=dpi)
    fig.savefig(out_pdf.with_suffix('.png'), dpi=dpi)
    plt.close(fig)
    return dict(sites=len(rows), identity=len(identity), dense=len(dense),
                live=len(live), a4_stem=a4, a6_stem=a6, c1_stem=c1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=Path, default=DEFAULT_CSV,
                    help='audit CSV (default: analysis/absorption_sites.csv)')
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--dpi', type=int, default=300)
    args = ap.parse_args()

    rows = load_rows(args.csv)
    summary = plot_drift_vs_rho(rows, args.out, args.dpi)
    print('read %d sites from %s' % (len(rows), args.csv))
    print('wrote %s (identity %d, dense %d, live %d)'
          % (args.out, summary['identity'], summary['dense'], summary['live']))


if __name__ == '__main__':
    main()
