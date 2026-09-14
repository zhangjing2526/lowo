"""Replot drift_vs_rho as a single-column figure.

Single-column figure for the ICASSP paper (Sec. 4.1), the central audit plot.
Reads absorption_sites.csv (produced by make_rho_scatter.py, which needs the
checkpoints); this script needs neither torch nor the checkpoints.

    python replot_rho_scatter.py                        # design width 3.39 in
    python replot_rho_scatter.py --width 3.386 \
        --compact --fontsize 9 \
        --out ../v13/figs/drift_vs_rho.pdf              # paper Fig. 3 (v13)

IMPORTANT -- design width must match the display width.  A figure designed at
3.39 in but included at ``width=0.70\\columnwidth`` (2.37 in) has all of its
type scaled by 0.70, which turns 6.5 pt labels into ~4.6 pt.  Pass the actual
display width here instead, so the point sizes below are the printed sizes.
For the v13 paper Fig. 3 the figure is printed at ``\\columnwidth`` (3.386 in),
so design at 3.386 in and set --fontsize 9 -> printed type is ~9 pt, clearing
the ICASSP minimum.

v6 (2026-09-13) -- added --fontsize; the base size is now the printed size
  when --width equals the display width.  Default output unchanged.
v5 (2026-09-13) -- added --width / --out.
v4 (2026-09-13) -- restructure:
  * draw the three classes separately; the 59 sites with a NON-TRIVIAL
    rotation and still exactly removable are the key set, and reporting
    "rho = 0 at 91/163" alone does not distinguish the 32 Q == I zeros.
  * legend states all three counts; the Q == I group gets its own annotation.

v3 (2026-09-09) -- publication-quality polish: legend inside the plot,
annotations decoupled from code with explicit xytext + arrowstyle, translucent
white bboxes, serif typeface, inward ticks, faint dashed gridlines, classic
navy / dark-red palette.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = r'G:\material\天枢\mamba'
CSV = os.path.join(ROOT, 'absorption_sites.csv')

_ap = argparse.ArgumentParser()
_ap.add_argument('--width', type=float, default=3.39,
                 help='design width in inches; set this to the *display* width '
                      '(\\columnwidth = 3.386 in) so the point sizes below are '
                      'the printed sizes')
_ap.add_argument('--height', type=float, default=None,
                 help='design height; default keeps the 3.39x1.60 aspect')
_ap.add_argument('--out', default=os.path.join(ROOT, 'figs', 'drift_vs_rho.pdf'))
_ap.add_argument('--compact', action='store_true',
                 help='drop the in-plot legend and the redundant Q==I '
                      'annotation; the caption already states n=59/32/72.  '
                      'Use for narrow (<=2.5 in) designs, where a full-size '
                      'legend would cover the data.')
_ap.add_argument('--fontsize', type=float, default=9.0,
                 help='base point size.  Set --width to the DISPLAY width so '
                      'this IS the printed size; 9.0 clears the ICASSP '
                      '"no smaller than nine point" minimum.')
ARGS = _ap.parse_args()
F = ARGS.fontsize

rows = []
with open(CSV) as fh:
    for r in csv.DictReader(fh):
        rows.append(dict(arm=r['arm'], label=r['label'], site=r['site'],
                         role=r['role'], groups=int(r['groups']),
                         drift=float(r['drift']),
                         rho=float(r['rho']),
                         fold_err=float(r['fold_err'])))

print('read %d sites from %s' % (len(rows), CSV))

# ---------------------------------------------------------------------------
# typeface / line weights: serif (Times New Roman style), inward ticks
# ---------------------------------------------------------------------------
plt.rcParams.update({
    # TrueType embedding (Type 42): required for IEEE PDF inspection.
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif',
                   'Nimbus Roman', 'serif'],
    'font.size': F,
    'axes.labelsize': F,
    'xtick.labelsize': F,
    'ytick.labelsize': F,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'lines.linewidth': 0.7,
    # Force math mode to use STIX, which has similar x-height to Times New
    # Roman.  Without this, axis labels mixing text and math (e.g. "basis drift
    # ‖Q-I‖_F") use two different fonts at the same point size but look
    # visibly different sizes because Computer Modern has a smaller x-height than
    # Times New Roman.
    'mathtext.fontset': 'stix',
})

fig, ax = plt.subplots(figsize=(ARGS.width,
                                ARGS.height if ARGS.height
                                else 1.60 * ARGS.width / 3.39))

# three classes: trivial zeros, non-trivial but removable, load-bearing
identity = [r for r in rows if r['rho'] < 1e-12 and r['drift'] == 0.0]
removed = [r for r in rows if r['rho'] < 1e-12 and r['drift'] > 0.0]
live = [r for r in rows if r['rho'] >= 1e-12]

# faint dashed horizontal gridlines (kept under the data)
ax.set_axisbelow(True)
ax.grid(axis='y', linestyle='--', linewidth=0.35, color='#c9c9c9', alpha=0.7)

ax.scatter([r['drift'] for r in identity], [0] * len(identity),
           s=19, marker='o', color='#9a9a9a', edgecolors='none', zorder=4,
           label=r'$\rho=0$, $Q\equiv I$  ($n=%d$)' % len(identity))
ax.scatter([r['drift'] for r in removed], [0] * len(removed),
           s=13, marker='o', facecolors='none', edgecolors='#1f4e79',
           linewidths=0.8, zorder=3,
           label=r'$\rho=0$, $Q\neq I$  ($n=%d$)' % len(removed))
ax.scatter([r['drift'] for r in live], [r['rho'] * 100 for r in live],
           s=15, marker='^', color='#c0392b', edgecolors='none', zorder=2,
           label=r'$\rho>0$  ($n=%d$)' % len(live))

# ---------------------------------------------------------------------------
# annotations: explicit xytext, arrowstyle '->', translucent white bbox
# ---------------------------------------------------------------------------
ann_bbox = dict(boxstyle='round,pad=0.18', facecolor='white',
                edgecolor='none', alpha=0.70)

if ARGS.compact:
    # Narrow design: the counts live in the caption, so no legend and no
    # Q == I annotation.  Only the two claims that need pointing at stay.
    ax.annotate(r'$\|Q-I\|_F{=}3.53$, $\rho{=}0$',
                xy=(3.53, 1.5), xytext=(3.80, 15),
                ha='right', va='bottom', fontsize=F, color='#1f4e79',
                bbox=ann_bbox,
                arrowprops=dict(arrowstyle='->', lw=0.5, color='#1f4e79',
                                shrinkA=2, shrinkB=2))
    ax.annotate(r'grouped stem, $\rho{=}88$%',
                xy=(3.53, 88), xytext=(1.55, 96),
                ha='left', va='center', fontsize=F, color='#c0392b',
                bbox=ann_bbox,
                arrowprops=dict(arrowstyle='->', lw=0.5, color='#c0392b',
                                shrinkA=2, shrinkB=2))
else:
    ax.annotate(r'$%d$ sites with $Q\equiv I$' % len(identity),
                xy=(0.02, 0.8), xytext=(1.32, 21),
                ha='left', va='center', fontsize=F - 0.5, color='#7a7a7a',
                bbox=ann_bbox,
                arrowprops=dict(arrowstyle='->', lw=0.5, color='#9a9a9a',
                                shrinkA=2, shrinkB=2))
    ax.annotate(r'$\|Q-I\|_F{=}3.53$, $\rho{=}0$',
                xy=(3.53, 0), xytext=(3.53, 34),
                ha='right', va='bottom', fontsize=F - 0.5, color='#1f4e79',
                bbox=ann_bbox,
                arrowprops=dict(arrowstyle='->', lw=0.5, color='#1f4e79',
                                shrinkA=2, shrinkB=2))
    ax.annotate(r'grouped stem, $\rho{=}88$%',
                xy=(3.53, 88), xytext=(2.72, 72),
                ha='center', va='center', fontsize=F - 0.5, color='#c0392b',
                bbox=ann_bbox,
                arrowprops=dict(arrowstyle='->', lw=0.5, color='#c0392b',
                                shrinkA=2, shrinkB=2))

# ---------------------------------------------------------------------------
# axes / legend (inside, upper-left, half-transparent white frame)
# ---------------------------------------------------------------------------
ax.set_xlabel(r'basis drift $\|Q-I\|_F$', labelpad=1.5)
ax.set_ylabel(r'escape fraction $\rho$ (%)', labelpad=1.5)
ax.set_xlim(-0.15, 4.0)
ax.set_ylim(-6, 110)
ax.set_yticks([0, 25, 50, 75, 100])

if not ARGS.compact:
    ax.legend(loc='upper left', frameon=True,
              facecolor='white', edgecolor='#b0b0b0',
              framealpha=0.72, fontsize=F - 0.5,
              handletextpad=0.5, borderpad=0.45, labelspacing=0.32)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout(pad=0.3)

out = ARGS.out
os.makedirs(os.path.dirname(out), exist_ok=True)
# pad_inches must be >= ~0.06: with a small pad and a SHORT design height the
# rotated y-label ("escape fraction rho (%)", a fixed-length vertical string)
# overruns the canvas and silently loses its last characters -- at
# pad_inches=0.02 with --height 1.18 the label degraded to
# 'escape fraction rho (%' and its ink measured x0 = -1.81pt (clipped).
fig.savefig(out, bbox_inches='tight', pad_inches=0.06)
fig.savefig(out.replace('.pdf', '.png'), dpi=300,
            bbox_inches='tight', pad_inches=0.06)
print('wrote %s  (identity %d, removed %d, live %d, width %.2f in, font %.1f pt)'
      % (out, len(identity), len(removed), len(live), ARGS.width, F))
