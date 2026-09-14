"""Fig. 1 -- the absorption fold and why the mixer class decides it.

ICASSP 2027 (v13): printed at ``\\columnwidth`` (3.386 in) in a one-column
``figure``.  The canvas is 3.39 in wide and cropped tight to ~2.7 in, so
including it at ``\\columnwidth`` scales the type UP by ~1.25x -- point sizes
here (~7.5) therefore print at ~9.4 pt, meeting the ICASSP "no smaller than
nine point" rule.  (A figure shown at 0.74\columnwidth prints 0.74x smaller.)

    python make_fig1_absorption.py --out ../v13/figs/fig1_absorption.pdf

Visual language (same as drift_vs_rho / a2_a6_paired_diff):
    purple = learned factor Q ; blue = fixed Haar bank ; gray = 1x1 mixer
    coral  = dense mixer  -> absorbed ; teal = grouped mixer -> load-bearing
Light fills + dark text only (no dark surfaces).
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = r'G:\material\天枢\mamba'

_ap = argparse.ArgumentParser()
_ap.add_argument('--out', default=os.path.join(ROOT, 'figs',
                                               'fig1_absorption.pdf'))
_ap.add_argument('--fontsize', type=float, default=7.5,
                 help='base point size; printed size = this x '
                      '(display_width / natural_width)')
ARGS = _ap.parse_args()
F = ARGS.fontsize

plt.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif',
                   'Nimbus Roman', 'serif'],
    'mathtext.fontset': 'dejavuserif',
    'font.size': F,
})

PURPLE_E, PURPLE_F, PURPLE_T = '#534AB7', '#EEEDFE', '#3C3489'
BLUE_E, BLUE_F, BLUE_T = '#185FA5', '#E6F1FB', '#0C447C'
GRAY_E, GRAY_F, GRAY_T = '#5F5E5A', '#F1EFE8', '#2C2C2A'
CORAL_E, CORAL_F, CORAL_T = '#D85A30', '#FAECE7', '#993C1D'
TEAL_E, TEAL_F, TEAL_T = '#0F6E56', '#E1F5EE', '#085041'

fig, ax = plt.subplots(figsize=(3.39, 1.88))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# mathtext's va='center' centres on the math axis, which sits high for
# descender-free formulas; nudge down so the ink is visually centred.
TEXT_DY = -0.9


def node(x, y, w, h, text, edge, face, tcol, fontsize=None):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0,rounding_size=1.6',
        linewidth=0.7, edgecolor=edge, facecolor=face, zorder=2))
    ax.text(x + w / 2, y + h / 2 + TEXT_DY, text, ha='center', va='center',
            fontsize=F if fontsize is None else fontsize, color=tcol, zorder=3)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=5.0,
        linewidth=0.7, color=GRAY_E, zorder=2, shrinkA=0.0, shrinkB=0.0))


H = 14.5
YA, YB = 72.5, 45.0          # bottom edge of each lane's boxes
YCA, YCB = YA + H / 2, YB + H / 2

# ---------------------------------------------------------------- lane A: learned
node(1, YA, 29, H, r'$Q\,\mathcal{K}_{\mathrm{haar}}$', PURPLE_E, PURPLE_F, PURPLE_T)
arrow(30.6, YCA, 39.4, YCA)
node(40, YA, 29, H, r'$W$', GRAY_E, GRAY_F, GRAY_T)
arrow(69.6, YCA, 76.4, YCA)
ax.text(78.5, YCA, r'$y$', ha='center', va='center', fontsize=F, color=GRAY_T)

# ---------------------------------------------------------------- lane B: fixed
node(1, YB, 29, H, r'$\mathcal{K}_{\mathrm{haar}}$', BLUE_E, BLUE_F, BLUE_T)
arrow(30.6, YCB, 39.4, YCB)
node(40, YB, 29, H, r"$W'$", GRAY_E, GRAY_F, GRAY_T)
arrow(69.6, YCB, 76.4, YCB)
ax.text(78.5, YCB, r'$y$', ha='center', va='center', fontsize=F, color=GRAY_T)

# ---------------------------------------------------------------- equivalence bracket
XB = 80.0
ax.plot([XB, XB], [YCB - 3.2, YCA + 3.2], color=GRAY_E, linewidth=0.7, zorder=2)
ax.plot([XB, XB + 2.4], [YCA, YCA], color=GRAY_E, linewidth=0.7, zorder=2)
ax.plot([XB, XB + 2.4], [YCB, YCB], color=GRAY_E, linewidth=0.7, zorder=2)
ax.text(XB + 3.2, (YCA + YCB) / 2, 'identical\noutput',
        ha='left', va='center', fontsize=F - 0.5, color=GRAY_E,
        linespacing=1.3)

# ---------------------------------------------------------------- fold caption
ax.text(40, 65.5, r'fold $Q$ into the adjacent $1{\times}1$ mixer',
        ha='center', va='center', fontsize=F - 0.3, color=GRAY_T)

# ------------------------------------------------- placements (compact symbols)
# The full three-way statement (W'=WR, R^T W, R^T W R) is in the caption; the
# figure keeps only the symbol key so the line fits at >=9 pt printed.
ax.text(50, 34.5, r"$W'\!=\!WQ$    $\cdot$    $Q^{\top}W$    $\cdot$    "
        r"$Q^{\top}WQ$",
        ha='center', va='center', fontsize=F - 0.5, color=GRAY_T)

# ---------------------------------------------------------------- verdicts
ax.add_patch(FancyBboxPatch((0.5, 16.0), 99, 13.5,
                            boxstyle='round,pad=0,rounding_size=1.6',
                            linewidth=0.7, edgecolor=CORAL_E,
                            facecolor=CORAL_F, zorder=2, linestyle=(0, (4, 3))))
ax.text(3.5, 22.75 + TEXT_DY,
        r"$\mathrm{groups}{=}1$ (dense): $\rho\!\equiv\!0$ at all $71$ sites",
        ha='left', va='center', fontsize=F - 0.3, color=CORAL_T, zorder=3)

ax.add_patch(FancyBboxPatch((0.5, 1.0), 99, 13.5,
                            boxstyle='round,pad=0,rounding_size=1.6',
                            linewidth=0.7, edgecolor=TEAL_E,
                            facecolor=TEAL_F, zorder=2))
ax.text(3.5, 7.75 + TEXT_DY,
        r"$\mathrm{groups}{=}8$ (grouped): $\rho\!>\!0$ at all $72$ sites "
        r"with $Q\!\neq\!I$",
        ha='left', va='center', fontsize=F - 0.3, color=TEAL_T, zorder=3)

out = ARGS.out
os.makedirs(os.path.dirname(os.path.abspath(out)) or '.', exist_ok=True)
fig.savefig(out, bbox_inches='tight', pad_inches=0.015)
fig.savefig(out.replace('.pdf', '.png'), dpi=400, bbox_inches='tight',
            pad_inches=0.015)
print('wrote', out, '(design fontsize %.2f pt)' % F)
