"""Compact A2 vs A6 paired-difference dumbbell (ICASSP single column).

One row per training seed.  The visual is the sign flip, not mean+/-std.

    python make_a2_a6_paired_diff.py
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = r'G:\material\天枢\mamba'

plt.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif',
                   'Nimbus Roman', 'serif'],
    'font.size': 6.5,
    'axes.linewidth': 0.6,
    'axes.labelsize': 6.5,
    'xtick.labelsize': 6.0,
    'ytick.labelsize': 6.5,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': False,
    'lines.linewidth': 0.85,
})

# Held-out mean Dice; deltas are Dice points (x100), matching the manuscript.
seeds = [
    dict(name='123', a2=0.9017, a6=0.8938, delta=+0.79),
    dict(name='456', a2=0.9016, a6=0.8931, delta=+0.85),
    dict(name='789', a2=0.8936, a6=0.9021, delta=-0.85),
]

NAVY = '#1f4e79'
RED = '#c0392b'

fig, ax = plt.subplots(figsize=(3.39, 1.18))
ys = [2, 1, 0]

ax.set_axisbelow(True)
ax.grid(axis='x', linestyle='--', linewidth=0.35, color='#c9c9c9', alpha=0.75)

for y, r in zip(ys, seeds):
    color = NAVY if r['delta'] > 0 else RED
    lo, hi = min(r['a2'], r['a6']), max(r['a2'], r['a6'])
    ax.plot([lo, hi], [y, y], color=color, lw=1.35,
            solid_capstyle='round', zorder=2)
    ax.scatter([r['a2']], [y], s=28, marker='o', color=NAVY, zorder=3,
               edgecolors='white', linewidths=0.4)
    ax.scatter([r['a6']], [y], s=28, marker='s', color=RED, zorder=3,
               edgecolors='white', linewidths=0.4)
    sign = '+' if r['delta'] > 0 else ''
    ax.text(hi + 0.00055, y, r'$%s%.2f$' % (sign, r['delta']),
            ha='left', va='center', fontsize=6.2, color=color)

ax.set_yticks(ys)
ax.set_yticklabels([r'seed %s' % r['name'] for r in seeds])
ax.set_xlabel('held-out Dice', labelpad=1.8)
ax.set_xlim(0.8916, 0.9068)
ax.set_ylim(-0.62, 3.05)
ax.set_xticks([0.894, 0.898, 0.902])
ax.tick_params(axis='y', length=0, pad=2.2)
ax.tick_params(axis='x', length=2.5)

for spine in ('top', 'right', 'left'):
    ax.spines[spine].set_visible(False)

legend = [
    Line2D([0], [0], marker='o', color='none', markerfacecolor=NAVY,
           markeredgecolor='none', markersize=5.0, label=r'A2 (frozen $Q{=}I$)'),
    Line2D([0], [0], marker='s', color='none', markerfacecolor=RED,
           markeredgecolor='none', markersize=4.6, label=r'A6 (learned $Q$)'),
]
ax.legend(handles=legend, loc='upper left', frameon=True,
          facecolor='white', edgecolor='#b0b0b0', framealpha=0.82,
          fontsize=5.8, handletextpad=0.35, borderpad=0.35,
          labelspacing=0.28, ncol=2, bbox_to_anchor=(0.0, 1.02),
          columnspacing=1.1)

fig.tight_layout(pad=0.28)
out = os.path.join(ROOT, 'figs', 'a2_a6_paired_diff.pdf')
fig.savefig(out, bbox_inches='tight', pad_inches=0.02)
fig.savefig(out.replace('.pdf', '.png'), dpi=300,
            bbox_inches='tight', pad_inches=0.02)
print('wrote', out)
