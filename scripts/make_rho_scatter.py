"""Dump every wavelet site of every checkpoint, then plot drift against rho.

The point of the figure: ||Q-I||_F (how far the basis moved) carries no
information about whether the move did anything.  Sites whose mixer is dense
sit exactly on rho = 0 no matter how far right they are; only the grouped
mixers rise off the axis.

    python make_rho_scatter.py            # writes absorption_sites.csv + the pdf
"""
import csv
import os

import matplotlib
matplotlib.use('Agg')
# TrueType embedding for IEEE PDF inspection (no Type 3 bitmaps).
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import torch

import verify_absorption as va

ROOT = r'G:\material\天枢\mamba'
LOGS = os.path.join(ROOT, 'logs')

# (arm, label, checkpoint) -- highest-Dice checkpoint of each run, the same
# ten networks the paper audits.  A4 seed-123 can be merged from
# lowo/analysis/a4_sites.csv when the full checkpoint is not on disk.
CKPTS = [
    ('A1', 'fixed, g1',
     rf'{LOGS}\segmamba_haar_so8_dicece_ds\segmamba_haar_fixed\model\best_model_0.9268.pt'),
    ('A2', 'fixed, g8',
     rf'{ROOT}\test\segmamba_haar_fixed_g8\model\best_model_0.9285.pt'),
    ('A3', 'theta, g1',
     rf'{LOGS}\segmamba_haar_dicece_ds\segmamba_haar_theta\model\best_model_0.9251.pt'),
    ('A4', 'theta, g8',
     rf'{LOGS}\segmamba_haar_dicece_ds\segmamba_haar_theta_g8_s123_s123\model\best_model_0.9224.pt'),
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

rows = []
for arm, label, path in CKPTS:
    if not os.path.exists(path):
        print(f'!! missing {arm}: {path}')
        continue
    sd = va.load_state_dict(path)
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

if not any(r['arm'] == 'A4' for r in rows):
    a4_csv = os.path.join(ROOT, 'lowo', 'analysis', 'a4_sites.csv')
    if os.path.exists(a4_csv):
        with open(a4_csv, newline='') as fh:
            for r in csv.DictReader(fh):
                if r['arm'] != 'A4-123':
                    continue
                rows.append(dict(arm='A4', label='theta, g8',
                                 site=r['site'], role=r['role'],
                                 groups=int(r['groups']),
                                 drift=float(r['drift']),
                                 rho=float(r['rho']),
                                 fold_err=float(r['fold_err'])))
        print('A4: 16 sites (from lowo/analysis/a4_sites.csv, seed 123)')

with open(os.path.join(ROOT, 'absorption_sites.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

n_inert = sum(1 for r in rows if r['rho'] < 1e-12)
print(f'\ntotal sites {len(rows)}   inert (rho=0) {n_inert}   '
      f'max fold err {max(r["fold_err"] for r in rows):.2e}')
print(f'max drift among inert sites: '
      f'{max(r["drift"] for r in rows if r["rho"] < 1e-12):.3f}')

# ------------------------------------------------------------------ figure
# v3 (2026-09-09, kept in sync with replot_rho_scatter.py) --
# publication-quality single-column (3.39 in) figure:
#   * legend inside the plot (upper-left clear region), half-transparent
#     white frame so it never overflows the canvas or eats into the data
#   * annotations decoupled via explicit xytext + arrowstyle '->' with a
#     translucent white bbox, so text never collides with points or grid
#   * serif (Times New Roman style) typeface, inward ticks, faint dashed
#     horizontal gridlines, classic navy / dark-red palette
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif',
                   'Nimbus Roman', 'serif'],
    'font.size': 6.5,
    'axes.linewidth': 0.6,
    'axes.labelsize': 6.5,
    'xtick.labelsize': 6.0,
    'ytick.labelsize': 6.0,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'lines.linewidth': 0.7,
})
fig, ax = plt.subplots(figsize=(3.39, 1.55))

inert = [r for r in rows if r['rho'] < 1e-12]
live = [r for r in rows if r['rho'] >= 1e-12]

ax.set_axisbelow(True)
ax.grid(axis='y', linestyle='--', linewidth=0.35, color='#c9c9c9', alpha=0.7)

ax.scatter([r['drift'] for r in inert], [0] * len(inert),
           s=13, marker='o', facecolors='none', edgecolors='#1f4e79',
           linewidths=0.8, zorder=3,
           label=fr'$\rho=0$  ($n={len(inert)}$)')
ax.scatter([r['drift'] for r in live], [r['rho'] * 100 for r in live],
           s=15, marker='^', color='#c0392b', edgecolors='none', zorder=3,
           label=fr'$\rho>0$  ($n={len(live)}$)')

ann_bbox = dict(boxstyle='round,pad=0.18', facecolor='white',
                edgecolor='none', alpha=0.65)
ax.annotate('A5 stem, inert', xy=(2.84, 0), xytext=(2.42, 18),
            ha='center', va='bottom', fontsize=5.5, color='#1f4e79',
            bbox=ann_bbox,
            arrowprops=dict(arrowstyle='->', lw=0.5, color='#1f4e79',
                            shrinkA=2, shrinkB=2))
ax.annotate('C1 stem, inert', xy=(3.53, 0), xytext=(3.53, 34),
            ha='right', va='bottom', fontsize=5.5, color='#1f4e79',
            bbox=ann_bbox,
            arrowprops=dict(arrowstyle='->', lw=0.5, color='#1f4e79',
                            shrinkA=2, shrinkB=2))
ax.annotate(fr'A6 stem, $\rho=88$%',
            xy=(3.53, 88), xytext=(2.98, 72),
            ha='center', va='center', fontsize=5.5, color='#c0392b',
            bbox=ann_bbox,
            arrowprops=dict(arrowstyle='->', lw=0.5, color='#c0392b',
                            shrinkA=2, shrinkB=2))

ax.set_xlabel(r'basis drift $\|Q-I\|_F$', labelpad=1.5)
ax.set_ylabel(r'escape fraction $\rho$ (%)', labelpad=1.5)
ax.set_xlim(-0.15, 4.0)
ax.set_ylim(-6, 110)
ax.set_yticks([0, 25, 50, 75, 100])
ax.legend(loc='upper left', frameon=True,
          facecolor='white', edgecolor='#b0b0b0',
          framealpha=0.72, fontsize=6.0,
          handletextpad=0.5, borderpad=0.45, labelspacing=0.35)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout(pad=0.3)

out = os.path.join(ROOT, 'figs', 'drift_vs_rho.pdf')
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.rcParams['text.usetex'] = False
fig.savefig(out, bbox_inches='tight', pad_inches=0.02)
fig.savefig(out.replace('.pdf', '.png'), dpi=300,
            bbox_inches='tight', pad_inches=0.02)
print(f'wrote {out}')
