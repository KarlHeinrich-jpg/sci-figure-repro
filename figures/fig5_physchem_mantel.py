# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Fig. 5  Jiupei physicochemical violins (a-g) + Mantel test (h)."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import LinearSegmentedColormap
from common import COL_A, COL_B, COL_C, violin, save

rng = np.random.default_rng(5)

fig = plt.figure(figsize=(13.5, 9.6))
STAR = '✱'
STAR_KW = dict(fontfamily='DejaVu Sans')

# ------------------------------------------------------------ violin data --
def mix(parts, lo, hi, target=None):
    d = np.concatenate([rng.normal(m, s, n) for m, s, n in parts])
    d = np.clip(d, lo, hi)
    if target is not None:
        d = d - d.mean() + target
    return d

panels = {
    'a': dict(ylabel='Starch(%)', ylim=(25, 50), yticks=[25, 30, 35, 40, 45, 50],
              data=[mix([(34, 1.6, 20), (36.4, 1.9, 24)], 28.3, 44.2, 34.8),
                    mix([(34.6, 2.2, 24), (36.2, 1.6, 20)], 29.0, 41.6, 34.6),
                    mix([(33.6, 2.4, 24), (37.0, 1.7, 24)], 27.9, 45.5, 34.4)],
              sig=('ns', 'ns', 'ns'), ptext=None),
    'b': dict(ylabel='Reducing sugar (mg/g)', ylim=(0, 40),
              yticks=[0, 10, 20, 30, 40],
              data=[mix([(13.2, 3.4, 30), (17.5, 2.6, 14)], 5.5, 32.0, 13.84),
                    mix([(12.2, 3.0, 30), (16.4, 2.4, 14)], 5.0, 25.2, 13.34),
                    mix([(11.2, 3.4, 34), (14.8, 2.6, 16)], 2.2, 25.4, 11.59)],
              sig=(STAR, STAR, 'ns'), ptext='$p$ < 0.05'),
    'c': dict(ylabel='Lactic acid (mg/g)', ylim=(0, 6), yticks=[0, 2, 4, 6],
              data=[mix([(2.3, 1.0, 30), (3.3, 1.1, 14)], 0.5, 5.8, 2.6),
                    mix([(2.5, 1.0, 32), (3.4, 0.9, 12)], 0.7, 5.0, 2.8),
                    mix([(2.3, 1.0, 32), (3.2, 0.9, 12)], 0.8, 4.7, 2.6)],
              sig=('ns', 'ns', 'ns'), ptext=None),
    'd': dict(ylabel='Titratable acidity(mmol/10g)', ylim=(0, 3),
              yticks=[0, 1, 2, 3],
              data=[mix([(0.92, 0.20, 44)], 0.50, 1.45, 0.92),
                    np.concatenate([rng.normal(1.0, 0.28, 38),
                                    rng.uniform(2.0, 2.8, 4)]),
                    mix([(0.75, 0.22, 46)], 0.30, 1.70, 0.79)],
              sig=(STAR, STAR, STAR), ptext='$p$ < 0.05'),
    'e': dict(ylabel='Ethanol (mg/g)', ylim=(-5, 25),
              yticks=[-5, 0, 5, 10, 15, 20, 25],
              data=[mix([(3.6, 2.8, 34), (7.2, 2.6, 10)], -1.2, 17.5, 4.51),
                    np.concatenate([rng.normal(3.8, 3.0, 36),
                                    rng.uniform(17, 21, 5)]),
                    mix([(4.4, 2.8, 36), (8.0, 2.4, 8)], -1.2, 15.2, 5.33)],
              sig=(STAR, 'ns', 'ns'), ptext='$p$<0.05'),
    'f': dict(ylabel='Temperature (℃)', ylim=(20, 50), yticks=[20, 30, 40, 50],
              data=[mix([(36.2, 3.2, 34), (39.0, 2.6, 10)], 27.0, 47.2, 36.6),
                    mix([(35.6, 3.0, 34), (38.4, 2.4, 10)], 28.2, 45.4, 36.2),
                    mix([(36.4, 3.2, 34), (39.4, 2.4, 10)], 29.0, 46.6, 37.0)],
              sig=('ns', 'ns', 'ns'), ptext=None),
    'g': dict(ylabel='Moisture (%)', ylim=(30, 50), yticks=[30, 35, 40, 45, 50],
              data=[mix([(40.0, 2.6, 34), (42.4, 1.8, 10)], 33.0, 46.5, 40.6),
                    mix([(40.8, 2.6, 34), (43.2, 1.8, 10)], 35.0, 47.5, 41.2),
                    mix([(41.4, 2.6, 34), (44.0, 1.8, 10)], 35.0, 48.2, 42.0)],
              sig=('ns', 'ns', 'ns'), ptext=None),
}
pos = {'a': [0.035, 0.68], 'b': [0.36, 0.68], 'c': [0.685, 0.68],
       'd': [0.035, 0.36], 'e': [0.36, 0.36], 'f': [0.685, 0.36],
       'g': [0.035, 0.045]}

for key, p in panels.items():
    x0, y0 = pos[key]
    ax = fig.add_axes([x0, y0, 0.245, 0.24])
    for x, d, c in ((1, p['data'][0], COL_A), (2, p['data'][1], COL_B),
                    (3, p['data'][2], COL_C)):
        violin(ax, x, d, c, width=0.52)
    ax.set_xlim(0.4, 3.6); ax.set_ylim(*p['ylim'])
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(['A-JP', 'B-JP', 'C-JP'],
                                                 fontsize=11, fontweight='bold')
    ax.set_yticks(p['yticks'])
    ax.set_ylabel(p['ylabel'], fontsize=11)
    lo, hi = p['ylim']
    ytop = hi - 0.09 * (hi - lo)
    ysub = hi - 0.17 * (hi - lo)
    s_top, s_ab, s_bc = p['sig']
    # top bracket A–C
    ax.plot([1, 1, 3, 3], [ytop - 0.06 * (hi - lo), ytop, ytop, ytop - 0.06 * (hi - lo)],
            color='k', lw=1.3, clip_on=False)
    ax.text(2, ytop + 0.015 * (hi - lo), s_top, ha='center', va='bottom', fontsize=12, **STAR_KW)
    # sub brackets
    ax.plot([1, 1, 2, 2], [ysub - 0.06 * (hi - lo), ysub, ysub, ysub - 0.06 * (hi - lo)],
            color='k', lw=1.3, clip_on=False)
    ax.text(1.5, ysub + 0.015 * (hi - lo), s_ab, ha='center', va='bottom', fontsize=12, **STAR_KW)
    ax.plot([2, 2, 3, 3], [ysub - 0.06 * (hi - lo), ysub, ysub, ysub - 0.06 * (hi - lo)],
            color='k', lw=1.3, clip_on=False)
    ax.text(2.5, ysub + 0.015 * (hi - lo), s_bc, ha='center', va='bottom', fontsize=12, **STAR_KW)
    if p['ptext']:
        ax.text(1.02, 1.045, p['ptext'], transform=ax.transAxes, fontsize=10,
                ha='right', va='bottom', style='italic')
    fig.text(x0 - 0.008, y0 + 0.245, key, fontsize=14, fontweight='bold')

# ================================================================ (h) ======
axh = fig.add_axes([0.285, 0.015, 0.665, 0.315])
axh.set_xlim(0, 100); axh.set_ylim(0, 100)
axh.axis('off')

envs = ['Temperature', 'Moisture', 'Titratable acidity', 'Starch',
        'Reducing sugar', 'Lactate', 'Ethanol']
corr = {(0, 1): 0.20, (0, 2): 0.06, (0, 3): 0.15, (0, 4): 0.22, (0, 5): 0.02,
        (0, 6): -0.06,
        (1, 2): 0.24, (1, 3): -0.18, (1, 4): 0.05, (1, 5): 0.08, (1, 6): 0.26,
        (2, 3): 0.04, (2, 4): 0.10, (2, 5): 0.03, (2, 6): -0.30,
        (3, 4): 0.20, (3, 5): 0.05, (3, 6): 0.16,
        (4, 5): 0.03, (4, 6): -0.14,
        (5, 6): 0.02}
mantel = [(0.12, 0.30, +1), (0.22, 0.08, +1), (0.24, 0.03, +1),
          (0.13, 0.04, +1), (0.62, 0.01, +1), (0.08, 0.40, +1),
          (-0.20, 0.15, -1)]

rdylbu = plt.get_cmap('RdYlBu_r')
norm_c = mpl.colors.Normalize(vmin=-0.3, vmax=0.3)

cell = 6.4
hx0, hy0 = 33, 50                      # top-left corner of the 7x7 grid
maxr = max(abs(v) for v in corr.values())
for i in range(7):
    yrow = hy0 - i * cell
    axh.add_patch(Rectangle((hx0, yrow - cell), 7 * cell, cell,
                            fill=False, edgecolor='#BBBBBB', lw=0.8))
    for j in range(i + 1, 7):
        r = corr[(i, j)]
        cx = hx0 + j * cell + cell / 2
        cy = yrow - cell / 2
        side = 2.6 + 4.6 * abs(r) / maxr
        axh.add_patch(Rectangle((cx - side / 2, cy - side / 2), side, side,
                                facecolor=rdylbu(norm_c(r)), edgecolor='#666666',
                                lw=0.4))
for j, name in enumerate(envs):
    axh.text(hx0 + j * cell + cell / 2 - 1, hy0 + 2.0, name, rotation=45,
             ha='left', va='bottom', fontsize=8)

score = (9, 13)
for i, (r, pv, sign) in enumerate(mantel):
    yrow = hy0 - i * cell - cell / 2
    x_end = hx0 - 2.0
    color = '#D94A3D' if sign > 0 else '#5B84C4'
    ls = '-' if pv < 0.05 else (0, (4, 2.2))
    lw = 1.0 if abs(r) < 0.1 else (2.0 if abs(r) < 0.3 else 3.6)
    t = np.linspace(0, 1, 60)
    mx, my = (score[0] + x_end) / 2, min(score[1], yrow) - 6
    bx = (1 - t) ** 2 * score[0] + 2 * (1 - t) * t * mx + t ** 2 * x_end
    by = (1 - t) ** 2 * score[1] + 2 * (1 - t) * t * my + t ** 2 * yrow
    axh.plot(bx, by, color=color, ls=ls, lw=lw, alpha=0.95)
axh.scatter([score[0]], [score[1]], s=90, marker='o', facecolor='#7A9CC6',
            edgecolor='#345A8C', lw=1.2, zorder=5)
axh.text(score[0] - 2.5, score[1] - 0.5, 'Score', ha='right', va='center',
         fontsize=10, fontweight='bold', zorder=6)

lx = 78
axh.text(lx, 95, "Mantel's r.sign", fontsize=9, fontweight='bold', va='center')
axh.plot([lx, lx + 3.5], [91, 91], color='#5B84C4', lw=1.6)
axh.text(lx + 4.5, 91, 'Negative', fontsize=8, va='center')
axh.plot([lx, lx + 3.5], [87.5, 87.5], color='#D94A3D', lw=1.6)
axh.text(lx + 4.5, 87.5, 'Positive', fontsize=8, va='center')

axh.text(lx, 82.5, "Mantel's p", fontsize=9, fontweight='bold', va='center')
axh.plot([lx, lx + 3.5], [78.5, 78.5], color='#555555', lw=1.4)
axh.text(lx + 4.5, 78.5, '$p$ < 0.05', fontsize=8, va='center')
axh.plot([lx, lx + 3.5], [75, 75], color='#555555', lw=1.4, ls=(0, (4, 2)))
axh.text(lx + 4.5, 75, '$p$ >= 0.05', fontsize=8, va='center')

axh.text(lx, 70, "Mantel's r.abs", fontsize=9, fontweight='bold', va='center')
axh.plot([lx, lx + 3.5], [66, 66], color='#555555', lw=1.0)
axh.text(lx + 4.5, 66, '< 0.1', fontsize=8, va='center')
axh.plot([lx, lx + 3.5], [62.5, 62.5], color='#555555', lw=2.2)
axh.text(lx + 4.5, 62.5, '0.1 – 0.3', fontsize=8, va='center')
axh.plot([lx, lx + 3.5], [59, 59], color='#555555', lw=3.6)
axh.text(lx + 4.5, 59, '>= 0.3', fontsize=8, va='center')

axh.text(lx, 53.5, "spearman's r", fontsize=9, fontweight='bold', va='center')
grad = np.linspace(1, 0, 256).reshape(-1, 1)
axh.imshow(grad, extent=[lx + 1, lx + 4.5, 26, 50], aspect='auto',
           cmap=rdylbu, norm=mpl.colors.Normalize(-0.3, 0.3), zorder=1)
for tv in (0.2, 0.1, 0.0, -0.1, -0.2):
    yy = 26 + (tv + 0.3) / 0.6 * 24
    axh.plot([lx + 4.5, lx + 5.2], [yy, yy], color='#333333', lw=0.6)
    axh.text(lx + 5.8, yy, f'{tv:.1f}', fontsize=7.5, va='center')

fig.text(0.272, 0.325, 'h', fontsize=14, fontweight='bold')
save(fig, 'fig5')
