# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Fig. 2  (a) QDA radar chart; (b) total acids; (c) total esters."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon as MplPolygon
from common import COL_A, COL_B, COL_C, violin, bracket, save

rng = np.random.default_rng(42)

fig = plt.figure(figsize=(12.6, 4.2))

# ============================================================ (a) radar ====
ax = fig.add_axes([0.015, 0.04, 0.30, 0.92])
ax.set_xlim(-1.65, 1.65); ax.set_ylim(-1.35, 1.55)
ax.axis('off')
ax.set_aspect('equal')

attrs = ['Grain aroma', 'Sour taste', 'Mellowness', 'Durability', 'Cleanliness']
ang = np.deg2rad(90 + np.arange(5) * -72)          # top, then clockwise
unit = np.c_[np.cos(ang), np.sin(ang)]

def ring_pts(r):
    return unit * r

vals = {'A': [7.5, 7.4, 6.0, 6.9, 8.6],
        'B': [7.0, 6.8, 5.4, 5.9, 7.7],
        'C': [4.5, 4.1, 4.0, 3.6, 4.4]}
cols = {'A': COL_A, 'B': COL_B, 'C': COL_C}

# grid rings (every 1 unit, pentagons) + spokes
for r in np.arange(1, 11) / 10 * 1.15:
    p = MplPolygon(ring_pts(r), closed=True, fill=False,
                   edgecolor='#999999', lw=0.6, zorder=1)
    ax.add_patch(p)
for i in range(5):
    ax.plot([0, unit[i, 0] * 1.15], [0, unit[i, 1] * 1.15],
            color='k', lw=1.6, zorder=2)
outer = MplPolygon(ring_pts(1.15), closed=True, fill=False,
                   edgecolor='k', lw=2.0, zorder=3)
ax.add_patch(outer)

for g in ('C', 'B', 'A'):
    v = np.array(vals[g]) / 10 * 1.15
    pts = unit * v[:, None]
    ax.add_patch(MplPolygon(pts, closed=True, facecolor=cols[g],
                            alpha=0.45 if g != 'C' else 0.55, lw=0, zorder=4))
    ax.add_patch(MplPolygon(pts, closed=True, fill=False, edgecolor=cols[g],
                            lw=2.2, zorder=5))

# axis labels with *** and legend
lab_pos = [(-0.10, 1.28), (1.42, 0.78), (1.30, -1.02), (-1.55, -1.02), (-1.62, 0.78)]
ha_list = ['center', 'left', 'left', 'right', 'right']
star_pos = [(0, 1.42), (1.30, 0.95), (1.05, -1.18), (-1.30, -1.18), (-1.35, 0.95)]
star_ha = ['center', 'right', 'center', 'center', 'left']
for i, name in enumerate(attrs):
    ax.text(*lab_pos[i], name, fontsize=10.5, ha=ha_list[i], va='center')
    ax.text(*star_pos[i], '***', fontsize=11, fontweight='bold',
            ha=star_ha[i], va='center')
handles = [Patch(facecolor=cols[g], edgecolor='k', lw=0.5, label=g) for g in 'ABC']
ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.92, 1.06),
          fontsize=9, handlelength=1.2, handleheight=0.9, edgecolor='#888888')

# ==================================================== (b) total acids ======
axb = fig.add_axes([0.40, 0.14, 0.17, 0.74])
dataA = np.concatenate([rng.normal(3.05, 0.55, 26), rng.normal(4.15, 0.75, 16)])
dataA = np.clip(dataA, 1.5, 5.75); dataA = dataA - dataA.mean() + 3.352
dataB = np.concatenate([rng.normal(2.65, 0.75, 22), rng.normal(4.75, 0.95, 20)])
dataB = np.clip(dataB, 1.5, 7.15); dataB = dataB - dataB.mean() + 3.679
dataC = np.concatenate([rng.normal(2.62, 0.42, 43), rng.normal(3.55, 0.22, 6)])
dataC = np.clip(dataC, 1.5, 4.0); dataC = dataC - dataC.mean() + 2.684

for x, d, c in ((1, dataA, COL_A), (2, dataB, COL_B), (3, dataC, COL_C)):
    violin(axb, x, d, c, width=0.55)
axb.set_xlim(0.4, 3.6); axb.set_ylim(0, 8)
axb.set_xticks([1, 2, 3]); axb.set_xticklabels(['A', 'B', 'C'], fontsize=13)
axb.set_yticks([0, 2, 4, 6, 8])
axb.set_ylabel('Total acids (g/L)')
bracket(axb, 1, 3, 7.35, '***', star=True)
bracket(axb, 1, 2, 6.55, 'ns')
bracket(axb, 2, 3, 6.55, '***', star=True)
axb.text(3.55, 7.75, '$p$< 0.001***', fontsize=9.5, ha='right', va='center')

# =================================================== (c) total esters ======
axc = fig.add_axes([0.64, 0.14, 0.17, 0.74])
dataA = np.concatenate([rng.normal(6.3, 1.0, 26), rng.normal(9.3, 1.3, 16)])
dataA = np.clip(dataA, 4.0, 12.0); dataA = dataA - dataA.mean() + 7.768
dataB = np.concatenate([rng.normal(6.2, 1.4, 24), rng.normal(10.6, 2.0, 18)])
dataB = np.clip(dataB, 2.0, 16.0); dataB = dataB - dataB.mean() + 8.360
dataC = np.clip(rng.normal(6.2, 1.11, 49), 3.4, 10.6)
dataC = dataC - dataC.mean() + 6.204

for x, d, c in ((1, dataA, COL_A), (2, dataB, COL_B), (3, dataC, COL_C)):
    violin(axc, x, d, c, width=0.55)
axc.set_xlim(0.4, 3.6); axc.set_ylim(0, 20)
axc.set_xticks([1, 2, 3]); axc.set_xticklabels(['A', 'B', 'C'], fontsize=13)
axc.set_yticks([0, 5, 10, 15, 20])
axc.set_ylabel('Total esters (g/L)')
bracket(axc, 1, 3, 18.3, '***', star=True)
bracket(axc, 1, 2, 16.6, 'ns')
bracket(axc, 2, 3, 16.6, '***', star=True)
axc.text(3.55, 19.35, '$p$< 0.001***', fontsize=9.5, ha='right', va='center')

fig.text(0.008, 0.97, 'a', fontsize=15, fontweight='bold')
fig.text(0.395, 0.97, 'b', fontsize=15, fontweight='bold')
fig.text(0.628, 0.97, 'c', fontsize=15, fontweight='bold')
save(fig, 'fig2')
