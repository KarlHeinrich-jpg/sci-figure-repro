# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Fig. 3  (a) OPLS-DA score plot; (b) VIP bar chart."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch
from common import save

rng = np.random.default_rng(7)

fig = plt.figure(figsize=(13.2, 5.0))

# ======================================================== (a) OPLS-DA ======
ax = fig.add_axes([0.045, 0.17, 0.40, 0.72])

def cluster(n, cx, cy, th, s1, s2):
    t = rng.normal(0, 1, n)
    u = rng.normal(0, 1, n)
    x = cx + t * s1 * np.cos(th) - u * s2 * np.sin(th)
    y = cy + t * s1 * np.sin(th) + u * s2 * np.cos(th)
    return x, y

groups = {
    'A': dict(n=40, cx=-4.0, cy=0.4, th=np.deg2rad(78),  s1=4.6, s2=2.4,
              pt='#D62B2B', el='#E87070'),
    'B': dict(n=40, cx=0.6, cy=-3.6, th=np.deg2rad(-48), s1=4.4, s2=2.3,
              pt='#3E8FD0', el='#A9CCE9'),
    'C': dict(n=44, cx=5.2, cy=-2.4, th=np.deg2rad(-65), s1=4.7, s2=2.4,
              pt='#57B336', el='#B7DC9C'),
}
for g, d in groups.items():
    x, y = cluster(d['n'], d['cx'], d['cy'], d['th'], d['s1'], d['s2'])
    ax.add_patch(Ellipse((d['cx'], d['cy']), 2.9 * d['s1'], 2.9 * d['s2'],
                         angle=np.rad2deg(d['th']), facecolor=d['el'],
                         alpha=0.55, lw=0, zorder=1))
    ax.scatter(x, y, s=62, color=d['pt'], zorder=3)

ax.set_xlim(-10.5, 10.5); ax.set_ylim(-10.5, 8.0)
ax.set_xticks([-10, -5, 0, 5, 10]); ax.set_yticks([-10, -5, 0, 5])
ax.set_xlabel('t1 [40.23%]', fontsize=13)
ax.set_ylabel('t2 [31.32%]', fontsize=13)
ax.set_title('OPLS-DA', fontsize=15, fontweight='bold', pad=8)
# legend below the axes
ax.legend(handles=[plt.Line2D([], [], marker='o', ls='', color=groups[g]['pt'],
                             markersize=10, label=g) for g in 'ABC'],
          title='class', loc='upper center', bbox_to_anchor=(0.5, -0.13),
          ncol=4, fontsize=11, title_fontproperties={'weight': 'bold', 'size': 12},
          frameon=False, columnspacing=1.0, handletextpad=0.3)

# ============================================================== (b) VIP ====
axb = fig.add_axes([0.60, 0.13, 0.32, 0.76])
comp = ['Ethyl linoleate', 'Ethyl valerate', 'Ethyl oleate', 'Hexanoic acid',
        'Isoamyl alcohol', 'Acetal', 'Ethyl butyrate', 'Ethyl caprylate',
        'Ethyl tetradecanoate', 'Heptanoic acid', '1,2-Propanediol', 'Furfural',
        'Pentanoic acid', '2-Phenylethanol', 'Methanol', 'Acetaldehyde',
        'Diethyl succinate', 'Isobutyraldehyde', 'Ethyl ether',
        'Tetramethylpyrazine', 'n-Propanol', 'Benzaldehyde', 'Ethyl acetate',
        'n-Propionic acid', '2,3-Butanediol', 'Acetic acid']
vip = np.array([0.96, 0.97, 0.985, 1.00, 1.02, 1.04, 1.065, 1.09, 1.10, 1.12,
                1.14, 1.16, 1.17, 1.19, 1.205, 1.225, 1.24, 1.255, 1.27, 1.29,
                1.305, 1.33, 1.35, 1.37, 1.40, 1.42])
ypos = np.arange(len(comp))[::-1]
axb.barh(ypos, vip, height=0.72, color='#4C7CC0', zorder=2)
axb.axvline(1.0, color='red', lw=2.2, zorder=3)
axb.set_yticks(ypos); axb.set_yticklabels(comp, fontsize=9.5)
axb.set_xlim(0, 1.55); axb.set_xticks(np.arange(0, 1.6, 0.2))
axb.set_xlabel('VIP', fontsize=14)
axb.legend(handles=[Patch(facecolor='#4C7CC0', label='VIP')],
           loc='upper right', bbox_to_anchor=(1.0, 1.09), fontsize=10,
           handlelength=1.5, handleheight=0.9, edgecolor='k')

fig.text(0.012, 0.94, 'a', fontsize=16, fontweight='bold')
fig.text(0.575, 0.94, 'b', fontsize=16, fontweight='bold')
save(fig, 'fig3')
