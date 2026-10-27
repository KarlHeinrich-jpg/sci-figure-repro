# -*- coding: utf-8 -*-
"""Shared style and helpers for the figure reproductions.

All scripts in ``figures/`` import from this module. Keeping the styling
constants in one place makes it easy to re-use the templates for a new paper:
change the palette / rcParams here and every figure follows.
"""
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(ROOT, 'outputs')

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': 1.4,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'xtick.major.width': 1.4, 'ytick.major.width': 1.4,
    'xtick.minor.size': 3, 'ytick.minor.size': 3,
    'axes.labelsize': 13,
    'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'legend.frameon': True,
    'savefig.dpi': 300,
})

# three quality grades (A blue, B red, C green) used by Figs 2/4/5
COL_A = '#4472C4'
COL_B = '#E8332B'
COL_C = '#62BB4D'
# lighter palette for the PCA panels (Fig 6c / 6f)
PCA_A = '#6BBF59'
PCA_B = '#94BDE2'
PCA_C = '#E97D77'
# LEfSe enriched-class colours (Fig 6a / 6d)
LEFSE_C = '#E5C31A'
LEFSE_A = '#4EA26B'
LEFSE_B = '#92C6E8'


def save(fig, name):
    """Write ``outputs/<name>.png`` at 300 dpi and close the figure."""
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, f'{name}.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'saved {path}')


# ---------------------------------------------------------------- violin ----
def violin(ax, xc, data, color, width=0.55, edge=None, alpha=0.8):
    """Draw one KDE violin centred at ``xc`` with dashed inner quartiles."""
    data = np.asarray(data, float)
    lo, hi = data.min(), data.max()
    pad = 0.04 * (hi - lo + 1e-9)
    grid = np.linspace(lo - pad, hi + pad, 240)
    try:
        dens = gaussian_kde(data, bw_method=0.28)(grid)
    except Exception:
        dens = np.ones_like(grid)
    dens = dens / dens.max() * width / 2
    edge = edge or color
    ax.fill_betweenx(grid, xc - dens, xc + dens, color=color, alpha=alpha, lw=0)
    ax.plot(xc - dens, grid, color=edge, lw=1.8)
    ax.plot(xc + dens, grid, color=edge, lw=1.8)
    q1, med, q3 = np.percentile(data, [25, 50, 75])
    for q, ls in ((q1, (0, (1, 1))), (med, (0, (4, 2))), (q3, (0, (1, 1)))):
        ax.plot([xc - 0.30 * width / 2, xc + 0.30 * width / 2], [q, q],
                color=edge, lw=1.1, ls=ls, alpha=0.9)


def bracket(ax, x1, x2, y, text, dy=0.0, lw=1.4, fs=12, color='k', star=False):
    """Significance bracket: horizontal bar with two short downward ticks."""
    ax.plot([x1, x1, x2, x2], [y - 0.35, y, y, y - 0.35], color=color, lw=lw,
            solid_capstyle='butt', clip_on=False)
    ax.text((x1 + x2) / 2, y + 0.12 + dy, text, ha='center', va='bottom',
            fontsize=fs, color=color, fontweight='bold' if star else 'normal')


def sig_ns_ns(ax, ytop, ysub, xs, yspan, label='ns'):
    """Convenience wrapper: top A-C bracket plus A-B and B-C brackets."""
    a, b, c = xs
    bracket(ax, a, c, ytop, label)
    bracket(ax, a, b, ysub, label)
    bracket(ax, b, c, ysub, label)
