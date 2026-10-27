# -*- coding: utf-8 -*-
"""Fig. 4  Lasso / regression / ROC / SHAP panels (a-h)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import norm
from common import save

rng = np.random.default_rng(11)

fig = plt.figure(figsize=(15.5, 8.6))

FEAT15 = ['Isobutyraldehyde', 'Ethyl lactate', 'Acetal', 'Isobutanol',
          'Reducing sugar', 'Diethyl succinate', 'Ethyl valerate', 'Acetic acid',
          'Ethyl decanoate', 'Sec-butanol', 'Starch', '2,3,5- Trimethylpyrazine',
          'Isoamyl acetate', 'Octanoic acid', 'Ethyl linoleate']

# ================================================================= (a) =====
ax = fig.add_axes([0.045, 0.55, 0.185, 0.36])
x = np.linspace(-3.3, -0.5, 200)
xopt = np.log10(0.06)
nfeat = 52
sel = np.zeros(nfeat, bool); sel[:15] = True
v0 = np.where(sel, rng.uniform(0.03, 0.24, nfeat), rng.uniform(-0.05, 0.10, nfeat))
v0[:4] = [0.24, -0.62, -0.45, 0.18]          # a few eye-catching paths
final = np.where(sel, v0 * rng.uniform(0.05, 0.45, nfeat), 0.0)
zx = rng.uniform(-1.6, -0.7, nfeat)           # where unselected hit zero
cols = np.array(mpl.cm.tab20(rng.permutation(20)))
for i in range(nfeat):
    if sel[i]:
        shrink = 1.0 / (1.0 + np.exp((x - (xopt - 0.05)) * 6.0))
        c = final[i] + (v0[i] - final[i]) * shrink
        c = c + 0.015 * np.sin(3 * x + i)
    else:
        zt = zx[i]
        c = np.where(x < zt, v0[i] * (x - x[0]) / (zt - x[0]), 0.0)
    ax.plot(x, c, lw=1.4, color=cols[i % 20], zorder=2)
ax.axvline(xopt, color='k', ls='--', lw=1.8, zorder=3,
           label='optimal lambda')
ax.set_xlim(-3.3, -0.5); ax.set_ylim(-0.68, 0.30)
ax.set_xticks([-3.0, -2.5, -2.0, -1.5, -1.0, -0.5])
ax.set_yticks([-0.6, -0.4, -0.2, 0.0, 0.2])
ax.set_xlabel('Log(lambda)')
ax.set_ylabel('Coefficients')
ax.legend(loc='upper right', fontsize=8, handlelength=1.6)

# ================================================================= (b) =====
ax = fig.add_axes([0.27, 0.55, 0.185, 0.36])
mse = 1.3 + 6.7 * np.exp(-3.0 * (x + 3.3))
err = 4.8 + 47.0 * np.exp(-1.55 * (x + 3.3))
ax.plot(x, mse, 'k-', lw=1.8, label='average error')
ax.plot(x, err, color='#E8D44D', ls=':', lw=1.8, label='lambda error')
for lam in np.linspace(-2.35, -0.55, 30):
    for f in range(10):
        ax.plot(lam, 1.3 + abs(rng.normal(0, 0.45)), '.', ms=2.5,
                color=mpl.cm.tab10(f % 10))
ax.set_xlim(-3.3, -0.5); ax.set_ylim(-2, 55)
ax.set_xticks([-3.0, -2.5, -2.0, -1.5, -1.0, -0.5])
ax.set_yticks([0, 10, 20, 30, 40, 50])
ax.set_xlabel('Log(lambda)')
ax.set_ylabel('Mean Squared Error')
ax.legend(loc='upper right', fontsize=8)

# ================================================================= (c) =====
ax = fig.add_axes([0.505, 0.55, 0.185, 0.36])
cf = [('Ethyl linoleate', 0.034), ('Octanoic acid', 0.007),
      ('Diethyl succinate', 0.072), ('Ethyl decanoate', -0.027),
      ('Acetic acid', -0.072), ('2,3,5- Trimethylpyrazine', -0.011),
      ('Ethyl lactate', -0.007), ('Ethyl valerate', -0.018),
      ('Isoamyl acetate', -0.003), ('Isobutanol', -0.043),
      ('Sec-butanol', -0.006), ('Acetal', 0.080),
      ('Isobutyraldehyde', -0.172), ('Reducing sugar', -0.013),
      ('Starch', -0.017)]
names = [c[0] for c in cf][::-1]
vals = np.array([c[1] for c in cf][::-1])
ax.barh(np.arange(15), vals, height=0.62, color='#4C7CC0')
ax.axvline(0, color='k', lw=0.9)
ax.set_yticks(np.arange(15)); ax.set_yticklabels(names, fontsize=8, fontweight='bold')
ax.set_xlim(-0.19, 0.088)
ax.set_xticks([-0.15, -0.10, -0.05, 0.00, 0.05])
ax.set_xticklabels(['-0.15', '-0.10', '-0.05', '0.00', '0.05'], fontsize=8)
ax.set_xlabel('Coefficients', fontsize=13)
ax.set_title('Feature Coefficients from Lasso Regression', fontsize=8.5,
             fontweight='bold', pad=4)
ax.tick_params(axis='x', labelsize=8)

# ================================================================= (d) =====
ax = fig.add_axes([0.77, 0.55, 0.185, 0.36])
refC = rng.uniform(55.5, 78.5, 95)
calC = 0.92 * refC + 5.5 + rng.normal(0, 2.0, 95)
ax.scatter(refC, np.clip(calC, 55, 90), s=16, color='#9B9B9B', zorder=2,
           label='Calibration')
refP = rng.uniform(58, 63, 12)
ax.scatter(refP, np.full(12, 55.4) + rng.normal(0, 0.25, 12), s=16,
           color='#9B9B9B', zorder=2)
refP = rng.uniform(59, 76, 25)
predP = 1.02 * refP + 3.2 + rng.normal(0, 2.4, 25)
ax.scatter(refP, np.clip(predP, 55, 89.5), s=18, color='#D6404A', zorder=3,
           label='Prediction')
xx = np.array([55, 79.5])
ax.plot(xx, 1.087 * xx - 4.8, 'k-', lw=1.8, label='fit line')
ax.set_xlim(53, 81); ax.set_ylim(53.5, 91)
ax.set_xticks(np.arange(55, 81, 5)); ax.set_yticks(np.arange(55, 91, 5))
ax.set_xlabel('Reference'); ax.set_ylabel('Predicted')
ax.set_title('Linear Regression', fontsize=12, pad=4)
ax.text(0.04, 0.965, 'RMSE = 3.71\nR$^2$ = -0.79', transform=ax.transAxes,
        fontsize=9, va='top', fontweight='bold')
ax.legend(loc='lower right', fontsize=7)

# ================================================================= (e) =====
ax = fig.add_axes([0.045, 0.075, 0.185, 0.36])
rocs = [('Logistic Regression', 0.73, '#1f77b4'),
        ('K Neighbors Classifier', 0.74, '#ff7f0e'),
        ('Random Forest Classifier', 0.75, '#2ca02c'),
        ('Linear Discriminant Analysis', 0.74, '#d62728'),
        ('MLP Classifier', 0.68, '#9467bd'),
        ('SVC', 0.73, '#8c564b'),
        ('Decision Tree Classifier', 0.63, '#e377c2'),
        ('Extra Tree Classifier', 0.75, '#7f7f7f'),
        ('Gradient Boosting Classifier', 0.79, '#bcbd22'),
        ('Bagging Classifier', 0.72, '#17becf'),
        ('AdaBoost Classifier', 0.70, '#1f77b4')]
fpr_grid = np.linspace(0, 1, 70)
for name, auc, color in rocs:
    mu = np.sqrt(2) * norm.ppf(min(auc, 0.999))
    npos, nneg = 40, 220
    sp = rng.normal(mu, 1, npos); sn = rng.normal(0, 1, nneg)
    flip = rng.random(npos) < 0.06
    sp[flip] = rng.normal(-0.3, 1, flip.sum())
    thr = np.sort(np.concatenate([sp, sn]))[::-1]
    tpr = np.array([(sp >= t).mean() for t in thr])
    fpr = np.array([(sn >= t).mean() for t in thr])
    ins = np.interp(fpr_grid, np.concatenate([[0], fpr, [1]]),
                    np.concatenate([[0], tpr, [1]]))
    if name in ('Random Forest Classifier', 'Decision Tree Classifier',
                'Extra Tree Classifier'):
        ins[:3] = [0, 0.20 + rng.uniform(0, 0.06), ins[2]]   # early jump
    ax.plot(np.concatenate([[0], fpr_grid, [1]]),
            np.concatenate([[0], ins, [1]]), color=color, lw=1.5)
ax.plot([0, 1], [0, 1], 'k--', lw=3)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xticks(np.arange(0, 1.1, 0.2)); ax.set_yticks(np.arange(0, 1.1, 0.2))
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC  Curve', fontsize=12, pad=4)
lab = [f'{n} ( AUC= {a} )' for n, a, _ in rocs]
h = [plt.Line2D([], [], color=c, lw=2) for _, _, c in rocs]
ax.legend(h, lab, loc='lower right', fontsize=5.6, handlelength=1.6,
          borderpad=0.4, labelspacing=0.25)

# ================================================== SHAP colour helpers ====
shap_cmap = LinearSegmentedColormap.from_list(
    'shap', ['#3B5CB8', '#7B6FC0', '#C05AA8', '#E83A6F'])

def shap_points(ax, seed=0, n=120, scale=0.10, skew=0.55):
    """generate one feature row of (shap, feature-value) points"""
    r = np.random.default_rng(seed)
    s = r.normal(0, scale, n)
    s = np.clip(s, -2.6 * scale, 2.6 * scale)
    v = np.clip(0.5 + (s / (2.4 * scale)) * 0.48 + r.normal(0, 0.11, n), 0.02, 0.98)
    return s, v

def beeswarm(ax, scales, outliers=None):
    for i, sc in enumerate(scales):
        y = len(scales) - 1 - i
        s, v = shap_points(ax, seed=100 + i, scale=sc)
        # jitter y proportional to local density (simple beeswarm look)
        order = np.argsort(np.abs(s))
        yy = np.empty_like(s)
        for k, j in enumerate(order):
            hit = np.abs(s[order[:k]] - s[j]) < 0.010
            lvl = hit.sum() % 5
            yy[j] = y + (lvl + (1 if (hit.sum() // 5) % 2 else 0)) * 0.115 * (1 if hit.sum() % 2 else -1)
        ax.scatter(s, yy, c=v, cmap=shap_cmap, vmin=0, vmax=1, s=11,
                   edgecolors='white', linewidths=0.25, zorder=3)
    if outliers:
        for (row, x0, colorv) in outliers:
            y = len(scales) - 1 - row
            ax.scatter([x0], [y + rng.uniform(-0.2, 0.2)], c=[[colorv]],
                       cmap=shap_cmap, vmin=0, vmax=1, s=11,
                       edgecolors='white', linewidths=0.25, zorder=3)

# ================================================================= (f) =====
ax = fig.add_axes([0.262, 0.075, 0.155, 0.36])
scales = [0.085, 0.075, 0.07, 0.065, 0.058, 0.052, 0.048, 0.045, 0.040,
          0.036, 0.033, 0.030, 0.027, 0.024, 0.021]
outl = [(6, 0.48, 0.10), (7, 0.57, 0.92), (8, 0.42, 0.08),
        (9, 0.26, 0.95), (9, 0.33, 0.90), (4, 0.30, 0.85), (2, 0.28, 0.90)]
beeswarm(ax, scales, outl)
ax.axvline(0, color='#888888', lw=1.0, zorder=1)
for i in range(len(scales)):
    ax.axhline(len(scales) - 1 - i, color='#CCCCCC', lw=0.5,
               ls=(0, (1, 3)), zorder=0)
ax.set_xlim(-0.12, 1.13); ax.set_ylim(-0.7, len(scales) - 0.3)
ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks(range(len(scales)))
ax.set_yticklabels(FEAT15[::-1], fontsize=8, fontweight='bold')
ax.set_xlabel('SHAP  Value', fontsize=11)
cax = fig.add_axes([0.437, 0.135, 0.010, 0.25])
cb = mpl.colorbar.ColorbarBase(cax, cmap=shap_cmap, orientation='vertical')
cb.set_ticks([])
cax.text(0.5, 1.06, 'High', ha='center', va='bottom', fontsize=7,
         transform=cax.transAxes)
cax.text(0.5, -0.06, 'Low', ha='center', va='top', fontsize=7,
         transform=cax.transAxes)
cax.text(-0.6, 0.5, 'Feature value', rotation=90, ha='right', va='center',
         fontsize=8, transform=cax.transAxes)

# ================================================================= (g) =====
ax = fig.add_axes([0.505, 0.075, 0.185, 0.36])
seg = {'A': [0.245, 0.090, 0.130, 0.060, 0.050, 0.055, 0.060, 0.055, 0.045,
             0.030, 0.030, 0.025, 0.020, 0.015, 0.008],
       'B': [0.065, 0.090, 0.105, 0.060, 0.055, 0.045, 0.030, 0.020, 0.015,
             0.020, 0.015, 0.012, 0.008, 0.005, 0.004],
       'C': [0.000, 0.110, 0.035, 0.085, 0.085, 0.060, 0.055, 0.050, 0.040,
             0.035, 0.025, 0.020, 0.015, 0.012, 0.006]}
colors_g = {'A': '#5B8FD9', 'B': '#C95FB0', 'C': '#8C9A2E'}
left = np.zeros(15)
for k in ('A', 'B', 'C'):
    ax.barh(np.arange(15)[::-1], seg[k], left=left, height=0.62,
            color=colors_g[k], label=f'Class {k}')
    left += np.array(seg[k])
ax.set_yticks(np.arange(15)[::-1])
ax.set_yticklabels(FEAT15, fontsize=8, fontweight='bold')
ax.set_xlim(0, 0.37); ax.set_xticks(np.arange(0, 0.36, 0.05))
ax.set_xticklabels([f'{v:.2f}' for v in np.arange(0, 0.36, 0.05)], fontsize=8)
ax.set_xlabel('SHAP  Value', fontsize=11)
ax.legend(loc='lower right', fontsize=8, handlelength=1.2, handleheight=1.0)

# ================================================================= (h) =====
ax = fig.add_axes([0.77, 0.075, 0.185, 0.36])
xs = np.arange(26)
real = np.array([78.2, 78.8, 79.2, 75.6, 77.0, 77.6, 78.9, 79.3, 79.6, 76.2,
                 77.3, 78.0, 78.6, 79.5, 77.8, 76.0, 77.4, 78.2, 78.8, 79.7,
                 78.3, 77.6, 78.4, 75.8, 77.2, 78.1])
pred = np.array([77.6, 78.4, 78.8, 76.4, 76.8, 77.9, 79.4, 79.0, 79.2, 77.5,
                 76.9, 78.3, 79.8, 78.9, 78.4, 77.0, 76.6, 78.6, 79.3, 79.0,
                 78.8, 77.9, 78.0, 76.6, 77.7, 78.3])
ax.plot(xs, real, '-o', color='#4472C4', ms=3.5, lw=1.2, label='Real value')
ax.plot(xs, pred, '-o', color='#ED9437', ms=3.5, lw=1.2, label='Predicted value')
ax.set_xlim(-1, 26); ax.set_ylim(74.5, 85.5)
ax.set_xticks(np.arange(0, 26, 5)); ax.set_yticks([75, 80, 85])
ax.set_xlabel('Test-Sample'); ax.set_ylabel('Score')
ax.set_title('Linear Regression', fontsize=12, pad=4)
ax.legend(loc='upper left', fontsize=7)

# panel letters
for letter, (lx, ly) in {'a': (0.012, 0.93), 'b': (0.242, 0.93),
                         'c': (0.478, 0.93), 'd': (0.745, 0.93),
                         'e': (0.012, 0.46), 'f': (0.240, 0.46),
                         'g': (0.478, 0.46), 'h': (0.745, 0.46)}.items():
    fig.text(lx, ly, letter, fontsize=15, fontweight='bold')
save(fig, 'fig4')
