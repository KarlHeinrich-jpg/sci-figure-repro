# -*- coding: utf-8 -*-
"""Fig. 1  Workflow schematic for constructing machine learning models.

Panel structure (as in the source paper):
  top-left  : data collection (bottles -> GC-FID; Jiupei -> process parameters)
  top-right : six-step machine-learning pipeline inside circles
  middle    : band "Machine learning classification or predictive models"
  bottom    : seven miniature algorithm diagrams
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import (Rectangle, Circle, Ellipse, Polygon,
                                FancyBboxPatch)
from common import save

rng = np.random.default_rng(1)

fig = plt.figure(figsize=(15.5, 10.6))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100); ax.set_ylim(0, 70)
ax.axis('off')

DIV = 33.5                      # divider between the two top blocks

def R(x, y, w, h, **kw):
    ax.add_patch(Rectangle((x, y), w, h, **kw))

# ============================================================== frames =====
R(0.6, 0.6, 98.8, 68.8, fill=False, edgecolor='k', lw=2.0)         # outer
ax.plot([DIV, DIV], [35.6, 69.4], color='k', lw=2.0)               # divider
R(0.6, 29.0, 98.8, 6.6, fill=False, edgecolor='k', lw=2.0)         # model band
R(DIV + 0.7, 35.6, 98.1 - DIV, 33.8, fill=False, edgecolor='k', lw=2.0)
ax.text(50, 32.3, 'Machine learning classification or predictive models',
        ha='center', va='center', fontsize=15, fontweight='bold')

# ================================================== (1) data collection ====
ax.text(17.0, 66.6, 'Data collection', ha='center', fontsize=15, fontweight='bold')

def bottle(x, y, label):
    ax.add_patch(FancyBboxPatch((x - 1.05, y), 2.1, 3.5,
                                boxstyle='round,pad=0.02,rounding_size=0.55',
                                facecolor='#F7F4EC', edgecolor='#8A8A8A', lw=1.0))
    R(x - 0.40, y + 3.2, 0.80, 1.0, facecolor='#F7F4EC', edgecolor='#8A8A8A', lw=1.0)
    ax.add_patch(Polygon([(x - 0.50, y + 4.2), (x + 0.50, y + 4.2),
                          (x + 0.38, y + 4.5), (x - 0.38, y + 4.5)],
                         facecolor='#C3392B', edgecolor='#8A8A8A', lw=0.8))
    ax.plot([x - 0.52, x + 0.52], [y + 3.2, y + 3.2], color='#C3392B', lw=1.8)
    ax.text(x, y - 1.0, label, ha='center', fontsize=12, fontweight='bold')

bottle(6.4, 55.6, 'B'); bottle(10.6, 55.6, 'C')
ax.annotate('', xy=(15.0, 57.4), xytext=(12.4, 57.4),
            arrowprops=dict(arrowstyle='-|>', lw=1.8, mutation_scale=11))

# GC-FID instrument
gx, gy = 15.6, 53.6
R(gx, gy, 8.2, 5.2, facecolor='#EDEDED', edgecolor='#777777', lw=1.2)
R(gx + 0.9, gy + 5.2, 1.6, 2.4, facecolor='#D8D8D8', edgecolor='#777777', lw=1.1)
R(gx + 1.2, gy + 6.1, 0.95, 2.3, facecolor='#B9B9B9', edgecolor='#777777', lw=1.0)
R(gx + 6.1, gy + 5.2, 1.7, 1.3, facecolor='#DCDCDC', edgecolor='#777777', lw=1.0)
R(gx + 6.22, gy + 5.32, 1.46, 0.95, facecolor='#2F4F6F', edgecolor='#555555', lw=0.6)
ax.add_patch(Circle((gx + 5.1, gy + 4.2), 0.32, facecolor='#C3392B',
                    edgecolor='#7A1F16', lw=0.8))
R(gx + 0.4, gy - 0.9, 7.4, 0.8, facecolor='#DDDDDD', edgecolor='#777777', lw=1.0)
for i in range(6):
    ax.plot([gx + 0.8 + i * 1.15, gx + 1.3 + i * 1.15], [gy - 0.55, gy - 0.55],
            color='#999999', lw=0.7)
ax.text(gx + 4.1, gy - 2.2, 'GC-FID', ha='center', fontsize=13, fontweight='bold')

# Jiupei mounds
def mound(x, y, label, s=1.0):
    ax.add_patch(Polygon([(x - 1.45 * s, y), (x - 0.62 * s, y + 1.5 * s),
                          (x + 0.62 * s, y + 1.5 * s), (x + 1.45 * s, y)],
                         facecolor='#4A4A4A', edgecolor='#2E2E2E', lw=1.0))
    ax.text(x, y - 0.9, label, ha='center', fontsize=11.5, fontweight='bold')

mound(3.4, 43.4, 'A-JP'); mound(7.4, 43.4, 'B-JP'); mound(11.4, 43.4, 'C-JP')
ax.annotate('', xy=(15.4, 44.4), xytext=(12.8, 44.4),
            arrowprops=dict(arrowstyle='-|>', lw=1.8, mutation_scale=11))

# process-parameter glyphs -------------------------------------------------
def droplet(x, y):
    ax.add_patch(Polygon([(x, y + 0.85), (x - 0.44, y - 0.05), (x + 0.44, y - 0.05)],
                         facecolor='#7EB6E8', edgecolor='#3C74A8', lw=0.8))
    ax.add_patch(Ellipse((x, y - 0.12), 0.88, 0.64, facecolor='#7EB6E8',
                         edgecolor='#3C74A8', lw=0.8))

def thermometer(x, y):
    R(x - 0.13, y - 0.3, 0.26, 1.5, facecolor='white', edgecolor='#C3392B', lw=1.0)
    ax.add_patch(Circle((x, y - 0.45), 0.29, facecolor='#C3392B',
                        edgecolor='#8A1F16', lw=0.8))
    ax.plot([x, x], [y - 0.3, y + 0.9], color='#C3392B', lw=1.6)

def flask(x, y):
    ax.add_patch(Polygon([(x - 0.17, y + 0.9), (x + 0.17, y + 0.9),
                          (x + 0.60, y - 0.5), (x - 0.60, y - 0.5)],
                         facecolor='#E8EEF5', edgecolor='#3C5A78', lw=1.0))
    ax.plot([x - 0.38, x + 0.38], [y - 0.1, y - 0.1], color='#7EA8CC', lw=2.0)

def molecule(x, y, col, n=3, spread=0.55, r=0.22):
    pts = [(x - 0.5 * (n - 1) * spread + i * spread,
            y + (0.30 if i % 2 else -0.20)) for i in range(n)]
    for i in range(n - 1):
        ax.plot(*zip(pts[i], pts[i + 1]), color='#555555', lw=1.2, zorder=1)
    for px, py in pts:
        ax.add_patch(Circle((px, py), r, facecolor=col, edgecolor='#555555',
                            lw=0.7, zorder=2))

def chain(x, y):
    for i in range(3):
        ax.add_patch(Ellipse((x - 0.48 + i * 0.5, y + (0.16 if i % 2 else -0.05)),
                             0.56, 0.40, facecolor='none', edgecolor='#8A6D2F', lw=1.3))

# row 1: three physical parameters
for xx, fn, lab in [(19.4, droplet, 'Moisture'),
                    (24.6, thermometer, 'Temperature'),
                    (29.6, flask, 'Titratable\nacidity')]:
    fn(xx, 49.6)
    ax.text(xx, 48.6, lab, ha='center', va='top', fontsize=6.2, linespacing=0.95)

# row 2: four chemical parameters
molecule(18.9, 46.3, '#C3392B', 3, 0.46)
ax.text(18.9, 45.2, 'Ethanol', ha='center', va='top', fontsize=5.9)
chain(22.7, 46.2)
ax.text(22.7, 45.2, 'Reducing\nsugar', ha='center', va='top', fontsize=5.9,
        linespacing=0.95)
molecule(26.6, 46.3, '#7A5CA8', 4, 0.36)
ax.text(26.6, 45.2, 'Starch', ha='center', va='top', fontsize=5.9)
molecule(30.4, 46.3, '#3C8A4E', 3, 0.32, 0.16)
ax.text(30.4, 45.2, 'Lacticacid', ha='center', va='top', fontsize=5.9)
ax.text(24.0, 40.6, 'Process parameters', ha='center', fontsize=13,
        fontweight='bold')

# ================================================ (2) ML pipeline circles ==
ax.text(66.5, 66.6, 'How machine learning works', ha='center', fontsize=15,
        fontweight='bold')

pos = {'fe': (41.0, 61.0), 'sd': (57.0, 61.0), 'smote': (72.5, 61.0),
       'shap': (41.0, 46.0), 'eval': (57.0, 46.0), 'sel': (72.5, 46.0)}
CR = 4.2
for x, y in pos.values():
    ax.add_patch(Circle((x, y), CR, facecolor='white', edgecolor='k', lw=1.6,
                        zorder=3))

def icon_fe(x, y):
    for r in (3.0, 2.4, 1.8, 1.2, 0.6):
        ax.add_patch(Ellipse((x - 0.2, y + 0.7), r * 1.5, r, angle=-18,
                             fill=False, edgecolor='#4A4A4A', lw=0.75, zorder=4))
    ax.add_patch(Circle((x - 1.7, y - 1.7), 1.1, facecolor='#7EA8D8',
                        edgecolor='#3C5A88', lw=0.8, zorder=4))
    ax.plot([x - 3.0, x + 2.8], [y - 2.6, y - 2.6], color='#777777', lw=0.8, zorder=4)
    ax.plot([x + 2.8, x + 2.8], [y - 2.6, y + 2.6], color='#777777', lw=0.8, zorder=4)
    ax.text(x - 3.3, y + 0.5, r'$\beta_1$', fontsize=7, zorder=5)
    ax.text(x + 3.05, y - 2.85, r'$\beta_p$', fontsize=7, zorder=5)

def icon_sd(x, y):
    for dy in (1.25, -1.25):
        ax.plot([x - 2.6, x - 0.2], [y + dy, y], color='#2E2E2E', lw=3.4,
                solid_capstyle='round', zorder=4)
    ax.plot([x - 0.2, x + 1.5], [y, y], color='#2E2E2E', lw=3.4,
            solid_capstyle='round', zorder=4)
    ax.add_patch(Polygon([(x + 1.3, y + 0.95), (x + 3.0, y), (x + 1.3, y - 0.95)],
                         facecolor='#2E2E2E', edgecolor='#2E2E2E', zorder=4))

def icon_smote(x, y):
    groups = [('#E8B23A', -1.9, 1.3), ('#4C9C5E', 1.3, 1.4),
              ('#D86A8A', -1.1, -1.5), ('#5B84C4', 1.6, -1.3)]
    for c, cx, cy in groups:
        pts = rng.normal(0, 0.45, (9, 2)) + [x + cx, y + cy]
        ax.scatter(*pts.T, s=8, color=c, zorder=4)
    ax.text(x - 2.0, y - 3.0, 'Synthetic data', fontsize=4.3, ha='center', zorder=5)
    ax.text(x + 1.5, y - 3.0, 'Randomly Synthetic Real data', fontsize=3.8,
            ha='center', zorder=5)

def icon_shap(x, y):
    n = 22
    for i in range(n):
        v = rng.random()
        ax.add_patch(Rectangle((x - 3.0, y - 2.45 + i * 0.225), 1.0, 0.20,
                               facecolor=mpl.cm.RdYlBu(v), edgecolor='none',
                               zorder=4))
        ax.add_patch(Rectangle((x - 1.8, y - 2.45 + i * 0.225), 3.2, 0.20,
                               facecolor=mpl.cm.RdYlBu(1 - v * 0.85),
                               edgecolor='none', zorder=4))
    ax.plot([x - 1.8, x - 1.8], [y - 2.5, y + 2.4], color='white', lw=0.7, zorder=5)

def icon_eval(x, y):
    for i in range(2):
        for j in range(2):
            v = 0.85 if i == j else 0.10
            ax.add_patch(Rectangle((x - 2.8 + j * 1.2, y + 0.1 - i * 1.2), 1.15,
                                   1.15, facecolor=mpl.cm.Greens(v),
                                   edgecolor='#888888', lw=0.6, zorder=4))
    xx = np.linspace(0, 1, 30)
    ax.plot(x - 2.8 + 2.5 * xx, y - 2.5 + 1.5 * xx ** 1.6, color='#999999',
            lw=0.9, zorder=4)
    ax.plot(x + 0.2 + 2.5 * xx, y - 2.5 + 1.9 * xx ** 1.1, color='#4A4A4A',
            lw=0.9, zorder=4)
    ax.plot([x - 2.8, x - 0.3], [y - 2.5, y - 1.0], color='#CCCCCC', lw=0.6, zorder=4)
    ax.plot([x + 0.2, x + 2.7], [y - 2.5, y - 0.6], color='#CCCCCC', lw=0.6, zorder=4)

def icon_sel(x, y):
    R(x - 2.9, y - 1.7, 5.8, 3.4, facecolor='#D9D9D9', edgecolor='#6E6E6E',
      lw=1.1, zorder=4)
    R(x - 2.4, y - 1.2, 4.8, 2.4, facecolor='#27405C', edgecolor='#1B2C42',
      lw=0.8, zorder=4)
    t = np.linspace(-2.1, 2.1, 100)
    ax.plot(x + t, y + 0.55 * np.sin(t * 2.6) * np.exp(-0.06 * t ** 2),
            color='white', lw=1.1, zorder=5)
    ax.add_patch(Circle((x - 2.0, y - 2.4), 0.26, facecolor='#E8D44D',
                        edgecolor='#8A6D2F', lw=0.7, zorder=5))
    ax.add_patch(Circle((x - 1.2, y - 2.4), 0.26, facecolor='#D86A8A',
                        edgecolor='#8A3A4E', lw=0.7, zorder=5))
    for i in range(4):
        R(x + 0.1 + i * 0.6, y - 2.58, 0.48, 0.32, facecolor='#F0C845',
          edgecolor='#8A6D2F', lw=0.5, zorder=5)
    R(x - 1.5, y - 3.0, 3.0, 0.32, facecolor='#B9B9B9', edgecolor='#6E6E6E',
      lw=0.8, zorder=5)

icon_fe(*pos['fe']); icon_sd(*pos['sd']); icon_smote(*pos['smote'])
icon_shap(*pos['shap']); icon_eval(*pos['eval']); icon_sel(*pos['sel'])

captions = {
 'fe': ('Feature engineering',
        'Using Lasso regression feature\nscreening methods to avoid model overfitting'),
 'sd': ('Select data',
        'Split the data you have into\ntow groups: training data and test data'),
 'smote': ('SMOTE',
           'Synthetic Minority Over-sampling Technique\nto resolve data imbalances'),
 'shap': ('SHAP model explanation',
          'Enhancing the credibility and transparency\nof machine learning models'),
 'eval': ('Model Evaluation',
          'Selection of models with optimal\nmodel evaluation metrics'),
 'sel': ('Select Model',
         'Selection of 11 machine learning classifiers\nand 9 predictive models'),
}
for k, (title, cap) in captions.items():
    x, y = pos[k]
    ax.text(x, y - CR - 0.6, title, ha='center', va='top', fontsize=10.5,
            fontweight='bold')
    ax.text(x, y - CR - 1.55, cap, ha='center', va='top', fontsize=6.8,
            linespacing=1.25)

# connectors: attach to circle edges on the correct side -------------------
def link(a, b, dy=0.85):
    (x1, y1), (x2, y2) = pos[a], pos[b]
    s = 1 if x2 > x1 else -1
    for d in (dy, -dy):
        ax.plot([x1 + s * CR, x2 - s * CR], [y1 + d, y2 + d],
                color='#555555', lw=1.2, zorder=2)

link('fe', 'sd'); link('sd', 'smote'); link('sel', 'eval'); link('eval', 'shap')

# SMOTE -> right -> Select Model ; left column -> Feature engineering
xs, ys = pos['smote']
ax.plot([xs + CR, 94.5], [ys, ys], color='#555555', lw=1.2, zorder=2)
ax.plot([94.5, 94.5], [ys, pos['sel'][1]], color='#555555', lw=1.2, zorder=2)
ax.plot([94.5, pos['sel'][0] + CR], [pos['sel'][1], pos['sel'][1]],
        color='#555555', lw=1.2, zorder=2)
ax.plot([DIV, 36.4], [57.5, 57.5], color='#555555', lw=1.2, zorder=2)
ax.plot([36.4, 36.4], [57.5, pos['fe'][1] - 1.4], color='#555555', lw=1.2, zorder=2)
ax.plot([36.4, pos['fe'][0] - CR], [pos['fe'][1] - 1.4, pos['fe'][1] - 1.4],
        color='#555555', lw=1.2, zorder=2)

# ============================================= (3) seven algorithm panels ===
model_labels = ['MLP', 'Gradient Boosting', 'KNN', 'Logistic Regression',
                'Random Forest', 'SVM', 'Linear Regression']
slot_w, gap = 13.0, 0.75
panels = []
for i, lab in enumerate(model_labels):
    x0 = 0.9 + i * (slot_w + gap)
    iax = fig.add_axes([(x0 + 0.5) / 100, 6.2 / 70, (slot_w - 1.0) / 100, 19.0 / 70])
    iax.set_xlim(-3.4, 3.4); iax.set_ylim(-3.4, 3.4)
    iax.axis('off'); iax.set_aspect('equal')
    ax.text(x0 + slot_w / 2, 3.4, lab, ha='center', fontsize=13,
            fontweight='bold')
    panels.append(iax)

# ---- MLP ----
p = panels[0]
layers = [[(2.0, y) for y in (1.9, 1.0, 0.1, -0.8, -1.7)],
          [(0.3, y) for y in (2.1, 1.2, 0.3, -0.6, -1.5)],
          [(-1.5, 1.1), (-1.5, 0.2), (-1.5, -0.7)]]
cols = ['#D8645A', '#4E86C6', '#E8C84D']
for a_, b_ in zip(layers, layers[1:]):
    for p1 in a_:
        for p2 in b_:
            p.plot(*zip(p1, p2), color='#AAAAAA', lw=0.6, zorder=1)
for L, c in zip(layers, cols):
    for pnt in L:
        p.add_patch(Circle(pnt, 0.40, facecolor=c, edgecolor='#555555',
                           lw=0.7, zorder=2))
p.text(2.85, 1.95, 'Classify\nposition', fontsize=5.4, ha='right')
p.text(-3.35, 0.2, 'Input', fontsize=5.8, ha='left')
p.text(2.9, 0.2, 'Output', fontsize=5.8, ha='right')
p.text(0.25, -2.55, 'Hidden layer', fontsize=5.8, ha='center')

# ---- Gradient Boosting ----
p = panels[1]
for i in range(3):
    for j in range(2):
        p.add_patch(Rectangle((-3.0 + i * 1.02, -0.9 + j * 1.02), 0.92, 0.92,
                              facecolor='#AFC8E8', edgecolor='#3C5A88', lw=0.7))
        p.text(-2.54 + i * 1.02, -0.44 + j * 1.02,
               '+' if (i + j) % 2 == 0 else r'$-$', ha='center', va='center',
               fontsize=9)
p.annotate('', xy=(0.9, 0), xytext=(0.1, 0),
           arrowprops=dict(arrowstyle='-|>', lw=1.2, mutation_scale=9))
p.add_patch(Rectangle((1.05, -1.2), 1.75, 2.4, facecolor='#E8A0A0',
                      edgecolor='#8A3A3A', lw=0.8))
for i in range(2):
    for j in range(2):
        p.text(1.48 + i * 0.8, -0.45 + j * 0.9, r'$+/-$', ha='center',
               va='center', fontsize=6.0)
p.text(-1.2, -2.05, 'Drug property1', fontsize=5.6, ha='center')
p.text(1.93, 1.65, 'Weak learners', fontsize=5.0, ha='center')

# ---- KNN ----
p = panels[2]
a_pts = rng.normal([-1.5, -1.15], 0.45, (12, 2))
b_pts = rng.normal([1.35, 1.35], 0.40, (11, 2))
p.scatter(*a_pts.T, marker='s', s=26, facecolor='#F0D24D', edgecolor='#8A6D2F',
          lw=0.6, zorder=2)
p.scatter(*b_pts.T, s=26, facecolor='#D8645A', edgecolor='#8A3A2E', lw=0.6, zorder=2)
p.scatter([0.6], [0.35], s=50, facecolor='none', edgecolor='k', lw=1.1, zorder=4)
p.annotate('', xy=(0.25, 0.1), xytext=(-0.35, -0.55),
           arrowprops=dict(arrowstyle='-|>', lw=0.9, mutation_scale=7))
p.text(3.30, 2.75, 'Category B', fontsize=5.6, ha='right')
p.text(-3.30, -2.55, 'Category A', fontsize=5.6, ha='left')
p.text(3.30, 0.30, 'New data point', fontsize=5.2, ha='right')
p.text(3.30, -0.30, 'Assigned to\nCategory A', fontsize=5.2, ha='right',
       va='top')

# ---- Logistic Regression ----
p = panels[3]
t = np.linspace(-3.2, 3.2, 120)
p.plot(t, 2.3 / (1 + np.exp(-2.2 * t)) - 0.9, color='#333333', lw=1.5)
up = rng.normal([-1.4, 1.45], [0.8, 0.13], (9, 2))
lo = rng.normal([1.4, -1.25], [0.8, 0.13], (9, 2))
p.scatter(*up.T, s=24, facecolor='#D8645A', edgecolor='#8A3A2E', lw=0.6, zorder=3)
p.scatter(*lo.T, marker='s', s=24, facecolor='#F0D24D', edgecolor='#8A6D2F',
          lw=0.6, zorder=3)
p.text(0.9, 2.55, 'sigmoid', fontsize=6.2)

# ---- Random Forest ----
p = panels[4]
p.text(0, 2.9, 'Restra', fontsize=6.0, ha='center')
for dx in (-2.2, 0, 2.2):
    p.plot([0, dx], [2.5, 1.6], color='#555555', lw=0.9)
    p.add_patch(Circle((dx, 1.4), 0.24, facecolor='white', edgecolor='k', lw=0.9))
    p.text(dx, 0.98, 'Branch Tree', fontsize=4.4, ha='center')
    for k in (-0.65, 0.65):
        p.plot([dx, dx + k], [1.15, 0.5], color='#555555', lw=0.8)
        p.add_patch(Circle((dx + k, 0.3), 0.18, facecolor='white',
                           edgecolor='k', lw=0.8))
    p.text(dx, -0.18, 'Root', fontsize=4.4, ha='center')
p.annotate('', xy=(0, -1.25), xytext=(0, -0.55),
           arrowprops=dict(arrowstyle='-|>', lw=0.9, mutation_scale=7))
p.text(0, -1.7, 'Majority Voting', fontsize=5.4, ha='center')
p.text(0, -2.45, 'Final Result', fontsize=5.4, ha='center')

# ---- SVM ----
p = panels[5]
p.plot([-3.1, 3.1], [-2.5, 1.9], color='#8A8A8A', lw=0.9)
p.plot([-3.1, 3.1], [-1.8, 2.8], color='#777777', lw=0.8, ls=(0, (4, 2)))
p.plot([-3.1, 3.1], [-3.2, 1.2], color='#777777', lw=0.8, ls=(0, (4, 2)))
up = rng.normal([-1.4, 1.55], 0.45, (13, 2))
lo = rng.normal([1.5, -1.25], 0.42, (13, 2))
p.scatter(*up.T, s=26, facecolor='#D8645A', edgecolor='#8A3A2E', lw=0.6, zorder=3)
p.scatter(*lo.T, marker='D', s=26, facecolor='#F0D24D', edgecolor='#8A6D2F',
          lw=0.6, zorder=3)
p.text(-3.3, 2.95, 'Ordered protein', fontsize=5.6, ha='left')
p.text(1.30, 1.35, 'Margin', fontsize=5.6)
p.text(1.5, -2.55, 'Disordered\nprotein', fontsize=5.8, ha='center')

# ---- Linear Regression ----
p = panels[6]
p.plot([-2.9, -2.9], [-2.4, 2.6], color='#555555', lw=1.0)
p.plot([-2.9, 2.9], [-2.4, -2.4], color='#555555', lw=1.0)
tt = rng.uniform(-2.4, 2.4, 16)
p.scatter(tt, 0.75 * tt + rng.normal(0, 0.26, 16), s=22,
          facecolor='#D8645A', edgecolor='#8A3A2E', lw=0.6, zorder=3)
p.plot([-2.6, 2.6], [-2.0, 2.0], color='#555555', lw=1.0, ls=(0, (4, 2)))
p.text(-2.75, 1.9, 'Data point', fontsize=5.8)
p.text(1.55, 2.45, 'Model', fontsize=5.6)
p.text(-3.35, 0, 'Weight', fontsize=5.4, rotation=90, va='center')
p.text(2.4, -2.8, 'Height', fontsize=5.4, ha='center')
for i in range(3):
    ax.add_patch(Ellipse((96.2 + i * 1.1, 13.5), 0.62, 0.95,
                         facecolor='#2E2E2E'))

save(fig, 'fig1')
