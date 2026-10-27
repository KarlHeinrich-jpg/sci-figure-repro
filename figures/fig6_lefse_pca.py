# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Fig. 6  LEfSe cladograms (a,d), community bars (b,e), PCA (c,f)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle, Ellipse, Rectangle
from common import PCA_A, PCA_B, PCA_C, LEFSE_A, LEFSE_B, LEFSE_C, save

def nd(level, color, children=(), leaf=False):
    return dict(level=level, color=color, children=list(children), leaf=leaf)

# ------------------------------------------------------------ cladogram ----
RADII = [0.16, 0.30, 0.44, 0.58, 0.74, 0.92]
BAND = 0.115
BAND_PER = [0.14, 0.14, 0.14, 0.16, 0.18, 0.18]
MID = [r + BAND / 2 for r in RADII]

def cladogram(ax, nodes, root, span):
    order = []
    def walk(n):
        for k in nodes[n]['children']:
            walk(k)
        if not nodes[n]['children']:
            order.append(n)
    walk(root)
    a0, a1 = span
    leaf_ang = {n: a0 + (i + 0.5) * (a1 - a0) / len(order) for i, n in enumerate(order)}
    spans = {}
    def span_of(n):
        if n in spans:
            return spans[n]
        kids = nodes[n]['children']
        s = (leaf_ang[n], leaf_ang[n]) if not kids else \
            (min(span_of(k)[0] for k in kids), max(span_of(k)[1] for k in kids))
        spans[n] = s
        return s

    def pol(r, deg):
        t = np.deg2rad(deg)
        return r * np.cos(t), r * np.sin(t)

    dw = (a1 - a0) / len(order)
    for n, d in nodes.items():
        if d['color'] and not d['leaf']:
            lo, hi = span_of(n)
            k = d['level'] - 1
            ax.add_patch(Wedge((0, 0), RADII[k] + BAND_PER[k], lo, hi,
                               width=BAND_PER[k], facecolor=d['color'],
                               alpha=0.5, lw=0))
        elif d['color'] and d['leaf']:
            ang = leaf_ang[n]
            ax.add_patch(Wedge((0, 0), RADII[5] + BAND_PER[5], ang - dw * 0.42,
                               ang + dw * 0.42, width=BAND_PER[5],
                               facecolor=d['color'], alpha=0.5, lw=0))
    def draw(n):
        d = nodes[n]
        lo, hi = span_of(n)
        ang = (lo + hi) / 2
        rm = MID[d['level'] - 1] if d['level'] >= 1 else 0.02
        if d['leaf']:
            x, y = pol(RADII[5] + BAND * 0.5, leaf_ang[n])
            ax.plot(*zip(pol(RADII[4] + BAND_PER[4] * 0.5, leaf_ang[n]), (x, y)), color='#888888',
                    lw=1.1, zorder=4)
            ax.add_patch(Circle((x, y), 0.015, facecolor='k', zorder=6))
            lx, ly = pol(RADII[5] + BAND * 1.25, leaf_ang[n])
            rot = leaf_ang[n] % 360
            if 90 < rot < 270:
                rot += 180
            ax.text(lx, ly, n, fontsize=5.5, style='italic', ha='center',
                    va='center', rotation=rot)
            return (x, y)
        x, y = pol(rm, ang)
        ax.add_patch(Circle((x, y), 0.020, facecolor='white', edgecolor='k',
                            lw=1.2, zorder=6))
        for k in d['children']:
            pk = draw(k)
            ax.plot(*zip((x, y), pk), color='#888888', lw=1.1, zorder=4)
        return (x, y)
    draw(root)
    ax.add_patch(Circle((0, 0), 0.024, facecolor='white', edgecolor='k',
                        lw=1.2, zorder=6))
    ax.set_xlim(-1.30, 1.30); ax.set_ylim(-1.30, 1.30)
    ax.set_aspect('equal'); ax.axis('off')

def class_legend(axc, items, x0, y0, dy, fs=7.4):
    for i, (lab, c) in enumerate(items):
        axc.add_patch(Rectangle((x0, y0 - i * dy), 0.022, 0.017,
                                facecolor=c, transform=axc.transAxes, clip_on=False))
        axc.text(x0 + 0.032, y0 - i * dy + 0.008, lab, fontsize=fs, style='italic',
                 va='center', transform=axc.transAxes)

# ------------------------------------------------------- bacteria tree -----
bact = {
 'root': nd(0, None, ['prot', 'acti', 'baci']),
 'prot': nd(1, LEFSE_A, ['alpha', 'gamma']),
 'alpha': nd(2, LEFSE_A, ['rhodo']),
 'rhodo': nd(3, LEFSE_A, ['acetob']),
 'acetob': nd(4, LEFSE_A, ['g_acb', 'g_kom']),
 'g_acb': nd(5, LEFSE_A, ['y1', 'z1', 'a2', 'b2', 'c2', 'd2', 'e2', 'w1', 'x1']),
 'y1': nd(6, LEFSE_B, leaf=True), 'z1': nd(6, LEFSE_A, leaf=True),
 'a2': nd(6, LEFSE_C, leaf=True), 'b2': nd(6, LEFSE_A, leaf=True),
 'c2': nd(6, LEFSE_A, leaf=True), 'd2': nd(6, LEFSE_A, leaf=True),
 'e2': nd(6, LEFSE_C, leaf=True), 'w1': nd(6, LEFSE_A, leaf=True),
 'x1': nd(6, LEFSE_A, leaf=True),
 'g_kom': nd(5, LEFSE_A, ['f2', 'g2', 'h2']),
 'f2': nd(6, LEFSE_A, leaf=True), 'g2': nd(6, LEFSE_A, leaf=True),
 'h2': nd(6, LEFSE_A, leaf=True),
 'gamma': nd(2, LEFSE_A, ['entero', 'coryn']),
 'entero': nd(3, LEFSE_A, ['erwi']),
 'erwi': nd(4, LEFSE_A, ['g_mix']),
 'g_mix': nd(5, LEFSE_A, ['i2']), 'i2': nd(6, LEFSE_A, leaf=True),
 'coryn': nd(3, LEFSE_A, ['f_cor']),
 'f_cor': nd(4, LEFSE_A, ['g_cor']),
 'g_cor': nd(5, LEFSE_A, ['c1', 'd1']),
 'c1': nd(6, LEFSE_A, leaf=True), 'd1': nd(6, LEFSE_A, leaf=True),
 'acti': nd(2, None, ['pseu']),
 'pseu': nd(3, LEFSE_C, ['f_pse']),
 'f_pse': nd(4, LEFSE_C, ['g_sac']),
 'g_sac': nd(5, LEFSE_C, ['e1']), 'e1': nd(6, LEFSE_C, leaf=True),
 'baci': nd(2, None, ['lact']),
 'lact': nd(3, None, ['strept', 'cal_f', 'wei_f', 'mam_f', 'ped_f']),
 'strept': nd(4, LEFSE_A, ['g_lac', 'g_leu', 'g_lpl', 'g_lab',
                           'g_len', 'g_lev', 'g_lig', 'g_lim']),
 'g_lac': nd(5, LEFSE_C, ['v1']), 'v1': nd(6, LEFSE_C, leaf=True),
 'g_leu': nd(5, LEFSE_C, ['o1']), 'o1': nd(6, LEFSE_C, leaf=True),
 'g_lpl': nd(5, LEFSE_A, ['j1']), 'j1': nd(6, LEFSE_A, leaf=True),
 'g_lab': nd(5, LEFSE_B, ['k1', 'l1', 'm1']),
 'k1': nd(6, LEFSE_A, leaf=True), 'l1': nd(6, LEFSE_A, leaf=True),
 'm1': nd(6, LEFSE_A, leaf=True),
 'g_len': nd(5, LEFSE_B, ['n1']), 'n1': nd(6, LEFSE_B, leaf=True),
 'g_lev': nd(5, LEFSE_B, ['p1']), 'p1': nd(6, LEFSE_B, leaf=True),
 'g_lig': nd(5, LEFSE_B, ['q1']), 'q1': nd(6, LEFSE_B, leaf=True),
 'g_lim': nd(5, LEFSE_A, ['r1']), 'r1': nd(6, LEFSE_A, leaf=True),
 'cal_f': nd(4, None, ['cal']), 'cal': nd(5, LEFSE_A, ['f1']),
 'f1': nd(6, LEFSE_A, leaf=True),
 'wei_f': nd(4, None, ['wei']), 'wei': nd(5, LEFSE_A, ['g1']),
 'g1': nd(6, LEFSE_A, leaf=True),
 'mam_f': nd(4, None, ['mamm']), 'mamm': nd(5, LEFSE_B, ['h1', 'i1']),
 'h1': nd(6, LEFSE_B, leaf=True), 'i1': nd(6, LEFSE_B, leaf=True),
 'ped_f': nd(4, None, ['g_ped']), 'g_ped': nd(5, None, ['s1', 't1', 'u1']),
 's1': nd(6, LEFSE_B, leaf=True), 't1': nd(6, LEFSE_B, leaf=True),
 'u1': nd(6, LEFSE_B, leaf=True),
}

fungi = {
 'root': nd(0, None, ['asc', 'muc']),
 'asc': nd(1, LEFSE_B, ['euro', 'sacc', 'schiz']),
 'euro': nd(2, LEFSE_A, ['eurot']),
 'eurot': nd(3, LEFSE_A, ['aspe']),
 'aspe': nd(4, LEFSE_A, ['g_asp', 'g_mon']),
 'g_asp': nd(5, LEFSE_A, ['w', 'x']),
 'w': nd(6, LEFSE_A, leaf=True), 'x': nd(6, LEFSE_A, leaf=True),
 'g_mon': nd(5, LEFSE_A, ['y', 'z']),
 'y': nd(6, LEFSE_A, leaf=True), 'z': nd(6, LEFSE_A, leaf=True),
 'sacc': nd(2, LEFSE_B, ['sacch']),
 'sacch': nd(3, LEFSE_B, ['f_pic', 'f_sce', 'zyg_g']),
 'f_pic': nd(4, LEFSE_B, ['g_pic']),
 'g_pic': nd(5, LEFSE_B, ['a1', 'b1']),
 'a1': nd(6, LEFSE_B, leaf=True), 'b1': nd(6, LEFSE_B, leaf=True),
 'f_sce': nd(4, LEFSE_B, ['g_sce', 'g_tor']),
 'g_sce': nd(5, LEFSE_B, ['c1']), 'c1': nd(6, LEFSE_C, leaf=True),
 'g_tor': nd(5, LEFSE_A, ['d1']), 'd1': nd(6, LEFSE_A, leaf=True),
 'zyg_g': nd(5, None, ['e1']), 'e1': nd(6, LEFSE_B, leaf=True),
 'schiz': nd(2, LEFSE_B, ['schizt']),
 'schizt': nd(3, LEFSE_B, ['f_schz']),
 'f_schz': nd(4, LEFSE_B, ['g_schz']),
 'g_schz': nd(5, LEFSE_B, ['f1']), 'f1': nd(6, LEFSE_B, leaf=True),
 'muc': nd(1, LEFSE_C, ['mucor']),
 'mucor': nd(2, LEFSE_C, ['mucorales']),
 'mucorales': nd(3, LEFSE_C, ['f_lic']),
 'f_lic': nd(4, LEFSE_C, ['g_lic']),
 'g_lic': nd(5, LEFSE_C, ['g1']), 'g1': nd(6, LEFSE_C, leaf=True),
}

bact_leg = [
 ('a: p__Proteobacteria', LEFSE_A), ('b: c__Alphaproteobacteria', LEFSE_A),
 ('c: c__Gammaproteobacteria', LEFSE_A), ('d: o__Corynebacteriales', LEFSE_A),
 ('e: o__Pseudonocardiales', LEFSE_C), ('f: o__Rhodospirillales', LEFSE_A),
 ('g: o__Enterobacterales', LEFSE_A), ('h: f__Corynebacteriaceae', LEFSE_A),
 ('i: f__Pseudonocardiaceae', LEFSE_C), ('j: f__Streptococcaceae', LEFSE_A),
 ('k: f__Acetobacteraceae', LEFSE_A), ('l: f__Erwiniaceae', LEFSE_A),
 ('m: g__Corynebacterium', LEFSE_A), ('n: g__Saccharomonospora', LEFSE_C),
 ('o: g__Calidibacillus', LEFSE_A), ('p: g__Weizmannia', LEFSE_A),
 ('q: g__Mammaliicoccus', LEFSE_B), ('r: g__Lactiplantibacillus', LEFSE_A),
 ('s: g__Lactobacillus', LEFSE_B), ('t: g__Lentilactobacillus', LEFSE_B),
 ('u: g__Leuconostoc', LEFSE_C), ('v: g__Levilactobacillus', LEFSE_B),
 ('w: g__Ligilactobacillus', LEFSE_B), ('x: g__Limosilactobacillus', LEFSE_A),
 ('y: g__Lactococcus', LEFSE_C),
 ('z: g__Acetobacter', LEFSE_A), ('a1: g__Komagataeibacter', LEFSE_A),
 ('b1: g__Mixta', LEFSE_A), ('c1: s__Corynebacterium_naruki', LEFSE_A),
 ('d1: s__Corynebacterium_variable', LEFSE_A),
 ('e1: s__Saccharomonospora_yirdiis', LEFSE_C),
 ('f1: s__Caldibacillus_thermoamylivorans', LEFSE_A),
 ('g1: s__Weizmannia_coagulans', LEFSE_A),
 ('h1: s__Mammaliicoccus_sciuri', LEFSE_B),
 ('i1: s__Staphylococcus_pettenkoferi', LEFSE_B),
 ('j1: s__Lactiplantibacillus_plantarum', LEFSE_A),
 ('k1: s__Lactobacillus_acetotolerans', LEFSE_A),
 ('l1: s__Lactobacillus_amyloyticus', LEFSE_A),
 ('m1: s__Lactobacillus_amylovorus', LEFSE_A),
 ('n1: s__Lentilactobacillus_buchneri', LEFSE_B),
 ('o1: s__Leuconostoc_citreum', LEFSE_C),
 ('p1: s__Levilactobacillus_brevis', LEFSE_B),
 ('q1: s__Ligilactobacillus_acidipiscis', LEFSE_B),
 ('r1: s__Limosilactobacillus_panis', LEFSE_A),
 ('s1: s__Pediococcus_pentosaceus', LEFSE_B),
 ('t1: s__Weissella_cibaria', LEFSE_B),
 ('u1: s__Weissella_paramesenteroides', LEFSE_B),
 ('v1: s__Lactococcus_lactis', LEFSE_C),
 ('w1: s__Acetobacter_aceti', LEFSE_A),
 ('x1: s__Acetobacter_ascendens', LEFSE_A),
 ('y1: s__Acetobacter_ghanensis', LEFSE_B),
 ('z1: s__Acetobacter_lovanienesis', LEFSE_A),
 ('a2: s__Acetobacter_orientalis', LEFSE_C),
 ('b2: s__Acetobacter_oryzoeni', LEFSE_A),
 ('c2: s__Acetobacter_pasteurianus', LEFSE_A),
 ('d2: s__Acetobacter_senegalensis', LEFSE_A),
 ('e2: s__Acetobacter_sp_JWB', LEFSE_C),
 ('f2: s__Komagataeibacter_medellinensis', LEFSE_A),
 ('g2: s__Komagataeibacter_oboediens', LEFSE_A),
 ('h2: s__Komagataeibacter_xylinus', LEFSE_A),
 ('i2: s__Mixta_sp_X22927', LEFSE_A),
]
fungi_leg = [
 ('a: p__Ascomycota', LEFSE_B), ('b: p__Mucoromycota', LEFSE_C),
 ('c: c__Eurotiomycetes', LEFSE_A), ('d: c__Saccharomycetes', LEFSE_B),
 ('e: c__Schizosaccharomycetes', LEFSE_B), ('f: c__Mucoromycetes', LEFSE_C),
 ('g: o__Eurotiales', LEFSE_A), ('h: o__Saccharomycetales', LEFSE_B),
 ('i: o__Schizosaccharomycetales', LEFSE_B), ('j: o__Mucorales', LEFSE_C),
 ('k: f__Aspergillaceae', LEFSE_A), ('l: f__Pichiaceae', LEFSE_B),
 ('m: f__Saccharomycetaceae', LEFSE_B),
 ('n: f__Schizosaccharomycetaceae', LEFSE_B),
 ('o: f__Lichtheimiaceae', LEFSE_C), ('p: g__Aspergillus', LEFSE_A),
 ('q: g__Monascus', LEFSE_A), ('r: g__Pichia', LEFSE_B),
 ('s: g__Saccharomyces', LEFSE_B), ('t: g__Torulaspora', LEFSE_A),
 ('u: g__Schizosaccharomyces', LEFSE_B), ('v: g__Lichtheimia', LEFSE_C),
 ('w: s__Aspergillus_chevalieri', LEFSE_A),
 ('x: s__Aspergillus_nidulans', LEFSE_A),
 ('y: s__Monascus_pilosus', LEFSE_A),
 ('z: s__Monascus_purpureus', LEFSE_A),
 ('a1: s__Pichia_kluyveri', LEFSE_B), ('b1: s__Pichia_kudriavzevii', LEFSE_B),
 ('c1: s__Saccharomyces_cerevisiae', LEFSE_C),
 ('d1: s__Torulaspora_delbrueckii', LEFSE_A),
 ('e1: s__Zygosaccharomyces_bailii', LEFSE_B),
 ('f1: s__Schizosaccharomyces_pombe', LEFSE_B),
 ('g1: s__Lichtheimia_ramose', LEFSE_C),
]

fig = plt.figure(figsize=(16, 9.4))

def cab_legend(ax):
    for i, (lab, c) in enumerate([('C', LEFSE_C), ('A', LEFSE_A), ('B', LEFSE_B)]):
        ax.add_patch(Rectangle((0.70 + i * 0.085, 0.97), 0.025, 0.035,
                               facecolor=c, transform=ax.transAxes, clip_on=False))
        ax.text(0.732 + i * 0.085, 0.988, lab, fontsize=9, va='center',
                transform=ax.transAxes)

# ================================================================= (a) =====
ax = fig.add_axes([0.000, 0.505, 0.225, 0.475])
cladogram(ax, bact, 'root', span=(283, 593))
cab_legend(ax)
leg = fig.add_axes([0.228, 0.505, 0.295, 0.475]); leg.axis('off')
per = 25
for i, (lab, c) in enumerate(bact_leg):
    col, row = divmod(i, per)
    yy = 0.90 - row * 0.0345
    xx = 0.02 + col * 0.335
    leg.add_patch(Rectangle((xx, yy - 0.008), 0.016, 0.022, facecolor=c,
                            transform=leg.transAxes, clip_on=False))
    leg.text(xx + 0.024, yy + 0.003, lab, fontsize=5.4, style='italic',
             va='center', transform=leg.transAxes)
fig.text(0.538, 0.74, 'Relative Abundance (%)', rotation=90, ha='center',
         va='center', fontsize=8)
fig.text(0.000, 0.975, 'a', fontsize=15, fontweight='bold')

# ================================================================= (b) =====
axb = fig.add_axes([0.545, 0.545, 0.145, 0.42])
bact_tax = ['g_Acetobacter', 'g_Bacillus', 'g_Komagataeibacter',
            'g_Kroppenstedtia', 'g_Staphylococcus', 'g_Lactobacillus',
            'g_Corynebacterium', 'g_Weizmannia', 'g_Pediococcus',
            'g_Thermoactinomyces', 'g_Saccharomonospora', 'g_Weissella',
            'g_Limosilactobacillus', 'g_Caldibacillus', 'g_Levilactobacillus',
            'g_Leuconostoc', 'g_Companilactobacillus', 'g_Lactiplantibacillus',
            'g_Lentilactobacillus', 'g_Mammaliicoccus', 'Others']
bact_cols = ['#DFC31A', '#4CA26B', '#96CBE8', '#2E5EA8', '#F07C29', '#F3EECC',
             '#E8542A', '#F7D5A8', '#7E6BB5', '#C13B6B', '#DE6FA0', '#7C7B3A',
             '#EDE98F', '#A8CDA5', '#A8D3DE', '#4E79C0', '#C3B4DC', '#E7E7E7',
             '#F0946F', '#C9CFE0', '#B39DD8']
samples = ['B-1', 'B-2', 'B-3', 'A-1', 'A-2', 'A-3', 'C-1', 'C-2', 'C-3']
main = dict(zip(samples, [66, 65, 64.5, 71, 71.5, 70.5, 61, 60.5, 61]))
bacil = dict(zip(samples, [23, 23.5, 24, 13, 12.5, 13, 26, 26.5, 26]))
koma = dict(zip(samples, [0.3, 0.3, 0.3, 5, 5.5, 5, 0.5, 0.5, 0.5]))
minor = {'B': [1.2, 0.5, 0.6, 0.5, 0.4, 0.3, 0.3, 0.4, 0.3, 0.4, 0.5, 0.4,
               0.6, 0.4, 0.5, 0.4],
         'A': [1.8, 0.5, 0.6, 0.8, 0.8, 0.6, 0.6, 0.3, 0.4, 0.3, 0.4, 0.5,
               0.4, 0.6, 0.5, 0.4],
         'C': [0.8, 2.3, 0.6, 0.5, 0.5, 0.4, 0.3, 0.4, 0.3, 0.4, 0.5, 0.4,
               0.6, 0.4, 0.5, 0.4]}
comp = {t: [] for t in bact_tax}
for s in samples:
    m = minor[s[0]]
    vals = [main[s], bacil[s], koma[s]] + m          # taxa 0..17
    vals.insert(5, 2.4)                              # g_Lactobacillus pale band
    vals.append(100 - sum(vals))                     # Others
    for t, v in zip(bact_tax, vals):
        comp[t].append(v)
bottom = np.zeros(9)
for t, c in zip(bact_tax, bact_cols):
    axb.bar(np.arange(9), comp[t], bottom=bottom, width=0.82, color=c,
            edgecolor='white', linewidth=0.3, label=t)
    bottom += np.array(comp[t])
axb.set_xlim(-0.7, 8.7); axb.set_ylim(0, 100)
axb.set_xticks(np.arange(9)); axb.set_xticklabels(samples, fontsize=10,
                                                  fontweight='bold')
axb.set_yticks(range(0, 101, 10))
axb.set_yticklabels(range(0, 101, 10), fontsize=9)
axb.set_ylabel('Relative Abundance (%)', fontsize=10)
axb.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=6.6,
           handlelength=1.0, handleheight=1.0, labelspacing=0.30, frameon=False)
fig.text(0.523, 0.975, 'b', fontsize=15, fontweight='bold')

# ================================================================= (c) =====
axc = fig.add_axes([0.795, 0.545, 0.15, 0.42])
pts = {'A': [(0.08, -0.42), (0.12, -0.45), (0.10, -0.38)],
       'B': [(0.30, 0.30), (0.325, 0.27), (0.335, 0.335)],
       'C': [(0.16, 0.12), (0.185, 0.09), (0.14, 0.07)]}
cols = {'A': PCA_A, 'B': PCA_B, 'C': PCA_C}
ells = {'A': (0.05, 0.06, -35), 'B': (0.05, 0.07, -40), 'C': (0.05, 0.055, -50)}
for g in 'ACB':
    xs, ys = zip(*pts[g])
    axc.scatter(xs, ys, s=48, color=cols[g], zorder=3)
    w, h, ang = ells[g]
    axc.add_patch(Ellipse((np.mean(xs), np.mean(ys)), w * 2.4, h * 2.4, angle=ang,
                          facecolor=cols[g], alpha=0.35, lw=0, zorder=1))
axc.set_xlim(-0.02, 0.37); axc.set_ylim(-0.50, 0.42)
axc.set_xticks([0.0, 0.1, 0.2, 0.3])
axc.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4])
axc.tick_params(labelsize=9)
axc.set_xlabel('PCA1 (79.08%)', fontsize=10)
axc.set_ylabel('PCA2 (10.9%)', fontsize=10)
axc.set_title('PCA Analysis -Bacteria', fontsize=11, fontweight='bold')
axc.text(0.03, 0.955, '$P$=0.005\n$R$=0.8864', transform=axc.transAxes,
         fontsize=9, va='top')
axc.legend(handles=[plt.Line2D([], [], marker='o', ls='', color=cols[g],
                              markersize=7, label=g) for g in 'ABC'],
           title='Samples', loc='center left', bbox_to_anchor=(1.02, 0.5),
           fontsize=8, title_fontsize=9, frameon=False)
fig.text(0.772, 0.975, 'c', fontsize=15, fontweight='bold')

# ================================================================= (d) =====
ax = fig.add_axes([0.000, 0.015, 0.225, 0.45])
cladogram(ax, fungi, 'root', span=(10, 370))
cab_legend(ax)
leg = fig.add_axes([0.228, 0.015, 0.295, 0.45]); leg.axis('off')
for i, (lab, c) in enumerate(fungi_leg):
    col, row = divmod(i, 25)
    yy = 0.90 - row * 0.0345
    xx = 0.02 + col * 0.335
    leg.add_patch(Rectangle((xx, yy - 0.008), 0.016, 0.022, facecolor=c,
                            transform=leg.transAxes, clip_on=False))
    leg.text(xx + 0.024, yy + 0.003, lab, fontsize=5.4, style='italic',
             va='center', transform=leg.transAxes)
fig.text(0.538, 0.245, 'Relative Abundance (%)', rotation=90, ha='center',
         va='center', fontsize=8)
fig.text(0.000, 0.485, 'd', fontsize=15, fontweight='bold')

# ================================================================= (e) =====
axe = fig.add_axes([0.545, 0.065, 0.145, 0.38])
fun_tax = ['g_Pichia', 'g_Saccharomyces', 'g_Schizosaccharomyces',
           'g_Zygosaccharomyces', 'g_Monascus', 'g_Torulaspora',
           'g_Aspergillus', 'g_Wickerhamomyces', 'g_Rhizopus',
           'g_Plectosphaerella', 'g_Alternaria', 'g_Lichtheimia',
           'g_Camarosporomyces', 'g_Cladosporium', 'g_Saccharomycopsis',
           'g_Penicillium', 'g_Others']
fun_cols = ['#E5C31A', '#4CA26B', '#96CBE8', '#2E5EA8', '#F07C29', '#F3EECC',
            '#E8542A', '#F7D5A8', '#7E6BB5', '#C13B6B', '#DE6FA0', '#7C7B3A',
            '#EDE98F', '#A8CDA5', '#A8D3DE', '#4E79C0', '#B39DD8']
pichia = dict(zip(samples, [48, 48.5, 47.5, 47, 47.5, 46.5, 72, 72.5, 71.5]))
sacch = dict(zip(samples, [44, 44, 44.5, 22, 22, 22, 22, 21.5, 22]))
sch = dict(zip(samples, [0.8, 0.8, 0.8, 6.5, 6.5, 6, 0.5, 0.5, 0.5]))
zyg = dict(zip(samples, [0.6, 0.6, 0.6, 5, 5, 5, 0.4, 0.4, 0.4]))
mon = dict(zip(samples, [0.5, 0.5, 0.5, 8, 7.5, 8, 0.4, 0.4, 0.4]))
tor = dict(zip(samples, [1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 0.8, 0.8, 0.8]))
asp = dict(zip(samples, [0.5, 0.5, 0.5, 1.3, 1.3, 1.3, 0.5, 0.5, 0.5]))
fcomp = {t: [] for t in fun_tax}
for s in samples:
    vals = {t: 0.25 for t in fun_tax}
    vals['g_Pichia'] = pichia[s]; vals['g_Saccharomyces'] = sacch[s]
    vals['g_Schizosaccharomyces'] = sch[s]
    vals['g_Zygosaccharomyces'] = zyg[s]; vals['g_Monascus'] = mon[s]
    vals['g_Torulaspora'] = tor[s]; vals['g_Aspergillus'] = asp[s]
    vals['g_Others'] = 100 - sum(vals.values())
    for t in fun_tax:
        fcomp[t].append(vals[t])
bottom = np.zeros(9)
for t, c in zip(fun_tax, fun_cols):
    axe.bar(np.arange(9), fcomp[t], bottom=bottom, width=0.82, color=c,
            edgecolor='white', linewidth=0.3, label=t)
    bottom += np.array(fcomp[t])
axe.set_xlim(-0.7, 8.7); axe.set_ylim(0, 100)
axe.set_xticks(np.arange(9)); axe.set_xticklabels(samples, fontsize=10,
                                                  fontweight='bold')
axe.set_yticks(range(0, 101, 10))
axe.set_yticklabels(range(0, 101, 10), fontsize=9)
axe.set_ylabel('Relative Abundance (%)', fontsize=10)
axe.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=6.6,
           handlelength=1.0, handleheight=1.0, labelspacing=0.30, frameon=False)
fig.text(0.523, 0.50, 'e', fontsize=15, fontweight='bold')

# ================================================================= (f) =====
axf = fig.add_axes([0.795, 0.065, 0.15, 0.38])
pts = {'A': [(0.045, -0.04), (0.075, -0.06), (0.06, 0.02)],
       'B': [(0.25, -0.20), (0.265, -0.24), (0.245, -0.28)],
       'C': [(0.20, 0.26), (0.215, 0.23), (0.19, 0.30)]}
ells = {'A': (0.05, 0.06, 40), 'B': (0.05, 0.07, 35), 'C': (0.05, 0.06, -30)}
for g in 'ACB':
    xs, ys = zip(*pts[g])
    axf.scatter(xs, ys, s=48, color=cols[g], zorder=3)
    w, h, ang = ells[g]
    axf.add_patch(Ellipse((np.mean(xs), np.mean(ys)), w * 2.4, h * 2.4, angle=ang,
                          facecolor=cols[g], alpha=0.35, lw=0, zorder=1))
axf.set_xlim(0.0, 0.32); axf.set_ylim(-0.38, 0.38)
axf.set_xticks([0.0, 0.1, 0.2, 0.3]); axf.set_yticks([-0.25, 0.0, 0.25])
axf.tick_params(labelsize=9)
axf.set_xlabel('PCA1 (75.11%)', fontsize=10)
axf.set_ylabel('PCA2 (14.15%)', fontsize=10)
axf.set_title('PCA Analysis -Fungi', fontsize=11, fontweight='bold')
axf.text(0.03, 0.955, '$P$=0.004\n$R$=0.7534', transform=axf.transAxes,
         fontsize=9, va='top')
axf.legend(handles=[plt.Line2D([], [], marker='o', ls='', color=cols[g],
                              markersize=7, label=g) for g in 'ABC'],
           title='Samples', loc='center left', bbox_to_anchor=(1.02, 0.5),
           fontsize=8, title_fontsize=9, frameon=False)
fig.text(0.772, 0.50, 'f', fontsize=15, fontweight='bold')

save(fig, 'fig6')
