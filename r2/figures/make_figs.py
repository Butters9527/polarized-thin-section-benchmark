# -*- coding: utf-8 -*-
"""Figures 4-6 of the R2 manuscript from the R2 results (r2_results.zip unpacked).

usage: python make_figs.py --r2 <folder with analysis/, index/, results/> --out <output folder>
"""
import argparse
import collections
import csv
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

ap = argparse.ArgumentParser()
ap.add_argument('--r2', default=None, help='folder with index/ and results/ (unpacked r2_results.zip)')
ap.add_argument('--index', default=None, help='folder with sections.csv and folds.csv (default: <r2>/index)')
ap.add_argument('--results', default=None, help='folder with <setting>/<model>/*.json (default: <r2>/results)')
ap.add_argument('--out', default='.')
A = ap.parse_args()
OUT = A.out
IDX = A.index or os.path.join(A.r2, 'index')
RES = A.results or os.path.join(A.r2, 'results')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif'], 'font.size': 12,
                     'axes.titlesize': 12, 'axes.labelsize': 12, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
                     'legend.fontsize': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#555', 'axes.linewidth': 0.8, 'savefig.dpi': 600,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
# Times New Roman is required: stop instead of silently falling back to another font
if "Times New Roman" not in {f.name for f in font_manager.fontManager.ttflist} and not os.environ.get("ALLOW_FONT_FALLBACK"):
    raise SystemExit("Times New Roman is not installed, so the figure would be drawn in another font. "
                     "Install Times New Roman (standard on Windows) and run again.")
print('font used:', font_manager.findfont(font_manager.FontProperties(family='serif')))
C = {'ppl': '#2a78d6', 'xpl0': '#eb6834', 'dual': '#1baf7a', 'other': '#eda100'}
MK = {'ppl': 'o', 'xpl0': 's', 'dual': '^'}
LAB = {'ppl': 'PPL', 'xpl0': 'XPL', 'dual': 'PPL+XPL'}

secs = {r['section_key']: r for r in csv.DictReader(open(os.path.join(IDX, 'sections.csv'), encoding='utf-8-sig'))}
folds = collections.defaultdict(dict)
for r in csv.DictReader(open(os.path.join(IDX, 'folds.csv'), encoding='utf-8-sig')):
    folds[(r['protocol'], int(r['rep']), int(r['fold']))][r['section_key']] = r['role']


def load(setting, model):
    files = sorted(glob.glob(os.path.join(RES, setting, model, '*.json')))
    assert files, f'no results for {setting}/{model} under {RES}'
    return [json.load(open(p, encoding='utf-8')) for p in files]


def pooled_by_rep(setting, model):
    d = collections.defaultdict(list)
    for r in load(setting, model):
        for t in r['test']:
            d[r['job']['rep']].append(t['pred_center'] == t['label'])
    return np.array([np.mean(v) for _, v in sorted(d.items())]) * 100


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'))
    plt.close(fig)
    print('written:', os.path.join(OUT, name + '.png'))


# ---------- Figure 4: protocols + sister-section dose-response (7.2 in wide)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={'width_ratios': [1.25, 1]})
prots = [('slide_orig', 'Section-level\nrandom'), ('grouped_orig', 'Rock-name\ngrouped'),
         ('locality_orig', 'Rock-name +\nlocality grouped')]
w = 0.24
x = np.arange(3)
for i, m in enumerate(['ppl', 'xpl0', 'dual']):
    v = [pooled_by_rep(s, m) for s, _ in prots]
    ax[0].bar(x + (i - 1) * w, [a.mean() for a in v], w * 0.9, yerr=[a.std(ddof=1) for a in v], color=C[m],
              label=LAB[m], capsize=2, error_kw={'lw': 0.8, 'ecolor': '#333'})
ax[0].set_xticks(x); ax[0].set_xticklabels([p[1] for p in prots]); ax[0].set_ylim(60, 92)
ax[0].set_ylabel('Overall accuracy (%)'); ax[0].legend(frameon=False, loc='upper right')
ax[0].set_title('(a)', loc='left'); ax[0].yaxis.grid(True, color='#e5e5e5', lw=0.6); ax[0].set_axisbelow(True)
rid = lambda k: (secs[k]['rock_class'], secs[k]['rock_id'])
sis = {k: [o for o in secs if o != k and rid(o) == rid(k)] for k in secs}
for m in ['ppl', 'xpl0', 'dual']:
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in load('slide_orig', m):
        mm = folds[('slide', r['job']['rep'], r['job']['fold'])]
        for t in r['test']:
            n = sum(1 for o in sis[t['section_key']] if mm[o] == 'train')
            per[n][r['job']['rep']].append(t['pred_center'] == t['label'])
    mu, sd = [], []
    for n in (0, 1, 2):
        v = np.array([np.mean(per[n][rp]) for rp in sorted(per[n])]) * 100
        mu.append(v.mean()); sd.append(v.std(ddof=1))
    off = {'ppl': -0.06, 'xpl0': 0, 'dual': 0.06}[m]
    ax[1].errorbar(np.array([0, 1, 2]) + off, mu, yerr=sd, color=C[m], marker=MK[m], ms=6, lw=2, capsize=2,
                   label=LAB[m])
ax[1].set_xticks([0, 1, 2]); ax[1].set_xlabel('Sister sections in training set')
ax[1].set_ylabel('Overall accuracy (%)'); ax[1].set_ylim(60, 92); ax[1].set_xlim(-0.3, 2.3)
ax[1].set_title('(b)', loc='left'); ax[1].yaxis.grid(True, color='#e5e5e5', lw=0.6); ax[1].set_axisbelow(True)
ax[1].legend(frameon=False, loc='lower right')
fig.tight_layout()
save(fig, 'Figure4_protocols')

# ---------- Figure 5: pooled confusion matrices, rock-name-grouped CV (2 x 2, 7.2 in wide)
models = [('ppl', '(a) PPL, ResNet-50'), ('dual', '(b) PPL+XPL, ResNet-50'),
          ('seq_rpe90', '(c) Seq-Attn-RPE, ResNet-50'), ('swint_dual', '(d) PPL+XPL, Swin-T')]
fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.4))
for a, (m, t) in zip(axs.flat, models):
    cm = np.zeros((3, 3))
    for r in load('grouped_orig', m):
        for q in r['test']:
            cm[q['label'], q['pred_center']] += 1
    cm /= 10                                     # mean per repetition (each section tested once per repetition)
    rn = cm / cm.sum(1, keepdims=True) * 100
    a.imshow(rn, cmap='Blues', vmin=0, vmax=100)
    for i in range(3):
        for j in range(3):
            a.text(j, i, f'{rn[i, j]:.1f}%\n({cm[i, j]:.1f})', ha='center', va='center', fontsize=11,
                   color='white' if rn[i, j] > 55 else '#222')
    a.set_xticks(range(3)); a.set_yticks(range(3))
    a.set_xticklabels(['Ign.', 'Met.', 'Sed.']); a.set_yticklabels(['Ign.', 'Met.', 'Sed.'])
    a.set_xlabel('Predicted'); a.set_ylabel('True'); a.set_title(t)
    for s in a.spines.values():
        s.set_visible(False)
    a.tick_params(length=0)
fig.tight_layout()
save(fig, 'Figure5_confusion')

# ---------- Figure 6: training curves on the fixed split, 10 seeds (7.2 in wide)
mods = [('ppl', 'PPL'), ('dual', 'PPL+XPL'), ('seq_attn', 'Seq-Attn'), ('seq_rpe90', 'Seq-Attn-RPE')]
fig, axs = plt.subplots(2, 4, figsize=(7.2, 4.9), sharex=True)
for j, (m, t) in enumerate(mods):
    for r in load('fixed_orig', m):
        h = r['history']
        e = [x['epoch'] for x in h]
        axs[0, j].plot(e, [x['train_loss'] for x in h], color=C['ppl'], lw=0.7, alpha=0.7)
        axs[0, j].plot(e, [x['val_loss'] for x in h], color=C['xpl0'], lw=0.7, alpha=0.7)
        axs[1, j].plot(e, [100 * x['train_acc'] for x in h], color=C['ppl'], lw=0.7, alpha=0.7)
        axs[1, j].plot(e, [100 * x['val_acc'] for x in h], color=C['xpl0'], lw=0.7, alpha=0.7)
        b = r['best_epoch']
        vb = [x for x in h if x['epoch'] == b][0]
        axs[1, j].plot([b], [100 * vb['val_acc']], 'o', color=C['xpl0'], ms=3.5, mec='white', mew=0.5)
    axs[0, j].set_title(t)
    axs[1, j].set_xlabel('Epoch')
    axs[0, j].set_ylim(0, 2.3)
    axs[1, j].set_ylim(30, 102)
    axs[1, j].set_xticks([0, 10, 20, 30])
    for a in axs[:, j]:
        a.yaxis.grid(True, color='#e5e5e5', lw=0.6)
        a.set_axisbelow(True)
    if j:
        axs[0, j].set_yticklabels([]); axs[1, j].set_yticklabels([])
axs[0, 0].set_ylabel('Loss'); axs[1, 0].set_ylabel('Accuracy (%)')
fig.legend([Line2D([], [], color=C['ppl'], lw=2), Line2D([], [], color=C['xpl0'], lw=2),
            Line2D([], [], marker='o', color=C['xpl0'], lw=0, ms=5)],
           ['Training', 'Early-stopping set', 'Selected checkpoint'], loc='upper center', ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=0.6)
save(fig, 'Figure6_curves')
