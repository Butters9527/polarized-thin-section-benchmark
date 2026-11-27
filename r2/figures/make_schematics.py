import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FS = 11
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif'],
                     'font.size': FS, 'mathtext.fontset': 'stix', 'savefig.dpi': 600,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
ap = argparse.ArgumentParser(); ap.add_argument('--out', default='.')
OUT = ap.parse_args().out; os.makedirs(OUT, exist_ok=True)
from matplotlib import font_manager
# Times New Roman is required: stop instead of silently falling back to another font
if "Times New Roman" not in {f.name for f in font_manager.fontManager.ttflist} and not os.environ.get("ALLOW_FONT_FALLBACK"):
    raise SystemExit("Times New Roman is not installed, so the figure would be drawn in another font. "
                     "Install Times New Roman (standard on Windows) and run again.")
print('font used:', font_manager.findfont(font_manager.FontProperties(family='serif')))
FILL = {'data': '#e8f1fb', 'split': '#fdf3e1', 'input': '#e6f5ee', 'model': '#f0ebf8', 'out': '#fbe9e7',
        'enc': '#fdf3e1'}
EDGE = '#3d5a80'


def box(ax, x, y, w, h, text, kind, fs=FS, lw=0.9, ls='-', bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.06', fc=FILL[kind],
                                ec=EDGE, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, linespacing=1.12,
            fontweight='bold' if bold else 'normal')


def arrow(ax, x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color='#333333', lw=0.9, shrinkA=0, shrinkB=0,
                                mutation_scale=9))


def label(ax, x, y, t):
    ax.text(x, y, t, ha='left', va='center', fontsize=FS, fontweight='bold', color='#222')


# ---------------------------------------------------------------- Figure 2: pipeline (top to bottom)
Wd, Ht = 7.2, 5.85
fig = plt.figure(figsize=(Wd, Ht))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, Wd); ax.set_ylim(0, Ht); ax.axis('off')
X0, X1 = 0.05, Wd - 0.05
# row 1: data
box(ax, 1.0, 4.82, 5.2, 0.98,
    'NJU dataset: 324 sections of 108 rock types (3 sister sections each)\n'
    '321 usable sections (3 excluded for a missing frame)\n'
    'Each section: 1 PPL frame + 7 XPL frames (0°–90°, 15° steps)', 'data')
# row 2: partitioning
label(ax, X0, 4.62, 'Partitioning of whole sections')
P = ['Rock-name\ngrouped CV\n103 groups\n10 × 5 (primary)', 'Rock-name +\nlocality grouped\nCV, 77 groups\n10 × 5',
     'Section-level\nrandom CV\n\n10 × 5', 'Fixed split\n256 / 32 / 33\n\n10 seeds']
w, gap = 1.66, 0.12
xs = [X0 + i * (w + gap) for i in range(4)]
xs = [x + (X1 - X0 - (4 * w + 3 * gap)) / 2 for x in xs]
ax.plot([3.6, 3.6], [4.82, 4.40], color='#333333', lw=0.9)
ax.plot([xs[0] + w / 2, xs[3] + w / 2], [4.40, 4.40], color='#333333', lw=0.9)
for i, (x, t) in enumerate(zip(xs, P)):
    box(ax, x, 3.1, w, 1.15, t, 'split', lw=1.6 if i == 0 else 0.9)
    arrow(ax, x + w / 2, 4.40, x + w / 2, 4.25)
arrow(ax, Wd / 2, 2.75, Wd / 2, 2.38)
ax.text(Wd / 2, 2.93, 'CV: inner early-stopping split of about 15% of the training sections in every fold',
        ha='center', va='center', fontsize=FS - 1, style='italic')
# row 3: inputs
label(ax, X0, 2.56, 'Input representation')
I = ['PPL\n(1 frame)', 'XPL@0°\n(1 frame)', 'PPL+XPL\n(6 channels)', '8-frame sequence\n(XPL7-Attn: 7 XPL)']
for x, t in zip(xs, I):
    box(ax, x, 1.75, w, 0.6, t, 'input')
# row 4: models
label(ax, X0, 1.45, 'Model')
mx1 = xs[0]; mw1 = xs[2] + w - xs[0]
box(ax, mx1, 0.62, mw1, 0.62, 'Single-frame and early-fusion models (Figure 3a,b)\n'
    'ResNet-50, EfficientNet-B0, ViT-B/16, Swin-T', 'model')
box(ax, xs[3], 0.62, w, 0.62, 'Sequence models\nResNet-50 (Figure 3c)', 'model')
for x in xs[:3]:
    arrow(ax, x + w / 2, 1.75, x + w / 2, 1.24)
arrow(ax, xs[3] + w / 2, 1.75, xs[3] + w / 2, 1.24)
# row 5: output
box(ax, 1.1, 0.04, 5.0, 0.38, 'One prediction per section: igneous, metamorphic or sedimentary', 'out')
arrow(ax, mx1 + mw1 / 2, 0.62, 3.0, 0.42)
arrow(ax, xs[3] + w / 2, 0.62, 5.0, 0.42)
fig.savefig(os.path.join(OUT, 'Figure2_pipeline.png')); fig.savefig(os.path.join(OUT, 'Figure2_pipeline.pdf'))
plt.close(fig)

# ---------------------------------------------------------------- Figure 3: architectures
Wd, Ht = 7.2, 6.3
fig = plt.figure(figsize=(Wd, Ht))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, Wd); ax.set_ylim(0, Ht); ax.axis('off')
CW, GAP = 2.28, 0.2


def column(x, title, items, top=5.72):
    ax.text(x + CW / 2, 6.02, title, ha='center', va='center', fontsize=FS, fontweight='bold')
    y = top; prev = None
    for text, kind, h in items:
        y -= h
        box(ax, x, y, CW, h, text, kind, ls='--' if kind == 'enc' else '-')
        if prev is not None:
            arrow(ax, x + CW / 2, prev, x + CW / 2, y + h)
        prev = y; y -= GAP


h1, h2, h3 = 0.42, 0.62, 0.82
column(0.06, '(a) PPL, XPL', [
    ('PPL or XPL@0° frame\n3 × 224 × 224', 'input', h2),
    ('Backbone (ResNet-50,\nEfficientNet-B0,\nViT-B/16 or Swin-T)', 'model', h3),
    ('Global pooling', 'model', h1),
    ('Dropout (p = 0.3)', 'model', h1),
    ('Linear → 3 classes', 'out', h1),
])
column(2.46, '(b) PPL+XPL', [
    ('PPL + XPL@0° concatenated\n6 × 224 × 224', 'input', h2),
    ('First layer expanded\nto 6 channels, Eq. (1)', 'model', h2),
    ('Remaining\nbackbone layers', 'model', h2),
    ('Global pooling', 'model', h1),
    ('Dropout (p = 0.3)', 'model', h1),
    ('Linear → 3 classes', 'out', h1),
])
column(4.86, '(c) Sequence models', [
    ('8 frames: PPL + XPL\n0°–90° (XPL7-Attn:\n7 XPL frames)', 'input', h3),
    ('Shared ResNet-50\nper frame → 2048-d\nfeature per frame', 'model', h3),
    ('+ Rotational positional\nencoding (RPE only)', 'enc', h2),
    ('Self-attention, 8 heads\n(not in Seq-Mean)', 'model', h2),
    ('Mean over frames', 'model', h1),
    ('LayerNorm, Dropout (0.3)', 'model', h1),
    ('Linear → 3 classes', 'out', h1),
])
fig.savefig(os.path.join(OUT, 'Figure3_models.png')); fig.savefig(os.path.join(OUT, 'Figure3_models.pdf'))
plt.close(fig)
print('ok')
