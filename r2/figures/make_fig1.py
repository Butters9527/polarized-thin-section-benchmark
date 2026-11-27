# -*- coding: utf-8 -*-
"""Figure 1: 8-frame extinction sequence of one NJU thin section (PPL + XPL 0-90 deg).

The NJU photographs carry a small burned-in scale box (red 500 um bar in a white box).
The script detects that bar in every frame, uses its length in pixels as the scale
calibration, covers the original box and draws a new, legible scale bar.

usage: python make_fig1.py --nju <NJU folder> --out <output folder> [--section 火1二辉橄榄岩2]
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib import patheffects as pe
from matplotlib.patches import FancyBboxPatch, Rectangle

FONT = "Times New Roman"
FS = 12          # titles and panel labels
FS_BAR = 11      # scale-bar label
plt.rcParams.update({"font.family": "serif", "font.serif": [FONT, "Liberation Serif"], "font.size": FS,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600})
TITLES = ["PPL", "XPL@0°", "XPL@15°", "XPL@30°", "XPL@45°", "XPL@60°", "XPL@75°", "XPL@90°"]
LETTERS = "abcdefgh"

# Times New Roman is required: stop instead of silently falling back to another font
if "Times New Roman" not in {f.name for f in font_manager.fontManager.ttflist} and not os.environ.get("ALLOW_FONT_FALLBACK"):
    raise SystemExit("Times New Roman is not installed, so the figure would be drawn in another font. "
                     "Install Times New Roman (standard on Windows) and run again.")


def find_frames(root, section):
    want = {f"{section}-{p}": p for p in range(1, 9)}
    found = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            stem, ext = os.path.splitext(fn)
            if ext.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp") and stem in want:
                found[want[stem]] = os.path.join(dp, fn)
    missing = [p for p in range(1, 9) if p not in found]
    if missing:
        sys.exit(f"frames {missing} of {section} not found under {root}")
    return [found[p] for p in range(1, 9)]


def longest_run(row, gap=2):
    best = cur = miss = 0
    for v in row:
        if v:
            cur += 1 + miss if cur else 1
            miss = 0
            best = max(best, cur)
        elif cur:
            miss += 1
            if miss > gap:
                cur = miss = 0
    return best


def _flood(mask, seeds):
    """connected component of True pixels in `mask` reachable from `seeds` (4-neighbour)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    stack = [(y, x) for y, x in seeds if 0 <= y < h and 0 <= x < w and mask[y, x]]
    for y, x in stack:
        seen[y, x] = True
    while stack:
        y, x = stack.pop()
        for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                seen[yy, xx] = True
                stack.append((yy, xx))
    return seen


def detect_scale_box(a):
    """Return (bar_length_px, (x0, y0, x1, y1)) of the burned-in scale box, or None.
    The bar is the longest horizontal run of red pixels in the bottom-right corner;
    the label box is the white region connected to the pixels just above/below the bar."""
    H, W, _ = a.shape
    oy, ox = int(0.70 * H), int(0.55 * W)
    reg = a[oy:, ox:].astype(int)
    r, g, b = reg[..., 0], reg[..., 1], reg[..., 2]
    red = (r > 140) & (g < 110) & (b < 110) & (r - g > 60)
    runs = np.array([longest_run(red[y]) for y in range(red.shape[0])])
    L = runs.max()
    if L < 0.025 * W:                      # no bar-like red line
        return None
    rows = np.nonzero(runs >= 0.8 * L)[0]
    # keep the block of rows around the longest run
    y_best = int(np.argmax(runs))
    rows = rows[np.abs(rows - y_best) <= 6]
    xs = np.concatenate([np.nonzero(red[y])[0] for y in rows])
    xs = xs[(xs >= np.median(xs) - L) & (xs <= np.median(xs) + L)]
    bx0, bx1, by0, by1 = xs.min(), xs.max(), rows.min(), rows.max()
    white = reg.min(axis=2) > 215
    seeds = [(by0 - k, x) for k in (2, 3, 4) for x in range(bx0, bx1 + 1, 3)] + \
            [(by1 + k, x) for k in (2, 3) for x in range(bx0, bx1 + 1, 3)]
    comp = _flood(white, seeds)
    x0, x1, y0, y1 = bx0, bx1, by0, by1
    if comp.any():
        cys, cxs = np.nonzero(comp)
        # ignore runaway fills into large white grains
        if (cxs.max() - cxs.min()) < 3 * L and (cys.max() - cys.min()) < 1.2 * L:
            x0, x1 = min(x0, cxs.min()), max(x1, cxs.max())
            y0, y1 = min(y0, cys.min()), max(y1, cys.max())
    return int(L), (int(x0 + ox), int(y0 + oy), int(x1 + ox), int(y1 + oy))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nju", required=True, help="NJU dataset folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--section", default="火1二辉橄榄岩2", help="section key (Figure 1 uses 火1二辉橄榄岩2)")
    ap.add_argument("--bar-um", type=float, default=500.0, help="length of the burned-in bar in micrometres")
    ap.add_argument("--bar-px", type=float, default=None, help="override: bar length in pixels")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print("font used:", font_manager.findfont(font_manager.FontProperties(family="serif")))

    paths = find_frames(a.nju, a.section)
    imgs = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
    H, W, _ = imgs[0].shape
    det = [detect_scale_box(im) for im in imgs]
    for p, d in zip(paths, det):
        print(os.path.basename(p), "detected bar:", "none" if d is None else f"{d[0]} px, box {d[1]}")
    lens = [d[0] for d in det if d is not None]
    if a.bar_px:
        bar_px = a.bar_px
    elif lens:
        bar_px = float(np.median(lens))
    else:
        sys.exit("scale bar not detected; rerun with --bar-px <length of the 500 um bar in pixels>")
    print(f"scale: {a.bar_um:g} um = {bar_px:.1f} px  ({a.bar_um / bar_px:.3f} um/px)")

    # layout: 2 rows x 4 panels, 7.2 in wide
    figw, gap, left, right = 7.2, 0.05, 0.02, 0.02
    pw = (figw - left - right - 3 * gap) / 4
    ph = pw * H / W
    title_h, row_gap = 0.26, 0.06
    figh = 2 * (ph + title_h) + row_gap + 0.02
    fig = plt.figure(figsize=(figw, figh))
    ppi = W / pw                                   # image pixels per printed inch
    pt = ppi / 72.0                                # image pixels per point
    for k, im in enumerate(imgs):
        r, c = divmod(k, 4)
        x = left + c * (pw + gap)
        y = figh - (r + 1) * (ph + title_h) - r * row_gap
        ax = fig.add_axes([x / figw, y / figh, pw / figw, ph / figh])
        ax.imshow(im, interpolation="lanczos")
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
        ax.set_title(TITLES[k], fontsize=FS, pad=3)
        ax.text(0.035, 0.955, f"({LETTERS[k]})", transform=ax.transAxes, ha="left", va="top", fontsize=FS,
                fontweight="bold", color="white", path_effects=[pe.withStroke(linewidth=2.2, foreground="black")])
        # new scale bar, bottom right
        pad, txt_h, bar_t = 3.5 * pt, FS_BAR * pt, 1.6 * pt
        txt_w = 0.52 * FS_BAR * 6 * pt               # approx. width of "500 μm"
        bw = max(bar_px, txt_w) + 2 * pad
        bh = txt_h + 2.5 * pt + bar_t + 2 * pad
        bx1, by1 = W - 0.02 * W, H - 0.025 * H
        bx0, by0 = bx1 - bw, by1 - bh
        d = det[k]
        if d is not None:                           # make sure the old box is covered
            ox0, oy0, ox1, oy1 = d[1]
            m = 3
            cx0, cy0 = min(bx0, ox0 - m), min(by0, oy0 - m)
            cx1, cy1 = max(bx1, ox1 + m), max(by1, oy1 + m)
            if (cx0, cy0, cx1, cy1) != (bx0, by0, bx1, by1):
                ax.add_patch(Rectangle((cx0, cy0), cx1 - cx0, cy1 - cy0, fc="white", ec="none", zorder=3))
        ax.add_patch(FancyBboxPatch((bx0, by0), bw, bh, boxstyle=f"round,pad=0,rounding_size={2 * pt}",
                                    fc="white", ec="none", alpha=0.92, zorder=4))
        cx = (bx0 + bx1) / 2
        ax.add_patch(Rectangle((cx - bar_px / 2, by1 - pad - bar_t), bar_px, bar_t, fc="black", ec="none", zorder=5))
        ax.text(cx, by1 - pad - bar_t - 2.0 * pt, f"{a.bar_um:g} μm", ha="center", va="bottom", fontsize=FS_BAR,
                zorder=6)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, f"Figure1_sequence.{ext}"))
    plt.close(fig)

    # check sheet: original corners with the detected box
    fig, axs = plt.subplots(2, 4, figsize=(10, 4))
    for k, (ax, im) in enumerate(zip(axs.flat, imgs)):
        ax.imshow(im[int(0.7 * H):, int(0.55 * W):]); ax.set_title(TITLES[k], fontsize=9); ax.axis("off")
        if det[k] is not None:
            x0, y0, x1, y1 = det[k][1]
            ax.add_patch(Rectangle((x0 - int(0.55 * W), y0 - int(0.7 * H)), x1 - x0, y1 - y0, fill=False,
                                   ec="cyan", lw=1))
    fig.suptitle(f"Detected scale boxes (cyan); bar = {bar_px:.0f} px = {a.bar_um:g} um", fontsize=10)
    fig.savefig(os.path.join(a.out, "Figure1_scalebar_check.png"), dpi=150)
    print("written:", os.path.join(a.out, "Figure1_sequence.png"))


if __name__ == "__main__":
    main()
