#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_fig10_gradcam_fixed.py

把两个 1×4 Grad-CAM 素材图拼成论文 Figure 10（2×4）。
关键修复：
1) 不再把 1×4 素材机械地按 1/4 宽度后直接贴图；
2) 每个四分区内先自动检测“真正的小图边界”（去掉 Matplotlib 留下的白边）；
3) 8 个图号框都以“小图实际边界”为基准定位：
      左边距 = LABEL_MARGIN_PX
      上边距 = LABEL_MARGIN_PX
   因此 (a)–(h) 的灰色半透明方框到各自小图左、上边界的距离完全一致；
4) 图号文字严格置于方框中心；
5) 全部 8 张小图统一缩放为同一尺寸，不发生拉伸变形。

用法：
    python generate_fig10_gradcam_fixed.py \
        --top gradcam_igneous_4_1_correct.png \
        --bottom gradcam_metamorphic_2_2_wrong.png \
        --out fig10_gradcam_fixed.png
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# =========================
# 可调参数
# =========================
CANVAS_W = 2048

PANEL_SIZE = 440       # 每张小图最终宽/高（正方形）
LEFT_MARGIN = 95       # 给纵向行标签预留空间
RIGHT_MARGIN = 28
COL_GAP = 35
TOP_TITLE_H = 50
ROW_GAP = 34
BOTTOM_MARGIN = 20

# 图号框参数：最关键
LABEL_MARGIN_PX = 8    # 方框边界到小图左/上边界的统一距离
LABEL_BOX_PX = 34      # 灰色半透明方框边长
LABEL_ALPHA = 145      # 0~255，越大越不透明
LABEL_FONT_SIZE = 22

TITLE_FONT_SIZE = 25
ROW_FONT_SIZE = 22

TITLES = ["PPL image", "PPL + Grad-CAM", "XPL image", "XPL + Grad-CAM"]
TOP_ROW_TEXT = "igneous 4_1\ncorrect (p = 0.95)"
BOTTOM_ROW_TEXT = "metamorphic 2_2\nmisclassified (p = 0.77)"


def find_font(size: int, bold: bool = False):
    """优先 Times New Roman；找不到时回退到常见 serif 字体。"""
    candidates = []
    if bold:
        candidates += [
            "C:/Windows/Fonts/timesbd.ttf",
            "/Library/Fonts/Times New Roman Bold.ttf",
            "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        ]
    else:
        candidates += [
            "C:/Windows/Fonts/times.ttf",
            "/Library/Fonts/Times New Roman.ttf",
            "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def content_bbox(img: Image.Image, white_threshold: int = 248):
    """
    找出真正图像内容的 bbox。
    素材由 matplotlib 输出，外围背景接近纯白；因此按“非白像素”找内容边界。
    """
    arr = np.asarray(img.convert("RGB"))
    mask = np.any(arr < white_threshold, axis=2)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, img.width, img.height)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def split_and_crop_row(path: str):
    """
    先按 1/4 宽度划分素材，再在每个四分区内检测真正的小图边界并紧裁。
    这样能修复第2/3/4列因为 matplotlib wspace 导致的横向漂移。
    """
    src = Image.open(path).convert("RGB")
    edges = np.linspace(0, src.width, 5).round().astype(int)

    panels = []
    bboxes = []
    for i in range(4):
        q = src.crop((edges[i], 0, edges[i + 1], src.height))
        bbox = content_bbox(q)
        crop = q.crop(bbox)

        # 保持比例缩放，再中心裁成正方形（通常原素材本身就是约500×500）
        w, h = crop.size
        scale = PANEL_SIZE / min(w, h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        crop = crop.resize((nw, nh), Image.Resampling.LANCZOS)

        left = max(0, (nw - PANEL_SIZE) // 2)
        top = max(0, (nh - PANEL_SIZE) // 2)
        crop = crop.crop((left, top, left + PANEL_SIZE, top + PANEL_SIZE))

        panels.append(crop)
        bboxes.append(bbox)

    return panels, bboxes


def draw_centered_text(draw, xy_center, text, font, fill):
    box = draw.textbbox((0, 0), text, font=font)
    tw = box[2] - box[0]
    th = box[3] - box[1]
    x = xy_center[0] - tw / 2 - box[0]
    y = xy_center[1] - th / 2 - box[1]
    draw.text((x, y), text, font=font, fill=fill)


def add_panel_label(canvas_rgba, panel_x, panel_y, label, draw, font):
    """
    方框边界精确相对 panel 的左上角定位。
    这比 ax.text(..., bbox=...) 更可靠，因为 bbox 的 padding 会影响真实边界。
    """
    x0 = panel_x + LABEL_MARGIN_PX
    y0 = panel_y + LABEL_MARGIN_PX
    x1 = x0 + LABEL_BOX_PX
    y1 = y0 + LABEL_BOX_PX

    # 半透明灰框
    draw.rectangle([x0, y0, x1, y1], fill=(80, 80, 80, LABEL_ALPHA))

    # 图号文字严格居中
    draw_centered_text(
        draw,
        ((x0 + x1) / 2, (y0 + y1) / 2),
        f"({label})",
        font,
        (255, 255, 255, 255),
    )


def draw_rotated_row_label(canvas_rgba, text, center_y, font):
    # 先在透明临时图层画水平文字，再旋转90°
    dummy = Image.new("RGBA", (700, 100), (255, 255, 255, 0))
    d = ImageDraw.Draw(dummy)
    bbox = d.multiline_textbbox((0, 0), text, font=font, spacing=3, align="center")
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    tmp = Image.new("RGBA", (int(round(tw + 20)), int(round(th + 20))), (255, 255, 255, 0))
    td = ImageDraw.Draw(tmp)
    td.multiline_text((10 - bbox[0], 10 - bbox[1]), text, font=font,
                      fill=(0, 0, 0, 255), spacing=3, align="center")

    rot = tmp.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    x = 12
    y = int(round(center_y - rot.height / 2))
    canvas_rgba.alpha_composite(rot, (x, y))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", default="gradcam_igneous_4_1_correct.png")
    parser.add_argument("--bottom", default="gradcam_metamorphic_2_2_wrong.png")
    parser.add_argument("--out", default="fig10_gradcam_fixed.png")
    args = parser.parse_args()

    top_panels, top_bboxes = split_and_crop_row(args.top)
    bottom_panels, bottom_bboxes = split_and_crop_row(args.bottom)

    total_panels_w = 4 * PANEL_SIZE + 3 * COL_GAP
    needed_w = LEFT_MARGIN + total_panels_w + RIGHT_MARGIN
    if needed_w > CANVAS_W:
        raise ValueError(
            f"当前布局需要宽度 {needed_w}px，但 CANVAS_W={CANVAS_W}px。"
            "请减小 PANEL_SIZE/COL_GAP/边距，或增大 CANVAS_W。"
        )

    # 把多余宽度均匀留给右侧，保持左边纵向标签位置稳定
    start_x = LEFT_MARGIN + (CANVAS_W - needed_w) // 2

    top_y = TOP_TITLE_H
    bottom_y = top_y + PANEL_SIZE + ROW_GAP
    canvas_h = bottom_y + PANEL_SIZE + BOTTOM_MARGIN

    canvas = Image.new("RGBA", (CANVAS_W, canvas_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas, "RGBA")

    title_font = find_font(TITLE_FONT_SIZE, bold=False)
    row_font = find_font(ROW_FONT_SIZE, bold=False)
    label_font = find_font(LABEL_FONT_SIZE, bold=True)

    # 列标题
    for col, title in enumerate(TITLES):
        x = start_x + col * (PANEL_SIZE + COL_GAP)
        cx = x + PANEL_SIZE / 2
        box = draw.textbbox((0, 0), title, font=title_font)
        tw = box[2] - box[0]
        draw.text((cx - tw / 2 - box[0], 8 - box[1]),
                  title, font=title_font, fill=(0, 0, 0, 255))

    # 贴8张图 + 图号
    letters = "abcdefgh"
    for row, panels in enumerate([top_panels, bottom_panels]):
        y = top_y if row == 0 else bottom_y
        for col, panel in enumerate(panels):
            x = start_x + col * (PANEL_SIZE + COL_GAP)
            canvas.paste(panel.convert("RGBA"), (x, y))
            add_panel_label(canvas, x, y, letters[row * 4 + col], draw, label_font)

    # 左侧纵向行标签
    draw_rotated_row_label(canvas, TOP_ROW_TEXT, top_y + PANEL_SIZE / 2, row_font)
    draw_rotated_row_label(canvas, BOTTOM_ROW_TEXT, bottom_y + PANEL_SIZE / 2, row_font)

    # 输出为RGB PNG
    canvas.convert("RGB").save(args.out, dpi=(300, 300))

    print(f"saved: {args.out}")
    print("Detected source-panel bboxes (within each 1/4 strip):")
    print(" top   :", top_bboxes)
    print(" bottom:", bottom_bboxes)
    print(f"Panel-label box margin = {LABEL_MARGIN_PX}px for all 8 panels.")


if __name__ == "__main__":
    main()
