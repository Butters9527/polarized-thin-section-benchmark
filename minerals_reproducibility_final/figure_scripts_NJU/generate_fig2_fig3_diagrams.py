r"""
generate_fig2_fig3_diagrams.py

生成 Figure 2（数据流程图）和 Figure 3（三模型架构图）。

数据来源：无需外部数据文件——这两张图的内容是方法本身的结构描述
（数据划分方式、模型架构层次），数字/文字直接来自论文4.1-4.4节的方法定义，
不依赖任何实验运行结果。

字体：Liberation Serif（Linux下与Times New Roman度量兼容的开源替代字体）。
如果在Windows本地跑，安装了Times New Roman的话，可以把下面
plt.rcParams['font.family'] 那一行改成 'Times New Roman'，效果更精确一致；
不改也没关系，Liberation Serif视觉上跟Times New Roman几乎无法区分。

依赖：matplotlib（pip install matplotlib）

用法：
    python generate_fig2_fig3_diagrams.py
输出：
    fig2_pipeline.png
    fig3_architecture.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

# 字体设置。
FONT_FAMILY = "Times New Roman"
plt.rcParams["font.family"] = FONT_FAMILY
plt.rcParams["mathtext.fontset"] = "stix"

# 字体可用性检查：matplotlib找不到指定字体时会*静默*降级成默认字体（通常是
# DejaVu Sans），只在命令行刷一堆findfont警告、图还是能正常生成。这里采用显式字体检查，避免在字体缺失时静默降级。
_available_fonts = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
if FONT_FAMILY not in _available_fonts:
    raise SystemExit(
        f"错误：系统里没有安装字体 '{FONT_FAMILY}'，无法继续生成图片。\n"
        f"请确认该字体已安装（Windows上'Times New Roman'一般随系统自带），"
        f"或者把脚本里 FONT_FAMILY 改成一个系统里确实存在的字体名。\n"
        f"（不要让脚本在找不到字体时静默降级——那样生成的图字体是错的，"
        f"看起来正常但跟其他版本对不上。）"
    )


# ============================================================================
# 通用绘图辅助函数
# ============================================================================
def box(ax, x, y, w, h, text, fc="#EAF2FB", ec="#2E5C8A", fontsize=10.5, weight="normal", linespacing=1.4):
    b = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.1, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, weight=weight, linespacing=linespacing)


def arrow(ax, x1, y1, x2, y2, color="#333333"):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
        linewidth=1.1, color=color,
    )
    ax.add_patch(a)


# ============================================================================
# Figure 1：数据流程图
# ============================================================================
def make_fig1():
    fig, ax = plt.subplots(figsize=(9.2, 3.3))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 3.4)
    ax.axis("off")

    box(ax, 0.15, 1.3, 1.75, 1.5,
        "8-frame\nacquisition per\nsection\n(1 PPL @0\u00b0\n+ 7 XPL\n@0\u00b0-90\u00b0,15\u00b0)",
        fc="#EAF2FB")
    box(ax, 2.25, 1.3, 1.75, 1.5,
        "Slide-level\nstratified split\n(seed = 42)\ntrain 256 /\nval 32 / test 33\nsections",
        fc="#FBF3E3")
    box(ax, 4.35, 2.55, 2.05, 0.72, "exp01: single PPL\nframe (3ch)", fc="#EAF7EC", fontsize=9.5)
    box(ax, 4.35, 1.64, 2.05, 0.72, "exp02: PPL+XPL@0\u00b0\nconcat (6ch)", fc="#EAF7EC", fontsize=9.5)
    box(ax, 4.35, 0.73, 2.05, 0.72, "exp03/03b/04: full\n8-frame sequence", fc="#EAF7EC", fontsize=9.5)
    box(ax, 6.75, 2.55, 1.55, 0.72, "ResNet-50\n(3ch input)", fc="#F3E9F7", fontsize=9.5)
    box(ax, 6.75, 1.64, 1.55, 0.72, "ResNet-50\n(6ch, Eq. 1 init)", fc="#F3E9F7", fontsize=9.5)
    box(ax, 6.75, 0.73, 1.55, 0.72, "Shared ResNet-50\n+ attention (\u00b1PE)", fc="#F3E9F7", fontsize=9.5)
    box(ax, 8.65, 1.3, 1.75, 1.5,
        "Slide-level\nprediction\n(1 of 3\nlithological\nclasses)\nheld-out test /\n5-fold CV",
        fc="#FDEAEA")

    arrow(ax, 1.9, 2.05, 2.25, 2.05)
    arrow(ax, 4.0, 2.35, 4.35, 2.91)
    arrow(ax, 4.0, 2.05, 4.35, 2.0)
    arrow(ax, 4.0, 1.75, 4.35, 1.09)
    arrow(ax, 6.4, 2.91, 6.75, 2.91)
    arrow(ax, 6.4, 2.0, 6.75, 2.0)
    arrow(ax, 6.4, 1.09, 6.75, 1.09)
    arrow(ax, 8.3, 2.91, 8.65, 2.35)
    arrow(ax, 8.3, 2.0, 8.65, 2.05)
    arrow(ax, 8.3, 1.09, 8.65, 1.75)

    plt.tight_layout()
    tmp_path = "fig2_pipeline_raw.png"
    plt.savefig(tmp_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # bbox_inches='tight' 在这类用 ax.axis('off') + 手工 patch 画的图上，
    # 有时算出的裁剪框会比实际内容大一圈（尤其是底部留白明显）。
    # 用 PIL 直接按真实像素内容边界裁剪，四周各留 20px 的小边距。
    im = Image.open(tmp_path).convert("RGB")
    arr = np.array(im.convert("L"))
    rows = np.where((arr < 250).any(axis=1))[0]
    cols = np.where((arr < 250).any(axis=0))[0]
    pad = 20
    top = max(0, rows.min() - pad)
    bottom = min(arr.shape[0], rows.max() + pad)
    left = max(0, cols.min() - pad)
    right = min(arr.shape[1], cols.max() + pad)
    im.crop((left, top, right, bottom)).save("fig2_pipeline.png")
    print("saved fig2_pipeline.png")


# ============================================================================
# Figure 2：三模型架构图
# ============================================================================
def render_column(ax, x, top_y, col_w, boxes, arrow_gap=0.5):
    """按固定箭头长度(arrow_gap)自上而下渲染一列框图，保证每个箭头长度完全一致，
    不会出现某些箭头被下一个框遮住、或者箭头后面空白忽大忽小的问题。

    boxes: [(text, n_lines, fc), ...]，n_lines 决定框高：
        1行->0.55，2行->0.85，3行->1.6（3行文字在9pt字号+1.4倍行距下
        实测至少需要约1.57个轴单位，1.6留了安全余量，不会再顶穿边框）。
    返回：最后一个框的底部 y 坐标（供调用方知道这一列内容画到哪里结束）。
    """
    height_by_lines = {1: 0.72, 2: 1.08, 3: 1.85}
    y = top_y
    n = len(boxes)
    for i, (text, n_lines, fc) in enumerate(boxes):
        h = height_by_lines[n_lines]
        box_bottom = y - h
        box(ax, x, box_bottom, col_w, h, text, fc=fc)
        if i < n - 1:
            arrow(ax, x + col_w / 2, box_bottom, x + col_w / 2, box_bottom - arrow_gap)
        y = box_bottom - arrow_gap
    return y + arrow_gap  # 最后一个框的底部（不含多余的箭头间隙）


def make_fig2():
    fig, ax = plt.subplots(figsize=(9.6, 6.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 17.2)
    ax.axis("off")

    col_w, gap = 2.7, 0.35
    x0, x1, x2 = 0.3, 0.3 + col_w + gap, 0.3 + 2 * (col_w + gap)
    header_y = 15.15   # 标题文字块的顶部（用 va="top" 精确锚定，不再靠猜的间距）
    top_y = 13.85       # 第一排框的顶部，紧跟在标题下方，只留一点点呼吸空间

    header0 = ax.text(x0 + col_w / 2, header_y, "exp01: single-frame\nPPL baseline",
            ha="center", va="top", fontsize=10.5, weight="bold", linespacing=1.35)
    header1 = ax.text(x1 + col_w / 2, header_y, "exp02: dual-modal\nearly fusion",
            ha="center", va="top", fontsize=10.5, weight="bold", linespacing=1.35)
    header2 = ax.text(x2 + col_w / 2, header_y, "exp03/03b/04:\ntemporal attention",
            ha="center", va="top", fontsize=10.5, weight="bold", linespacing=1.35)

    # 每列左上角的 (a)/(b)/(c) 面板标签。
    # 重要：不能把标签x坐标写死（如 x0-0.05），因为标题文字是居中对齐的，
    # 不同字体（如Windows的Times New Roman 比 Linux的Liberation Serif更宽）
    # 渲染出来的实际宽度不同，标题左边缘的位置会跟着变，写死坐标在换字体后
    # 会导致标签跟标题文字撞在一起（实测过：DejaVu Serif等更宽字体下会重叠）。
    # 正确做法：先把标题画出来，用renderer实际测量出每个标题文字块的左边缘
    # 在data坐标系里的位置，再把标签用 ha="right" 锚定在"左边缘减去固定间距"处——
    # 这样无论换成什么字体，标签和标题之间的间距都保持恒定，不会撞字。
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_gap = 0.12  # 标签右边缘跟标题左边缘之间保留的固定间距（data坐标单位）

    def place_panel_label(header_text_obj, label_text):
        bbox = header_text_obj.get_window_extent(renderer=renderer)
        bbox_data = bbox.transformed(ax.transData.inverted())
        left_edge_x = bbox_data.x0
        ax.text(left_edge_x - label_gap, header_y, label_text,
                fontsize=13, weight="bold", ha="right", va="top")

    place_panel_label(header0, "(a)")
    place_panel_label(header1, "(b)")
    place_panel_label(header2, "(c)")

    col0_boxes = [
        ("PPL frame\n(3, 224, 224)", 2, "#EAF2FB"),
        ("ResNet-50 backbone\n(ImageNet pretrained)", 2, "#EAF2FB"),
        ("GAP \u2192 2048-d feature", 1, "#EAF2FB"),
        ("Dropout (p = 0.3)", 1, "#EAF2FB"),
        ("Linear(2048 \u2192 3)", 1, "#FDEAEA"),
        ("Slide-level class\n(igneous / metam. / sed.)", 2, "#EAF7EC"),
    ]
    col1_boxes = [
        ("PPL + XPL@0\u00b0\nconcat \u2192 (6, 224, 224)", 2, "#EAF2FB"),
        ("Adapted Conv1 [64,6,7,7]\nW_new via Eq. (1)", 2, "#EAF2FB"),
        ("ResNet-50 backbone\n(remaining layers)", 2, "#EAF2FB"),
        ("GAP \u2192 Dropout(0.3)", 1, "#EAF2FB"),
        ("Linear(2048 \u2192 3)", 1, "#FDEAEA"),
        ("Slide-level class\n(igneous / metam. / sed.)", 2, "#EAF7EC"),
    ]
    col2_boxes = [
        ("8-frame sequence\n(1 PPL + 7 XPL\n@0\u201390\u00b0, 15\u00b0 step)", 3, "#EAF2FB"),
        ("Shared ResNet-50\n(per-frame, weight-tied)\n\u2192 [B, 8, 2048]", 3, "#EAF2FB"),
        ("exp04 only: + rotational\npositional encoding\n(sin/cos of \u03b8)", 3, "#FBF3E3"),
        ("Multi-head self-attention\n(8 heads, embed = 2048)\nacross frame axis", 3, "#EAF2FB"),
        ("Mean-pool \u2192 LayerNorm\n\u2192 Dropout", 2, "#EAF2FB"),
        ("Linear(2048 \u2192 3)", 1, "#FDEAEA"),
        ("Slide-level class", 1, "#EAF7EC"),
    ]

    render_column(ax, x0, top_y, col_w, col0_boxes)
    render_column(ax, x1, top_y, col_w, col1_boxes)
    render_column(ax, x2, top_y, col_w, col2_boxes)

    tmp_path = "fig3_architecture_raw.png"
    plt.savefig(tmp_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 跟 Figure 1 一样，用实际像素内容边界裁剪，而不是硬编码比例
    im = Image.open(tmp_path).convert("RGB")
    arr = np.array(im.convert("L"))
    rows = np.where((arr < 250).any(axis=1))[0]
    cols = np.where((arr < 250).any(axis=0))[0]
    pad = 20
    top = max(0, rows.min() - pad)
    bottom = min(arr.shape[0], rows.max() + pad)
    left = max(0, cols.min() - pad)
    right = min(arr.shape[1], cols.max() + pad)
    im.crop((left, top, right, bottom)).save("fig3_architecture.png")
    print("saved fig3_architecture.png")

if __name__ == "__main__":
    make_fig1()
    make_fig2()
