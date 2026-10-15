r"""
generate_fig5_confusion_matrix.py

生成 Figure 5：exp01 与 exp02 的 slide-level 混淆矩阵对比图。

数据来源：矩阵整数计数与论文 Table 1 的 overall accuracy 和 per-class recall 一致：

    exp01: [[12,1,0],[4,7,1],[1,0,7]]，OA=78.79%(26/33)
        Rec-Ign=0.923, Rec-Met=0.583, Rec-Sed=0.875
    exp02: [[11,1,1],[2,9,1],[0,0,8]]，OA=84.85%(28/33)
        Rec-Ign=0.846, Rec-Met=0.750, Rec-Sed=1.000

矩阵行=真实类别，列=预测类别，顺序均为 [Igneous, Metamorphic, Sedimentary]。

字体：Liberation Serif（Windows本地有Times New Roman可以把下面这行改成
'Times New Roman'）。

依赖：matplotlib, numpy

用法：
    python generate_fig5_confusion_matrix.py
输出：
    fig5_confmat.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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

CLASSES = ["Igneous", "Metam.", "Sedim."]

CM_EXP01 = np.array([[12, 1, 0], [4, 7, 1], [1, 0, 7]])
CM_EXP02 = np.array([[11, 1, 1], [2, 9, 1], [0, 0, 8]])
RECALLS_EXP01 = [0.923, 0.583, 0.875]
RECALLS_EXP02 = [0.846, 0.750, 1.000]


def plot_cm(ax, cm, recalls, title, oa_text):
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=12)
    n = len(CLASSES)
    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            color = "white" if val > 7 else "black"
            ax.text(j, i, str(val), ha="center", va="center", fontsize=13, color=color)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(CLASSES, fontsize=9.5)
    ax.set_yticklabels(CLASSES, fontsize=9.5)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(f"{title}\n{oa_text}", fontsize=10, linespacing=1.6)

    # 高亮变质岩行，与正文重点分析保持一致
    rect_y = 0.5
    ax.plot([-0.5, 2.5, 2.5, -0.5, -0.5], [rect_y, rect_y, rect_y + 1, rect_y + 1, rect_y],
            color="#B23A48", linewidth=1.7, linestyle="--")

    for i, r in enumerate(recalls):
        ax.annotate(f"Rec={r:.2f}", xy=(1.02, (2 - i) / 2), xycoords="axes fraction",
                    ha="left", va="center", fontsize=8.3, color="#444444")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6))
    plot_cm(axes[0], CM_EXP01, RECALLS_EXP01, "exp01 (PPL baseline)", "OA = 78.79% (26/33)")
    plot_cm(axes[1], CM_EXP02, RECALLS_EXP02, "exp02 (PPL+XPL early fusion)", "OA = 84.85% (28/33)")
    axes[0].text(-0.16, 1.14, "(a)", transform=axes[0].transAxes,
                 fontsize=13, fontweight="bold", va="top", ha="left")
    axes[1].text(-0.16, 1.14, "(b)", transform=axes[1].transAxes,
                 fontsize=13, fontweight="bold", va="top", ha="left")

    plt.subplots_adjust(wspace=0.65, top=0.80, bottom=0.13, left=0.07, right=0.90)
    plt.savefig("fig5_confmat.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved fig5_confmat.png")


if __name__ == "__main__":
    main()
