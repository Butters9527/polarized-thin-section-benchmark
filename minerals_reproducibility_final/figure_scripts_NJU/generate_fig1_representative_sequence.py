r"""
generate_fig1_representative_sequence.py

生成 Figure 1：代表性消光序列图（NJU数据集，二辉橄榄岩薄片，1张PPL+7张XPL旋转角度）。

数据来源：NJU数据集火成岩单元的一块二辉橄榄岩薄片，
按论文3.1节描述的标准8帧采集协议拍摄：
    图1 = PPL @0°
    图2-8 = XPL @0°, 15°, 30°, 45°, 60°, 75°, 90°（按15°递增）

该图直接展示真实原始照片本身，用于说明
一张真实岩石照片"的意见（Reviewer 3原话："no representative thin-section images"）。
跟Figure 10（Grad-CAM）的区别：Grad-CAM图服务于可解释性分析、只有2个样本、叠加了
热力图；这张图服务于让读者一开始就直观理解"PPL/XPL旋转序列长什么样"，不带任何
模型输出叠加，是干净的原始数据展示，放在3.1节采集协议描述刚讲完的位置。

字体：Liberation Serif（Windows本地有Times New Roman可以把下面这行改成
'Times New Roman'）。

依赖：matplotlib, numpy, Pillow

用法：
    python generate_fig1_representative_sequence.py \
        --dir path/to/photos/ \
        --prefix "火1二辉橄榄岩2-"

    默认假设8张图片命名为 {prefix}1.jpg 到 {prefix}8.jpg，其中第1张是PPL，
    第2-8张依次是XPL@0°,15°,30°,45°,60°,75°,90°。如果你的照片顺序/命名不同，
    用 --files 参数显式指定8个文件路径，按PPL, XPL@0,15,30,45,60,75,90的顺序传入。

输出：
    fig1_representative_sequence.png
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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

LABELS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
TITLES = [
    "PPL",
    "XPL @0\u00b0", "XPL @15\u00b0", "XPL @30\u00b0",
    "XPL @45\u00b0", "XPL @60\u00b0", "XPL @75\u00b0", "XPL @90\u00b0",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=None, help="8张照片所在目录")
    parser.add_argument("--prefix", default="火1二辉橄榄岩2-",
                         help="文件名前缀，默认拼成 {prefix}1.jpg ... {prefix}8.jpg")
    parser.add_argument("--files", nargs=8, default=None,
                         help="显式指定8个文件路径（按PPL, XPL@0..90顺序），"
                              "跟 --dir/--prefix 二选一")
    parser.add_argument("--out", default="fig1_representative_sequence.png")
    args = parser.parse_args()

    if args.files:
        paths = args.files
    else:
        if args.dir is None:
            raise SystemExit("必须提供 --dir 或 --files 其中之一")
        import os
        paths = [os.path.join(args.dir, f"{args.prefix}{i}.jpg") for i in range(1, 9)]

    imgs = [np.array(Image.open(p)) for p in paths]

    fig, axes = plt.subplots(2, 4, figsize=(11.5, 6.1))

    for idx in range(8):
        row, col = divmod(idx, 4)
        ax = axes[row, col]
        ax.imshow(imgs[idx])
        ax.axis("off")
        ax.set_title(TITLES[idx], fontsize=10.5)
        ax.text(0.02, 0.98, LABELS[idx], transform=ax.transAxes,
                fontsize=11, fontweight="bold", color="white", va="top", ha="left",
                bbox=dict(boxstyle="square,pad=0.15", fc="black", ec="none", alpha=0.55))

    plt.tight_layout()
    plt.savefig(args.out, dpi=280, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
