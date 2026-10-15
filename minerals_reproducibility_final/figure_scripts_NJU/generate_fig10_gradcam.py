r"""
generate_fig10_gradcam.py

生成 Figure 10：exp02 的 Grad-CAM 归因图，两个代表性样例（一个正确分类、一个误判）
上下两排对比。

数据来源：gradcam_viz.py 生成的原始 Grad-CAM 输出图（热力图和显微照片本身
都是真实的Grad-CAM计算结果，不是本脚本生成的——本脚本只是把这两张已经存在
的PNG重新拼版、加上英文标题和(a)-(h)面板字母）：

    上排：gradcam_igneous_4_1_correct.png       （正确分类，置信度0.95）
    下排：gradcam_metamorphic_2_2_wrong.png     （误判，置信度0.77错判为igneous）

原图约定：gradcam_viz.py 输出的是"四个等宽面板横排（PPL原图 / PPL+CAM /
XPL原图 / XPL+CAM），不含任何嵌入式标题栏"的干净图。本脚本按内容实际边界
自动裁掉四周可能存在的白边，再把剩余宽度严格四等分——不再依赖任何写死的
像素坐标，只要上游 gradcam_viz.py 保持"四等宽面板+无标题栏"这个约定，
不管图片实际分辨率是多少，这里都能正确处理。

字体：Liberation Serif（Windows本地有Times New Roman可以把下面这行改成
'Times New Roman'）。

依赖：matplotlib, numpy, Pillow

用法：
    python generate_fig10_gradcam.py \
        --correct path/to/gradcam_igneous_4_1_correct.png \
        --wrong path/to/gradcam_metamorphic_2_2_wrong.png
输出：
    fig10_gradcam.png
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

COL_TITLES = ["PPL image", "PPL + Grad-CAM", "XPL image", "XPL + Grad-CAM"]


def load_panels(path: str):
    """把1x4素材图切成4个面板，每个面板都单独检测自己的真实内容边界——
    不是"整张图先统一裁边、再机械四等分"这种旧做法。

    根因（2026-08-07确认）：gradcam_viz.py用matplotlib
    subplots_adjust(wspace=...)横排4张子图，每一列内部实际内容离该列
    区块左边界的留白宽度并不完全相等，会随列数逐渐增大或减小（实测样例：
    约10px/8px/5px/2px递减）。如果先把整张大图统一trim一次外围白边、
    再简单按1/4宽度切分，这部分逐列递减的"列内部留白差异"会原样保留在
    切出来的每个面板里——图号标签是按"面板/axes边界"定位的，所以标签
    相对面板边界的坐标其实完全没有偏差，但相对"照片肉眼可见的真实边缘"
    就会产生越到右边越明显的视觉偏移，这正是(c)(d)(g)(h)看起来"位置
    不对"的真正原因，不是标签坐标计算错了。

    修复方式：先把整张图按1/4宽度切成4块，再对每一块单独做内容边界检测
    （去掉这一块自己的内部留白），而不是复用整张图统一裁过的边界。这样
    无论matplotlib每列留白宽度差多少，切出来的每个面板都从各自真实内容
    的左上角开始，8个面板的图号标签就能始终紧贴各自照片的实际边缘。"""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    quarter_w = w // 4

    panels = []
    for i in range(4):
        x0 = i * quarter_w
        x1 = w if i == 3 else (i + 1) * quarter_w
        quarter = im.crop((x0, 0, x1, h))

        q_arr = np.array(quarter.convert("L"))
        rows = np.where((q_arr < 250).any(axis=1))[0]
        cols = np.where((q_arr < 250).any(axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            raise ValueError(f"{path}：第{i+1}个区块是空白的，检测不到任何内容。")
        top, bottom = int(rows.min()), int(rows.max()) + 1
        left, right = int(cols.min()), int(cols.max()) + 1
        panels.append(np.array(quarter.crop((left, top, right, bottom))))
    return panels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--correct", required=True,
                         help="正确分类样例的原始Grad-CAM图路径（如 gradcam_igneous_4_1_correct.png）")
    parser.add_argument("--wrong", required=True,
                         help="误判样例的原始Grad-CAM图路径（如 gradcam_metamorphic_2_2_wrong.png）")
    parser.add_argument("--correct-label", default="igneous_4_1\ncorrect (p = 0.95)")
    parser.add_argument("--wrong-label", default="metamorphic_2_2\nmisclassified (p = 0.77)")
    parser.add_argument("--out", default="fig10_gradcam.png")
    args = parser.parse_args()

    correct_panels = load_panels(args.correct)
    wrong_panels = load_panels(args.wrong)

    fig, axes = plt.subplots(2, 4, figsize=(11.5, 5.9))

    # 每个小图独立标 (a)-(d)（第一行）、(e)-(h)（第二行），而不是整行只标一个字母——
    # 每张显微照片/热力图都是独立的子图，按标准多面板图规范逐个标注。
    #
    # 标签框样式：无描边（ec="none"）、半透明黑底（alpha=0.55）、pad=0.15紧凑
    # 贴合文字——保持最终版式。中间测试过白色粗描边、浅灰
    # 细描边等方案来解决"框在深色/彩色背景上边界不易辨认"的问题，但都被
    # 最终采用无描边方案。框的实际坐标位置本身从未有过偏差
    # （已用像素级测量反复验证过），如果后续还有类似反馈，请先确认是否
    # 真的是坐标问题，而不是本样式在深色背景下天然存在的可辨识度取舍。
    row0_letters = ["(a)", "(b)", "(c)", "(d)"]
    row1_letters = ["(e)", "(f)", "(g)", "(h)"]

    for j in range(4):
        axes[0, j].imshow(correct_panels[j])
        axes[0, j].axis("off")
        axes[0, j].set_title(COL_TITLES[j], fontsize=10.5)
        axes[0, j].text(0.02, 0.98, row0_letters[j], transform=axes[0, j].transAxes,
                         fontsize=11, fontweight="bold", color="white", va="top", ha="left",
                         bbox=dict(boxstyle="square,pad=0.15", fc="black", ec="none", alpha=0.55))

    for j in range(4):
        axes[1, j].imshow(wrong_panels[j])
        axes[1, j].axis("off")
        axes[1, j].text(0.02, 0.98, row1_letters[j], transform=axes[1, j].transAxes,
                         fontsize=11, fontweight="bold", color="white", va="top", ha="left",
                         bbox=dict(boxstyle="square,pad=0.15", fc="black", ec="none", alpha=0.55))

    axes[0, 0].text(-0.12, 0.5, args.correct_label, transform=axes[0, 0].transAxes,
                     rotation=90, va="center", ha="center", fontsize=9.5, linespacing=1.4)
    axes[1, 0].text(-0.12, 0.5, args.wrong_label, transform=axes[1, 0].transAxes,
                     rotation=90, va="center", ha="center", fontsize=9.5, linespacing=1.4)

    plt.tight_layout()
    plt.savefig(args.out, dpi=280, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
