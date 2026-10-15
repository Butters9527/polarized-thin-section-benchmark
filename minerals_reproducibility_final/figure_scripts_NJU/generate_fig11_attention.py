r"""
generate_fig11_attention.py

生成 Figure 11：exp03 全测试集(N=33)各帧平均注意力权重柱状图。

数据来源：experiments/attn_viz_exp03/aggregate_attention.csv
（由 scripts/attention_viz.py --exp exp03 生成，对33个held-out测试薄片的
frame-to-frame attention权重取平均得到，8帧分别对应PPL、XPL@0°~90°）。

csv 格式（3列）：
    frame, mean_attention_received, std_attention_received
    PPL, 0.035, 0.037
    XPL@0°, 0.182, 0.190
    ...

字体：Liberation Serif（Windows本地有Times New Roman可以把下面这行改成
'Times New Roman'）。

依赖：matplotlib, pandas

用法：
    python generate_fig11_attention.py --csv path/to/aggregate_attention.csv
输出：
    fig11_attention.png
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True,
                         help="experiments/attn_viz_exp03/aggregate_attention.csv 的路径")
    parser.add_argument("--out", default="fig11_attention.png")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    frame_labels = df["frame"].tolist()
    means = df["mean_attention_received"].values
    stds = df["std_attention_received"].values

    fig, ax = plt.subplots(figsize=(7.3, 4.2))
    # 第一根柱子（PPL）用灰色区分，其余XPL帧用蓝色
    colors = ["#8c8c8c"] + ["#1f77b4"] * (len(frame_labels) - 1)
    ax.bar(range(len(frame_labels)), means, yerr=stds, capsize=3, color=colors,
           error_kw={"linewidth": 1.1})
    ax.set_xticks(range(len(frame_labels)))
    ax.set_xticklabels(frame_labels, rotation=40, ha="right", fontsize=10)
    ax.set_ylabel("Mean attention received\n(averaged over heads, query positions, N = 33)",
                  fontsize=10.2)
    ax.axhline(1 / 8, color="gray", linestyle="--", linewidth=1, label="Uniform baseline (1/8)")
    ax.set_title("Mean attention weight received by each frame (exp03)", fontsize=10.8)
    ax.tick_params(axis="y", labelsize=9.5)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
