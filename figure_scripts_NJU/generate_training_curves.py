r"""
generate_training_curves.py

生成 Figure 4、6、7、8、9——五个训练曲线图，全部用同一个模板函数画，风格统一。

数据来源（每个函数调用时需要对应的 train_log.json 文件路径）：
    Figure 4 -> experiments/exp01_baseline_relog/train_log.json
    Figure 6 -> experiments/exp02_dual_stream_relog/train_log.json
    Figure 7 -> experiments/exp03_sequence_relog/train_log.json
    Figure 8 -> experiments/exp03b_frozen_relog/train_log.json
    Figure 9 -> experiments/exp04_rpe/train_log.json

train_log.json 的格式是 {"history": [{"epoch":1,"train_loss":...,"train_acc":...,
"val_loss":...,"val_acc":...}, ...]}（exp04 的记录里没有 val_loss 字段，
脚本会自动处理，只画 Training Loss 不画 Val Loss）。

这些 json 文件是用 train_sequence_relog.py / train_sequence_b_relog.py（exp03/03b）
或对应的 exp01/exp02/exp04 训练脚本重训一遍后生成的，重训只是为了拿到逐epoch记录
画曲线用，跟论文里正式报告的 Table 1 数字是两回事——重训的 best_val_acc 会因为
GPU 非确定性跟原始 checkpoint 略有出入，这是预期内的，论文数字以原始 checkpoint 为准。

字体：Liberation Serif（Windows本地有Times New Roman可以把下面这行改成
'Times New Roman'）。

依赖：matplotlib（pip install matplotlib）

用法：
    python generate_training_curves.py \
        --exp01 path/to/exp01_baseline_relog/train_log.json \
        --exp02 path/to/exp02_dual_stream_relog/train_log.json \
        --exp03 path/to/exp03_sequence_relog/train_log.json \
        --exp03b path/to/exp03b_frozen_relog/train_log.json \
        --exp04 path/to/exp04_rpe/train_log.json

任何一个参数不给，就跳过对应的那张图，不会报错。
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def plot_curves(log_path: str, title: str, out_path: str) -> None:
    """通用训练曲线绘图函数：左图 Loss，右图 Accuracy。

    自动兼容两种 train_log.json 格式：
      - exp01/exp02/exp04 relog 脚本存的格式：{"config": {...}, "history": [...], "best_val_acc": ...}
      - exp03/exp03b_relog.py（train_sequence_relog.py）存的格式：直接是 [...] 列表
    """
    with open(log_path) as f:
        data = json.load(f)
    history = data["history"] if isinstance(data, dict) else data
    has_val_loss = "val_loss" in history[0]

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    # exp01/exp02/exp04 的 relog 脚本存的 acc 已经是百分数（如 93.75）；
    # exp03/exp03b_relog.py 存的 acc 是 0-1 之间的小数（如 0.78）。
    # 统一转换成百分数，否则同一批图里坐标轴量纲会不一致。
    if max(train_acc) <= 1.0:
        train_acc = [a * 100 for a in train_acc]
        val_acc = [a * 100 for a in val_acc]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    # -- 左图：Loss --
    axes[0].plot(epochs, train_loss, marker="o", ms=4.5, label="Train Loss",
                 color="#1f77b4", linewidth=1.4)
    if has_val_loss:
        val_loss = [h["val_loss"] for h in history]
        axes[0].plot(epochs, val_loss, marker="o", ms=4.5, label="Val Loss",
                     color="#ff7f0e", linewidth=1.4)
        loss_title = "Loss"
    else:
        loss_title = "Training Loss"
    axes[0].set_xlabel("Epoch", fontsize=10.5)
    axes[0].set_ylabel("Loss", fontsize=10.5)
    axes[0].set_title(loss_title, fontsize=10.5)
    axes[0].tick_params(labelsize=9.5)
    axes[0].legend(fontsize=9, frameon=True)
    axes[0].set_xticks(epochs)
    axes[0].text(-0.05, 1.06, "(a)", transform=axes[0].transAxes,
                 fontsize=13, fontweight="bold", va="bottom", ha="left")

    # -- 右图：Accuracy --
    axes[1].plot(epochs, train_acc, marker="o", ms=4.5, label="Train Acc (%)",
                 color="#1f77b4", linewidth=1.4)
    axes[1].plot(epochs, val_acc, marker="o", ms=4.5, label="Val Acc (%)",
                 color="#ff7f0e", linewidth=1.4)
    axes[1].set_xlabel("Epoch", fontsize=10.5)
    axes[1].set_ylabel("Accuracy (%)", fontsize=10.5)
    axes[1].set_title("Train/Val Accuracy", fontsize=10.5)
    axes[1].tick_params(labelsize=9.5)
    axes[1].legend(fontsize=9, frameon=True)
    axes[1].set_xticks(epochs)
    axes[1].text(-0.05, 1.06, "(b)", transform=axes[1].transAxes,
                 fontsize=13, fontweight="bold", va="bottom", ha="left")

    fig.suptitle(title, fontsize=11.5, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.subplots_adjust(top=0.80)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    best_idx = val_acc.index(max(val_acc))
    print(f"saved {out_path}  (best val_acc={val_acc[best_idx]:.2f}% at epoch {epochs[best_idx]}, "
          f"{len(history)} epochs total)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp01", default=None, help="Figure 4 数据源")
    parser.add_argument("--exp02", default=None, help="Figure 6 数据源")
    parser.add_argument("--exp03", default=None, help="Figure 7 数据源")
    parser.add_argument("--exp03b", default=None, help="Figure 8 数据源")
    parser.add_argument("--exp04", default=None, help="Figure 9 数据源")
    args = parser.parse_args()

    jobs = [
        (args.exp01, "exp01 (PPL baseline)", "fig4_exp01curves.png"),
        (args.exp02, "exp02 (PPL+XPL)", "fig6_exp02curves.png"),
        (args.exp03, "exp03 (full fine-tune)", "fig7_exp03curves.png"),
        (args.exp03b, "exp03b (frozen+high dropout)", "fig8_exp03bcurves.png"),
        (args.exp04, "exp04 (RPE)", "fig9_exp04curves.png"),
    ]

    for log_path, title, out_path in jobs:
        if log_path is None:
            print(f"跳过 {out_path}（未提供对应的 --xxx 参数）")
            continue
        plot_curves(log_path, title, out_path)


if __name__ == "__main__":
    main()
