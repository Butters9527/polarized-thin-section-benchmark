r"""
generate_training_curves.py   （2026-10-15 版）

生成 Figure 4、6、7、8、9 —— 五张训练曲线图，用同一个模板函数绘制，风格统一。

本版要点
--------
1. Figure 7 只使用 dropout=0.3 的重跑记录 exp03_sequence_relog_d03（与论文中原始 exp03
   checkpoint 的超参数一致）。旧的 exp03_sequence_relog 是 dropout=0.5 重跑的结果
   （7 个 epoch，最佳 val 78.12%），与论文图注不符，脚本会拒绝用它画图。
2. 不带参数运行（或双击同名 .bat）即可一次画完全部五张图。每个日志按以下顺序查找：
       ① 本脚本所在文件夹\train_log for Fig4_6_7_8_9\<实验名>\train_log.json
       ② 项目根目录\experiments\<实验名>\train_log.json
3. 图注核对：画图前，把每个日志的“总 epoch 数 / 最佳 epoch / 最佳验证准确率”与论文
   图注中的数值逐一比对，不一致就不出图并说明原因。这样即使拿错了日志文件，也不会
   再悄悄生成一张错误的图。确实需要画不同数据时（比如重新训练后），加 --no-check。
4. 输出图片保存在本脚本所在文件夹；读取 json 时显式使用 UTF-8。
   绘图部分（尺寸、字体、配色、线宽、标签位置）与之前版本完全相同。

论文图注对应的数值（best = 第一次达到最高 val_acc 的 epoch，与训练脚本保存
best checkpoint 的规则一致）：
    Figure 4  exp01   13 个 epoch，best epoch 8，val 93.75%
    Figure 6  exp02    9 个 epoch，best epoch 4，val 90.62%
    Figure 7  exp03   14 个 epoch，best epoch 9，val 90.62%   （dropout=0.3 重跑）
    Figure 8  exp03b  10 个 epoch，best epoch 5，val 84.38%
    Figure 9  exp04   12 个 epoch，best epoch 7，val 87.50%

两种 train_log.json 格式都支持：
  - exp01/exp02/exp04：{"config": {...}, "history": [...], ...}，acc 为百分数
  - exp03/exp03b 重跑：直接是 [...] 列表，acc 为 0–1 小数（脚本自动换算成百分数）
exp04 的记录里没有 val_loss，左图只画 Training Loss。

这些重跑只用于展示训练过程；论文 Table 1 的数字以原始 checkpoint 为准。

依赖：matplotlib

用法：
    双击 generate_training_curves.bat                      （推荐）
    python generate_training_curves.py                     （画全部五张）
    python generate_training_curves.py --only exp03        （只画 Figure 7）
    python generate_training_curves.py --exp03 某路径\train_log.json   （指定日志）
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)   # figure_scripts_NJU 的上一级 = 项目根目录
LOCAL_LOG_DIR = os.path.join(SCRIPT_DIR, "train_log for Fig4_6_7_8_9")

# (参数名, 日志文件夹名, 图标题, 输出文件名, 图号)
JOBS = [
    ("exp01",  "exp01_baseline_relog",     "exp01 (PPL baseline)",         "fig4_exp01curves.png",  "Figure 4"),
    ("exp02",  "exp02_dual_stream_relog",  "exp02 (PPL+XPL)",              "fig6_exp02curves.png",  "Figure 6"),
    ("exp03",  "exp03_sequence_relog_d03", "exp03 (full fine-tune)",       "fig7_exp03curves.png",  "Figure 7"),
    ("exp03b", "exp03b_frozen_relog",      "exp03b (frozen+high dropout)", "fig8_exp03bcurves.png", "Figure 8"),
    ("exp04",  "exp04_rpe",                "exp04 (RPE)",                  "fig9_exp04curves.png",  "Figure 9"),
]

# 论文图注对应的数值：(总 epoch 数, 最佳 epoch, 最佳 val_acc %)
EXPECTED = {
    "exp01":  (13, 8, 93.75),
    "exp02":  (9, 4, 90.625),
    "exp03":  (14, 9, 90.625),
    "exp03b": (10, 5, 84.375),
    "exp04":  (12, 7, 87.5),
}
# 旧的 dropout=0.5 exp03 重跑记录的特征，用于给出针对性的报错
OLD_EXP03_SIGNATURE = (7, 2, 78.125)

FONT_FAMILY = "Times New Roman"


def check_font(font_family: str) -> None:
    """matplotlib 找不到指定字体时会静默换成默认字体，图照样生成但字体是错的。
    这里显式检查，字体不可用就直接报错退出。"""
    available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    if font_family not in available:
        raise SystemExit(
            f"错误：系统里没有安装字体 '{font_family}'，无法继续生成图片。\n"
            f"Windows 上 'Times New Roman' 一般随系统自带；也可以用 --font 指定其他已安装的字体。"
        )
    plt.rcParams["font.family"] = font_family
    plt.rcParams["mathtext.fontset"] = "stix"


def find_log(folder_name: str):
    """按“本文件夹下的 train_log for Fig4_6_7_8_9 → 项目 experiments”顺序查找日志。"""
    candidates = [
        os.path.join(LOCAL_LOG_DIR, folder_name, "train_log.json"),
        os.path.join(PROJECT_ROOT, "experiments", folder_name, "train_log.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c, candidates
    return None, candidates


def guard_exp03_path(log_path: str) -> None:
    """防止按路径误用旧的 dropout=0.5 重跑记录画 Figure 7。"""
    norm = os.path.normpath(log_path).replace("\\", "/")
    if "/exp03_sequence_relog/" in norm:
        raise ValueError(
            "这是旧的 exp03_sequence_relog 记录（dropout=0.5），与论文中的 exp03 不一致。\n"
            "      请改用 exp03_sequence_relog_d03\\train_log.json。"
        )


def load_history(log_path: str):
    with open(log_path, encoding="utf-8") as f:
        data = json.load(f)
    history = data["history"] if isinstance(data, dict) else data
    if not history:
        raise ValueError("日志里没有任何 epoch 记录。")
    return history


def summarize(history):
    """返回 (总 epoch 数, 最佳 epoch, 最佳 val_acc 百分数)。"""
    epochs = [h["epoch"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    if max(h["train_acc"] for h in history) <= 1.0:
        val_acc = [a * 100 for a in val_acc]
    best_idx = val_acc.index(max(val_acc))   # 第一次达到最大值，与保存 best checkpoint 的规则一致
    return len(history), epochs[best_idx], val_acc[best_idx]


def check_against_caption(key: str, summary) -> None:
    n, best_ep, best_val = summary
    exp_n, exp_ep, exp_val = EXPECTED[key]
    if n == exp_n and best_ep == exp_ep and abs(best_val - exp_val) < 0.05:
        return
    msg = (f"日志内容与论文图注不一致：\n"
           f"      这个日志：{n} 个 epoch，best epoch {best_ep}，best val {best_val:.2f}%\n"
           f"      论文图注：{exp_n} 个 epoch，best epoch {exp_ep}，best val {exp_val:.2f}%")
    if key == "exp03" and (n, best_ep, round(best_val, 3)) == OLD_EXP03_SIGNATURE:
        msg += ("\n      这正是旧的 dropout=0.5 重跑记录（exp03_sequence_relog），不能用于 Figure 7。"
                "\n      请改用 exp03_sequence_relog_d03\\train_log.json。")
    else:
        msg += "\n      如果确实要画这份新数据，加参数 --no-check（并同步修改论文图注）。"
    raise ValueError(msg)


def plot_curves(history, title: str, out_path: str) -> None:
    """通用训练曲线绘图函数：左图 Loss，右图 Accuracy。"""
    has_val_loss = "val_loss" in history[0]

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    # exp01/exp02/exp04 存的 acc 已经是百分数；exp03/exp03b 重跑存的是 0–1 小数。
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


def main():
    parser = argparse.ArgumentParser(description="生成 Figure 4/6/7/8/9 训练曲线")
    for key, _, _, out_name, fig_no in JOBS:
        parser.add_argument(f"--{key}", default=None,
                            help=f"{fig_no}（{out_name}）使用的 train_log.json 路径；不给则自动查找")
    parser.add_argument("--only", nargs="+", choices=[j[0] for j in JOBS],
                        help="只画指定的图，例如 --only exp03")
    parser.add_argument("--out-dir", default=SCRIPT_DIR, help="图片输出文件夹（默认：本脚本所在文件夹）")
    parser.add_argument("--font", default=FONT_FAMILY, help="字体名（默认 Times New Roman）")
    parser.add_argument("--no-check", action="store_true",
                        help="不与论文图注核对（仅在确实要画新数据时使用）")
    args = parser.parse_args()

    check_font(args.font)
    os.makedirs(args.out_dir, exist_ok=True)

    explicit = {key: getattr(args, key) for key, *_ in JOBS}
    if args.only:
        selected = set(args.only)
    elif any(explicit.values()):
        selected = {k for k, v in explicit.items() if v}
    else:
        selected = {j[0] for j in JOBS}

    ok, failed = [], []
    for key, folder, title, out_name, fig_no in JOBS:
        if key not in selected:
            continue
        label = f"{fig_no} ({out_name})"
        try:
            if explicit[key]:
                log_path = os.path.abspath(explicit[key])
                if not os.path.exists(log_path):
                    raise ValueError(f"找不到指定的日志：{log_path}")
            else:
                log_path, tried = find_log(folder)
                if log_path is None:
                    raise ValueError("找不到日志，已查找：\n      " + "\n      ".join(tried))
            if key == "exp03":
                guard_exp03_path(log_path)
            history = load_history(log_path)
            summary = summarize(history)
            if not args.no_check:
                check_against_caption(key, summary)
            out_path = os.path.join(args.out_dir, out_name)
            plot_curves(history, title, out_path)
            n, best_ep, best_val = summary
            ok.append(f"[OK]   {label}: {n} 个 epoch，best epoch {best_ep}，best val {best_val:.2f}%\n"
                      f"       日志：{log_path}")
        except ValueError as e:
            failed.append(f"[失败] {label}: {e}")

    print("\n==== 生成结果 ====")
    for line in ok + failed:
        print(line)
    print(f"\n图片保存在：{os.path.abspath(args.out_dir)}")
    if failed:
        print(f"\n有 {len(failed)} 张图没有生成，请按上面的提示处理。")
        sys.exit(1)


if __name__ == "__main__":
    main()
