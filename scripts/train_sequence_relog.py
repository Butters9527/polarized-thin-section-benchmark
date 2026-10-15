r"""三大岩类分类 exp03 消光序列 Temporal Attention 训练脚本 —— _relog 版本（v1.0.1 修正）。

用途：以与原始 exp03 相同的超参数重新训练，并逐 epoch 记录训练过程（用于论文 Figure 7）。
v1.0.1 修正：显式传入 dropout=0.3（原始 exp03 checkpoint 的实际取值）。v1.0.0 中的版本没有
传入 dropout，使用了 TemporalAttentionModel 的类默认值 0.5，因此其重跑曲线与原始 exp03 不一致。
输出目录相应改为 experiments/exp03_sequence_relog_d03。

与原始 train_sequence.py 的唯一区别：
  1. 每个 epoch 结束后把 train_loss/train_acc/val_loss/val_acc 记录进 history 列表，
     训练结束（或 early stop）后写入 exp_dir/train_log.json，用于事后画训练曲线。
  2. 默认输出目录改为 experiments/exp03_sequence_relog_d03，不会覆盖原始的
     experiments/exp03_sequence/best.pth —— 原始 checkpoint 和论文里已经报告的
     78.79% 等数字完全不受影响，这次重训只是为了拿到一条形状正确的训练曲线。

其余逻辑（数据集、模型构造、优化器、调度器、early stopping 判断）与原脚本逐行一致，
除显式传入 dropout=0.3 外，没有改动任何超参数。

用法（PowerShell，项目根目录下，先激活 venv）：
    .\.venv\Scripts\python.exe scripts/train_sequence_relog.py

跑完会在 experiments/exp03_sequence_relog_d03/ 下生成：
    best.pth        —— 重训得到的权重（仅用于验证复现性，不用于论文数字）
    train_log.json  —— 逐 epoch 的 train/val loss 和 accuracy，用于绘制训练曲线
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from mineral_id.data import SequenceDataset
from mineral_id.data.transforms import get_eval_transform, get_train_transform
from mineral_id.models import TemporalAttentionModel


# ----------------------------------------------------------------------
# 数据（与原脚本一致）
# ----------------------------------------------------------------------

def build_loaders_seq(
    splits_dir: Path,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    train_ds = SequenceDataset(
        splits_dir / "train.csv",
        mode="train",
        transform=get_train_transform(),
    )
    val_ds = SequenceDataset(
        splits_dir / "val.csv",
        mode="val",
        transform=get_eval_transform(),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    print(f"train: {len(train_ds)} 条序列  val: {len(val_ds)} 条序列")
    return train_loader, val_loader


# ----------------------------------------------------------------------
# 训练 / 验证一个 epoch（与原脚本一致）
# ----------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for seqs, labels in loader:
        seqs, labels = seqs.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast("cuda", dtype=torch.bfloat16):
            logits = model(seqs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for seqs, labels in loader:
        seqs, labels = seqs.to(device), labels.to(device)
        with autocast("cuda", dtype=torch.bfloat16):
            logits = model(seqs)
            loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# ----------------------------------------------------------------------
# 主函数
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="exp03 消光序列 Temporal Attention 训练（_relog 版）")
    parser.add_argument("--splits-dir",   default="data/processed/splits")
    parser.add_argument("--epochs",       type=int, default=30)
    parser.add_argument("--batch-size",   type=int, default=8)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience",     type=int, default=5)
    parser.add_argument("--num-workers",  type=int, default=2)
    # 唯一改动：默认输出目录换成 _relog 后缀，不覆盖原始 exp03_sequence
    parser.add_argument("--dropout",      type=float, default=0.3)   # 原始 exp03 的实际取值
    parser.add_argument("--exp-dir",      default="experiments/exp03_sequence_relog_d03")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_loaders_seq(
        Path(args.splits_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # 与原脚本完全一致：不传 dropout，使用 TemporalAttentionModel 的类默认值（0.5）
    # 显式传入 dropout，与原始 exp03 一致（不依赖类默认值）
    model = TemporalAttentionModel(dropout=args.dropout).to(device)
    print(f"dropout = {args.dropout}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = GradScaler("cuda")

    best_val_acc = 0.0
    patience_counter = 0
    history = []  # <-- 新增：逐epoch记录

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        # <-- 新增：记录本 epoch 数据
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        with open(exp_dir / "train_log.json", "w") as f:
            json.dump(history, f, indent=2)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_acc": val_acc},
                exp_dir / "best.pth",
            )
            tag = "[best]"
        else:
            patience_counter += 1
            tag = f"(patience {patience_counter}/{args.patience})"

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"best {best_val_acc:.4f} {tag}"
        )

        if patience_counter >= args.patience:
            print(f"Early stopping：val_acc 连续 {args.patience} 个 epoch 无提升，停止训练。")
            break

    print(f"训练结束，最佳 val_acc = {best_val_acc:.4f}，模型已保存至 {exp_dir / 'best.pth'}")
    print(f"逐epoch日志已保存至 {exp_dir / 'train_log.json'}")


if __name__ == "__main__":
    main()
