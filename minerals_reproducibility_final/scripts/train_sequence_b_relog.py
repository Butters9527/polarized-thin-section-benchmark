r"""三大岩类分类 exp03b 消光序列 Temporal Attention 训练脚本（冻结前两层+高Dropout）—— _relog 版本。

与原始 train_sequence_b.py 的唯一区别：
  1. 每个 epoch 结束后记录 train/val loss/acc 到 history，写入
     exp_dir/train_log.json，用于事后画训练曲线。
  2. 默认输出目录改为 experiments/exp03b_frozen_relog，不覆盖原始
     experiments/exp03b_frozen/best.pth，论文数字不受影响。

其余逻辑（冻结层选择、dropout、优化器、调度器、early stopping）与原脚本逐行一致。

用法（PowerShell，项目根目录下，先激活 venv）：
    .\.venv\Scripts\python.exe scripts/train_sequence_b_relog.py

跑完会在 experiments/exp03b_frozen_relog/ 下生成 best.pth + train_log.json。
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
    parser = argparse.ArgumentParser(description="exp03b 消光序列 Temporal Attention（冻结+高Dropout，_relog版）")
    parser.add_argument("--splits-dir",   default="data/processed/splits")
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--batch-size",   type=int,   default=8)
    parser.add_argument("--lr",           type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience",     type=int,   default=5)
    parser.add_argument("--num-workers",  type=int,   default=2)
    parser.add_argument("--dropout",      type=float, default=0.5)
    # 唯一改动：默认输出目录换成 _relog 后缀，不覆盖原始 exp03b_frozen
    parser.add_argument("--exp-dir",      default="experiments/exp03b_frozen_relog")
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

    model = TemporalAttentionModel(dropout=args.dropout).to(device)

    # 冻结 backbone conv1/bn1/layer1/layer2（与原脚本一致）
    for name, param in model.backbone.named_parameters():
        if any(name.startswith(prefix) for prefix in
               ["conv1", "bn1", "layer1", "layer2"]):
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数：{trainable / 1e6:.2f} M")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
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
