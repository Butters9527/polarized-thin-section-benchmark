"""三大岩类分类 exp02 双流早期融合训练脚本。

【本版本改动】仅为补画学习曲线新增以下内容，其余训练逻辑一字未改：
  1. 新增 set_seed()，固定随机种子（与 train_sequence_rpe.py 保持一致），
     对应论文 Section 4.5 "Global random seeds were fixed at 42" 的描述。
     原脚本没有这一步，这是本次唯一的"逻辑性"改动。
  2. 新增 --seed 命令行参数，默认 42。
  3. 每个 epoch 结束后把 train_loss / train_acc / val_loss / val_acc 存进
     history 列表，训练结束后写成 train_log.json（格式对齐 exp04 的
     train_log.json，方便后续统一画图）。
  4. --exp-dir 默认值改成了 experiments/exp02_dual_stream_relog，
     不会覆盖原来 experiments/exp02_dual_stream/best.pth
     （那个 checkpoint 是论文里 84.85% 等数字的来源，不能动）。
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import models

from mineral_id.data import DualStreamDataset
from mineral_id.data.transforms import get_eval_transform, get_train_transform


# ----------------------------------------------------------------------
# 随机种子（新增）
# ----------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """固定随机种子，保证可复现（与 train_sequence_rpe.py 保持一致）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------
# 模型（未改动）
# ----------------------------------------------------------------------

def build_model_dual() -> nn.Module:
    """加载 ResNet-50 预训练权重，修改 conv1 接受 6 通道双流输入，fc 改为 3 类输出。

    权重初始化策略：
      - 前 3 通道：直接复用 ImageNet 预训练权重
      - 后 3 通道：复制前 3 通道权重
      - 整体 ×0.5，保持激活分布与原预训练一致
    """
    base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    old_conv1 = base.conv1  # Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    pretrained_weight = old_conv1.weight.data.clone()  # [64, 3, 7, 7]

    new_conv1 = nn.Conv2d(
        in_channels=6,
        out_channels=old_conv1.out_channels,
        kernel_size=old_conv1.kernel_size,
        stride=old_conv1.stride,
        padding=old_conv1.padding,
        bias=False,
    )
    with torch.no_grad():
        new_conv1.weight[:, :3, :, :] = pretrained_weight
        new_conv1.weight[:, 3:, :, :] = pretrained_weight
        new_conv1.weight *= 0.5

    base.conv1 = new_conv1
    base.fc = nn.Linear(base.fc.in_features, 3)
    return base


# ----------------------------------------------------------------------
# 数据（未改动）
# ----------------------------------------------------------------------

def build_loaders(
    splits_dir: Path,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    """构建 train / val DualStream DataLoader。"""
    train_ds = DualStreamDataset(
        splits_dir / "train.csv",
        mode="train",
        transform=get_train_transform(),
    )
    val_ds = DualStreamDataset(
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
    return train_loader, val_loader


# ----------------------------------------------------------------------
# 单 epoch 训练 / 验证（未改动）
# ----------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> tuple[float, float]:
    """运行一个训练 epoch，返回 (avg_loss, accuracy)。"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast("cuda", dtype=torch.bfloat16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """在 val/test 集上评估，返回 (avg_loss, accuracy)。"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        with autocast("cuda", dtype=torch.bfloat16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


# ----------------------------------------------------------------------
# 主训练循环
# ----------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)  # 新增

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_loaders(
        Path(args.splits_dir), args.batch_size, args.num_workers
    )
    print(f"train: {len(train_loader.dataset)} 张  val: {len(val_loader.dataset)} 张")

    model = build_model_dual().to(device)
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
    history: list[dict] = []  # 新增：逐epoch历史记录

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        # 新增：记录本epoch历史（accuracy存成百分比，和exp04的train_log.json格式一致）
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc * 100,
                "val_loss": val_loss,
                "val_acc": val_acc * 100,
            }
        )

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                },
                exp_dir / "best.pth",
            )
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"best {best_val_acc:.4f}"
            + (" [best]" if is_best else f" (patience {patience_counter}/{args.patience})")
        )

        if patience_counter >= args.patience:
            print(f"Early stopping：val_acc 连续 {args.patience} 个 epoch 无提升，停止训练。")
            break

    # 新增：保存训练历史（格式对齐 train_sequence_rpe.py 的 train_log.json）
    log_path = exp_dir / "train_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                "history": history,
                "best_val_acc": best_val_acc,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"训练历史已保存至 {log_path}")

    print(f"训练结束，最佳 val_acc = {best_val_acc:.4f}，模型已保存至 {exp_dir / 'best.pth'}")


# ----------------------------------------------------------------------
# 参数解析
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="三大岩类分类 exp02 双流早期融合训练")
    parser.add_argument("--splits-dir",    default="data/processed/splits")
    # 改动：默认输出目录换成新文件夹，不覆盖原 experiments/exp02_dual_stream
    parser.add_argument("--exp-dir",       default="experiments/exp02_dual_stream_relog")
    parser.add_argument("--epochs",        type=int,   default=30)
    parser.add_argument("--batch-size",    type=int,   default=16)
    parser.add_argument("--num-workers",   type=int,   default=2)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--weight-decay",  type=float, default=1e-4)
    parser.add_argument("--patience",      type=int,   default=5)
    parser.add_argument("--seed",          type=int,   default=42)  # 新增
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
