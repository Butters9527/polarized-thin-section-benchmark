"""test 集最终评估脚本。"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from mineral_id.data import DualStreamDataset, RockClassificationDataset, SequenceDataset
from mineral_id.data.transforms import get_eval_transform
from mineral_id.models import TemporalAttentionModel
from mineral_id.training.train import build_model
from mineral_id.training.train_dual import build_model_dual

# 类名映射：英文 → 中文
CLASS_NAMES_ZH = {
    "igneous":     "火成岩",
    "metamorphic": "变质岩",
    "sedimentary": "沉积岩",
}
# 按 CLASS_TO_IDX 顺序（0/1/2）排列的中文类名
LABELS_ZH = [
    CLASS_NAMES_ZH["igneous"],
    CLASS_NAMES_ZH["metamorphic"],
    CLASS_NAMES_ZH["sedimentary"],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="三大岩类分类评估")
    parser.add_argument("--checkpoint",  default="experiments/exp01_baseline/best.pth")
    parser.add_argument("--splits-dir",  default="data/processed/splits")
    parser.add_argument("--batch-size",  type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--dual",        action="store_true", help="使用双流模型与 DualStreamDataset")
    parser.add_argument("--seq",         action="store_true", help="使用序列模型与 SequenceDataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在：{ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 重建模型并加载权重
    if args.seq:
        model = TemporalAttentionModel().to(device)
    elif args.dual:
        model = build_model_dual().to(device)
    else:
        model = build_model().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 构建 test DataLoader
    transform = get_eval_transform()
    if args.seq:
        test_ds = SequenceDataset(
            Path(args.splits_dir) / "test.csv",
            mode="test",
            transform=transform,
        )
    elif args.dual:
        test_ds = DualStreamDataset(
            Path(args.splits_dir) / "test.csv",
            mode="test",
            transform=transform,
        )
    else:
        test_ds = RockClassificationDataset(
            Path(args.splits_dir) / "test.csv",
            mode="test",
            transform=transform,
        )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"test 集：{len(test_ds)} 张图像")
    print(f"checkpoint：{ckpt_path}  (训练时最佳 val_acc = {checkpoint['val_acc']:.4f}，Epoch {checkpoint['epoch']})\n")

    # 推理：收集所有预测与真实标签
    criterion = nn.CrossEntropyLoss()
    all_preds, all_labels = [], []
    total_loss, total = 0.0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            total += images.size(0)
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    test_loss = total_loss / total
    test_acc  = sum(p == l for p, l in zip(all_preds, all_labels)) / total

    # 摘要
    print(f"test_loss = {test_loss:.4f}  test_acc = {test_acc:.4f}  ({int(test_acc * total)}/{total})\n")

    # 混淆矩阵（行=真实，列=预测）
    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=LABELS_ZH, columns=LABELS_ZH)
    cm_df.index.name   = "真实\\预测"
    print("=== 混淆矩阵 ===")
    print(cm_df.to_string())
    print()

    # 每类 Precision / Recall / F1
    print("=== 分类报告 ===")
    print(classification_report(all_labels, all_preds, target_names=LABELS_ZH))


if __name__ == "__main__":
    main()
