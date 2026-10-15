r"""
Backbone comparison for the dual-modal PPL + XPL@0° early-fusion setting.

The 6-channel input adaptation follows the same initialization rule for each backbone:
the pretrained 3-channel input weights are copied to both modalities and scaled by 0.5.

Supported torchvision backbones:
    resnet50
    efficientnet_b0
    vit_b_16
    swin_t

Verified commands used for the comparison:
    python scripts/backbone_comparison.py --backbone resnet50 --batch-size 16
    python scripts/backbone_comparison.py --backbone efficientnet_b0 --batch-size 16
    python scripts/backbone_comparison.py --backbone vit_b_16 --batch-size 8
    python scripts/backbone_comparison.py --backbone swin_t --batch-size 16

ViT-B/16 uses batch size 8 because of its higher memory requirement on the 8 GB GPU;
the other three backbones use batch size 16.

Outputs:
    experiments/backbone_{name}/best.pth
    experiments/backbone_{name}/eval_log.txt
    experiments/backbone_comparison_summary.csv
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import models

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mineral_id.data.dataset_dual import DualStreamDataset
from mineral_id.data.transforms import get_train_transform, get_eval_transform

SEED = 42
N_CLASSES = 3
CLASS_NAMES_ZH = ["火成岩", "变质岩", "沉积岩"]
SPLITS_DIR = ROOT / "data" / "processed" / "splits"

LR_DEFAULT = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5
MAX_EPOCHS = 30


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── 通用的"3通道 -> 6通道"首层扩展：复制权重 + 整体减半（同 Equation 1）────────
def _expand_conv_to_6ch(old_conv: nn.Conv2d) -> nn.Conv2d:
    new_conv = nn.Conv2d(
        in_channels=6,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None),
    )
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight * 0.5
        new_conv.weight[:, 3:] = old_conv.weight * 0.5
        if old_conv.bias is not None:
            new_conv.bias[:] = old_conv.bias
    return new_conv


def build_model(backbone_name: str) -> nn.Module:
    """构建 6 通道输入、3 类输出的双模态模型，backbone 可替换。"""

    if backbone_name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        m.conv1 = _expand_conv_to_6ch(m.conv1)
        m.fc = nn.Linear(m.fc.in_features, N_CLASSES)
        return m

    elif backbone_name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        # features[0][0] 是 Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        old_conv = m.features[0][0]
        m.features[0][0] = _expand_conv_to_6ch(old_conv)
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, N_CLASSES)
        return m

    elif backbone_name == "vit_b_16":
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        # conv_proj 是 Conv2d(3, 768, kernel_size=16, stride=16)（patch embedding）
        old_conv = m.conv_proj
        m.conv_proj = _expand_conv_to_6ch(old_conv)
        in_features = m.heads.head.in_features
        m.heads.head = nn.Linear(in_features, N_CLASSES)
        return m

    elif backbone_name == "swin_t":
        m = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        # features[0][0] 是 Conv2d(3, 96, kernel_size=4, stride=4)（patch embedding）
        old_conv = m.features[0][0]
        m.features[0][0] = _expand_conv_to_6ch(old_conv)
        in_features = m.head.in_features
        m.head = nn.Linear(in_features, N_CLASSES)
        return m

    else:
        raise ValueError(f"不支持的 backbone: {backbone_name}")


def build_loaders(batch_size: int):
    train_ds = DualStreamDataset(SPLITS_DIR / "train.csv", mode="train",
                                  transform=get_train_transform(), align_augmentation=True)
    val_ds = DualStreamDataset(SPLITS_DIR / "val.csv", mode="val",
                                transform=get_eval_transform(), align_augmentation=False)
    test_ds = DualStreamDataset(SPLITS_DIR / "test.csv", mode="test",
                                 transform=get_eval_transform(), align_augmentation=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader


@torch.no_grad()
def run_inference(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with autocast("cuda", dtype=torch.bfloat16):
            outputs = model(images)
        all_preds.extend(outputs.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return np.array(all_preds), np.array(all_labels)


def quick_val_acc(model, val_loader, device) -> float:
    preds, labels = run_inference(model, val_loader, device)
    return float((preds == labels).mean() * 100.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True,
                         choices=["resnet50", "efficientnet_b0", "vit_b_16", "swin_t"])
    parser.add_argument("--batch-size", type=int, default=16,
                         help="vit_b_16 显存吃紧的话调小，比如 8 或 4")
    parser.add_argument("--lr", type=float, default=LR_DEFAULT)
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Backbone: {args.backbone}  |  batch_size={args.batch_size}")

    exp_dir = ROOT / "experiments" / f"backbone_{args.backbone}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = build_loaders(args.batch_size)
    print(f"train: {len(train_loader.dataset)}  val: {len(val_loader.dataset)}  "
          f"test: {len(test_loader.dataset)}")

    model = build_model(args.backbone).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"参数量：{n_params:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    best_val_acc = -1.0
    best_state = None
    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast("cuda", dtype=torch.bfloat16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()
        train_acc = correct / total * 100.0

        val_acc = quick_val_acc(model, val_loader, device)
        print(f"Epoch {epoch:02d}/{MAX_EPOCHS} | train_acc {train_acc:.1f}% | "
              f"val_acc {val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    torch.save({"backbone": args.backbone, "epoch": best_epoch,
                "model_state_dict": best_state, "val_acc": best_val_acc},
               exp_dir / "best.pth")

    # ── held-out test 最终评估（跟 scripts/evaluate.py 输出格式对齐）──
    preds, labels = run_inference(model, test_loader, device)
    test_acc = float((preds == labels).mean())
    macro_f1 = f1_score(labels, preds, average="macro")

    log_lines = []
    log_lines.append(f"backbone: {args.backbone}  (参数量 {n_params:.1f}M)")
    log_lines.append(f"test 集：{len(labels)} 张图像")
    log_lines.append(f"checkpoint 训练时最佳 val_acc = {best_val_acc:.2f}%，Epoch {best_epoch}\n")
    log_lines.append(f"test_acc = {test_acc:.4f}  ({int(test_acc*len(labels))}/{len(labels)})\n")

    cm = confusion_matrix(labels, preds)
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES_ZH, columns=CLASS_NAMES_ZH)
    log_lines.append("=== 混淆矩阵 ===")
    log_lines.append(cm_df.to_string())
    log_lines.append("")
    log_lines.append("=== 分类报告 ===")
    log_lines.append(classification_report(labels, preds, target_names=CLASS_NAMES_ZH))

    log_text = "\n".join(log_lines)
    print("\n" + log_text)

    with open(exp_dir / "eval_log.txt", "w", encoding="utf-8") as f:
        f.write(log_text)

    # ── 追加写入汇总 csv，方便几个 backbone 跑完后一起看 ──
    summary_path = ROOT / "experiments" / "backbone_comparison_summary.csv"
    is_new = not summary_path.exists()
    rec = []
    for c in range(N_CLASSES):
        mask = labels == c
        rec.append(float((preds[mask] == c).mean()) if mask.sum() > 0 else 0.0)

    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["backbone", "params_M", "test_oa", "test_macro_f1",
                              "rec_igneous", "rec_metamorphic", "rec_sedimentary",
                              "best_epoch", "best_val_acc"])
        writer.writerow([args.backbone, f"{n_params:.1f}", f"{test_acc*100:.2f}",
                          f"{macro_f1:.4f}", f"{rec[0]:.3f}", f"{rec[1]:.3f}", f"{rec[2]:.3f}",
                          best_epoch, f"{best_val_acc:.2f}"])

    print(f"\n结果已追加到：{summary_path}")
    print(f"Backbone summary updated: {summary_path}")


if __name__ == "__main__":
    main()
