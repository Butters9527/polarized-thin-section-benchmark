r"""
嵌套交叉验证（Nested Cross-Validation）— exp01 / exp02

现有 5-fold CV（kfold_cv.py）中，
每个 fold 的 validation 集同时被用来做 (a) early stopping 监控 和 (b) 最终指标汇报，
这会导致汇报的 CV 分数偏乐观（因为"选出来验证集分数最高的那个 checkpoint"本身
就是在拟合验证集）。

本脚本的做法（R2 原话："a nested CV, or a held-out inner split for early stopping
distinct from the outer evaluation fold"——用的是后一种，计算量只是原来的 1 倍，
不是嵌套 5×5 的 25 倍，8GB 显存 + 单卡跑得动）：

    外层：5-fold StratifiedKFold（薄片级分层，与现有 kfold_cv.py 完全一致，同一个 seed=42）
    内层：外层 train 部分（约 256~257 个薄片）再切一刀，85% 做 inner-train 用于梯度更新，
          15% 做 inner-val，只用来监控 early stopping、挑 best checkpoint
    评估：外层 val（约 64 个薄片，全程没有出现在任何训练/调参步骤里）—— 这才是
          本脚本真正汇报的"干净"分数，不再有信息泄漏

用法（PowerShell，项目根目录下，先激活 venv）：
    .\.venv\Scripts\python.exe scripts/nested_cv.py --mode exp01
    .\.venv\Scripts\python.exe scripts/nested_cv.py --mode exp02

输出：
    experiments/nested_cv_exp01/fold{0-4}/{best.pth, metrics.json}
    experiments/nested_cv_exp01/summary.json   （汇总 mean±std，这是论文里要用的数字）
    exp02 同理，目录名 nested_cv_exp02

跟旧的 experiments/kfold_exp01/、kfold_exp02/ 互不覆盖，可以在论文里并排展示两组数字，
说明"未嵌套 CV vs 嵌套 CV"差多少（预期嵌套版本会比现有 90.03%/89.41% 略低，这是预期内的、
也是这次修改本身要证明的点）。

预计耗时：单个 fold 几分钟量级（数据量小），5 fold × 2 个模式，总计约 20-40 分钟。
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# ── project root on sys.path ────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mineral_id.data.dataset import RockClassificationDataset
from mineral_id.data.dataset_dual import DualStreamDataset
from mineral_id.data.transforms import get_train_transform, get_eval_transform

# ── 常量（与 kfold_cv.py 保持一致，便于直接对比）────────────────────────────
SEED = 42
N_OUTER_FOLDS = 5
INNER_VAL_FRACTION = 0.15   # 外层 train 里切多少给 inner-val 做 early stopping
N_CLASSES = 3
CLASS_NAMES = ["igneous", "metamorphic", "sedimentary"]

LR = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 16
PATIENCE = 5
MAX_EPOCHS = 50
DROPOUT = 0.3

SLIDE_SUMMARY = ROOT / "data" / "processed" / "dataset_exploration" / "slide_summary.csv"
DATASET_SUMMARY = ROOT / "data" / "processed" / "dataset_exploration" / "dataset_summary.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── 模型工厂（与 kfold_cv.py 的 build_model 完全一致）────────────────────────
def build_model(mode: str) -> nn.Module:
    import torchvision.models as models

    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    if mode == "exp01":
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(in_features, N_CLASSES),
        )
    elif mode == "exp02":
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight * 0.5
            new_conv.weight[:, 3:, :, :] = old_conv.weight * 0.5
        backbone.conv1 = new_conv

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(in_features, N_CLASSES),
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return backbone


def make_dataset(mode: str, uid_list: list[str], split: str, master_df: pd.DataFrame):
    transform = get_train_transform() if split == "train" else get_eval_transform()
    mask = master_df["uid"].isin(set(uid_list))
    filtered_df = master_df[mask].reset_index(drop=True)

    if mode == "exp01":
        return RockClassificationDataset(df=filtered_df, transform=transform, mode=split if split in ("train","val","test") else "val")
    elif mode == "exp02":
        return DualStreamDataset(
            df=filtered_df,
            transform=transform,
            align_augmentation=(split == "train"),
            mode=split if split in ("train", "val", "test") else "val",
        )


# ── 评估 ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with autocast("cuda", dtype=torch.bfloat16):
            outputs = model(images)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    oa = (all_preds == all_labels).mean() * 100.0
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    per_class_recall = []
    for c in range(N_CLASSES):
        mask = all_labels == c
        rec = (all_preds[mask] == c).mean() if mask.sum() > 0 else 0.0
        per_class_recall.append(float(rec))

    return {
        "oa": float(oa),
        "macro_f1": float(macro_f1),
        "rec_igneous": per_class_recall[0],
        "rec_metamorphic": per_class_recall[1],
        "rec_sedimentary": per_class_recall[2],
        "n_samples": int(len(all_labels)),
    }


# ── 一个外层 fold（内含内层 early-stopping 划分）────────────────────────────
def run_one_outer_fold(
    mode: str,
    fold_idx: int,
    outer_train_uids: list[str],
    outer_val_uids: list[str],
    outer_train_labels: list[int],
    out_dir: Path,
    device: torch.device,
    master_df: pd.DataFrame,
) -> dict:
    set_seed(SEED + fold_idx)

    # ── 内层切分：outer_train → inner_train / inner_val ──
    inner_train_uids, inner_val_uids = train_test_split(
        outer_train_uids,
        test_size=INNER_VAL_FRACTION,
        stratify=outer_train_labels,
        random_state=SEED + 1000 + fold_idx,
    )

    print(f"\n{'='*70}")
    print(f"  Outer Fold {fold_idx}")
    print(f"  inner-train: {len(inner_train_uids)} slides "
          f"| inner-val (early stop only): {len(inner_val_uids)} slides "
          f"| outer-val (reported score): {len(outer_val_uids)} slides")
    print(f"{'='*70}")

    # 三方互不重叠的安全检查
    assert len(set(inner_train_uids) & set(inner_val_uids)) == 0
    assert len(set(inner_train_uids) & set(outer_val_uids)) == 0
    assert len(set(inner_val_uids) & set(outer_val_uids)) == 0

    train_ds = make_dataset(mode, inner_train_uids, "train", master_df)
    inner_val_ds = make_dataset(mode, inner_val_uids, "val", master_df)
    outer_val_ds = make_dataset(mode, outer_val_uids, "val", master_df)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=4, pin_memory=True)
    inner_val_loader = DataLoader(inner_val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                   num_workers=4, pin_memory=True)
    outer_val_loader = DataLoader(outer_val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                   num_workers=4, pin_memory=True)

    model = build_model(mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    best_inner_val_acc = -1.0
    patience_counter = 0
    best_state = None

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

        # 只在 inner-val 上决定是否 early stop / 存 checkpoint
        inner_metrics = evaluate(model, inner_val_loader, device)
        inner_val_acc = inner_metrics["oa"]

        print(f"  Ep{epoch:02d}  train_acc={train_acc:.1f}%  "
              f"inner_val_acc={inner_val_acc:.1f}%")

        if inner_val_acc > best_inner_val_acc:
            best_inner_val_acc = inner_val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stop at epoch {epoch} (patience={PATIENCE}, "
                      f"基于 inner-val，与最终汇报指标使用的 outer-val 完全不重叠)")
                break

    # 加载 inner-val 挑出来的最佳权重，去从没见过的 outer-val 上评估一次
    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / "best.pth")

    outer_metrics = evaluate(model, outer_val_loader, device)
    outer_metrics["best_inner_val_acc"] = best_inner_val_acc

    print(f"  >>> Outer-val（干净估计）: OA={outer_metrics['oa']:.2f}%  "
          f"F1={outer_metrics['macro_f1']:.3f}  "
          f"Rec-M={outer_metrics['rec_metamorphic']:.3f}")

    return outer_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["exp01", "exp02"], required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Mode: {args.mode}  |  Nested CV")

    master_df = pd.read_csv(DATASET_SUMMARY)
    master_df["uid"] = (
        master_df["rock_class"] + "_" + master_df["rock_id"].astype(str)
        + "_" + master_df["slide_id"].astype(str)
    )

    df = pd.read_csv(SLIDE_SUMMARY)
    df["uid"] = df["rock_class"] + "_" + df["rock_id"].astype(str) + "_" + df["slide_id"].astype(str)

    anomaly_mask = (
        ((df["rock_class"] == "sedimentary") & (df["rock_id"] == 12) & (df["slide_id"] == 3))
        | ((df["rock_class"] == "sedimentary") & (df["rock_id"] == 16) & (df["slide_id"] == 1))
        | ((df["rock_class"] == "sedimentary") & (df["rock_id"] == 9) & (df["slide_id"] == 1))
    )
    df = df[~anomaly_mask].reset_index(drop=True)
    assert len(df) == 321, f"Expected 321 slides after exclusion, got {len(df)}"

    class_map = {"igneous": 0, "metamorphic": 1, "sedimentary": 2}
    df["label"] = df["rock_class"].map(class_map)

    slide_ids = df["uid"].values
    labels = df["label"].values

    skf = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=SEED)

    out_root = ROOT / "experiments" / f"nested_cv_{args.mode}"
    out_root.mkdir(parents=True, exist_ok=True)

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(slide_ids, labels)):
        fold_dir = out_root / f"fold{fold_idx}"
        fold_dir.mkdir(exist_ok=True)

        outer_train_uids = slide_ids[train_idx].tolist()
        outer_val_uids = slide_ids[val_idx].tolist()
        outer_train_labels = labels[train_idx].tolist()

        assert len(set(outer_train_uids) & set(outer_val_uids)) == 0

        metrics = run_one_outer_fold(
            mode=args.mode,
            fold_idx=fold_idx,
            outer_train_uids=outer_train_uids,
            outer_val_uids=outer_val_uids,
            outer_train_labels=outer_train_labels,
            out_dir=fold_dir,
            device=device,
            master_df=master_df,
        )
        metrics["fold"] = fold_idx
        fold_results.append(metrics)

        with open(fold_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    keys = ["oa", "macro_f1", "rec_igneous", "rec_metamorphic", "rec_sedimentary"]
    summary = {}
    for k in keys:
        vals = [r[k] for r in fold_results]
        summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=0))}
    summary["fold_results"] = fold_results
    summary["note"] = (
        "Nested CV: early stopping uses an inner held-out split (15% of the outer-train "
        "slides) that is disjoint from the outer-validation fold reported here. "
        "Compare against experiments/kfold_{mode}/summary.json (non-nested) to quantify "
        "the optimism bias from using the same fold for early stopping and evaluation."
    )

    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  {args.mode.upper()}  Nested 5-fold CV Summary（干净估计，可直接写论文）")
    print(f"{'='*70}")
    label_map = {
        "oa": "OA    (%)", "macro_f1": "F1        ",
        "rec_igneous": "Rec-I     ", "rec_metamorphic": "Rec-M     ",
        "rec_sedimentary": "Rec-S     ",
    }
    for k, label in label_map.items():
        m, s = summary[k]["mean"], summary[k]["std"]
        unit = "%" if k == "oa" else ""
        print(f"  {label}: {m:.2f}{unit} ± {s:.2f}{unit}")
    print(f"{'='*70}\n")
    print(f"Full results saved to: {out_root / 'summary.json'}")
    print(f"对比：experiments/kfold_{args.mode}/summary.json 是未嵌套版本，两者差值就是"
          f"非嵌套协议相对于独立外层评估的偏差大小。")


if __name__ == "__main__":
    main()
