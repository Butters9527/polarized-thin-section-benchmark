"""
5-fold Cross-Validation for exp01 (single-frame PPL) and exp02 (dual-stream PPL+XPL).

Usage:
    python scripts/kfold_cv.py --mode exp01
    python scripts/kfold_cv.py --mode exp02

Output:
    experiments/kfold_exp01/  (fold0/ ~ fold4/ + summary.json)
    experiments/kfold_exp02/  (fold0/ ~ fold4/ + summary.json)
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# ── project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mineral_id.data.dataset import RockClassificationDataset
from mineral_id.data.dataset_dual import DualStreamDataset
from mineral_id.data.transforms import get_train_transform, get_eval_transform

# ── constants ─────────────────────────────────────────────────────────────────
SEED = 42
N_FOLDS = 5
N_CLASSES = 3
CLASS_NAMES = ["igneous", "metamorphic", "sedimentary"]

# Hyperparameters — identical to exp01 / exp02
LR = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 16
PATIENCE = 5
MAX_EPOCHS = 50          # safety ceiling; early stopping kicks in much sooner
DROPOUT = 0.3

SLIDE_SUMMARY = ROOT / "data" / "processed" / "dataset_exploration" / "slide_summary.csv"
DATASET_SUMMARY = ROOT / "data" / "processed" / "dataset_exploration" / "dataset_summary.csv"


# ── reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── model factory ─────────────────────────────────────────────────────────────
def build_model(mode: str) -> nn.Module:
    """Return a freshly initialised ResNet-50 model for the given mode."""
    import torchvision.models as models

    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    if mode == "exp01":
        # Standard 3-channel ResNet-50
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(in_features, N_CLASSES),
        )

    elif mode == "exp02":
        # Adapt conv1 to accept 6-channel input (PPL [3ch] + XPL [3ch])
        old_conv = backbone.conv1                          # (64, 3, 7, 7)
        new_conv = nn.Conv2d(
            6, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Channel-average initialisation: duplicate the 3-ch pretrained weights
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


# ── dataset factory ───────────────────────────────────────────────────────────
def make_dataset(mode: str, uid_list: list[str], split: str, master_df: pd.DataFrame):
    """Return a Dataset for the given uid list and train/val split."""
    transform = get_train_transform() if split == "train" else get_eval_transform()

    mask = master_df["uid"].isin(set(uid_list))
    filtered_df = master_df[mask].reset_index(drop=True)

    if mode == "exp01":
        return RockClassificationDataset(
            df=filtered_df,
            transform=transform,
        )
    elif mode == "exp02":
        return DualStreamDataset(
            df=filtered_df,
            transform=transform,
            align_augmentation=(split == "train"),
        )


# ── evaluation helper ─────────────────────────────────────────────────────────
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch[0], (list, tuple)):
                inputs = [x.to(device) for x in batch[0]]
                labels = batch[1].to(device)
                with autocast(dtype=torch.bfloat16):
                    outputs = model(*inputs)
            else:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                with autocast(dtype=torch.bfloat16):
                    outputs = model(inputs)

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
    }


# ── one fold ──────────────────────────────────────────────────────────────────
def train_one_fold(
    mode: str,
    fold_idx: int,
    train_slide_ids: list[str],
    val_slide_ids: list[str],
    out_dir: Path,
    device: torch.device,
    master_df: pd.DataFrame,
) -> dict:
    set_seed(SEED + fold_idx)   # different seed per fold, reproducible

    print(f"\n{'='*60}")
    print(f"  Fold {fold_idx}  |  train: {len(train_slide_ids)} slides"
          f"  |  val: {len(val_slide_ids)} slides")
    print(f"{'='*60}")

    # Datasets & loaders
    train_ds = make_dataset(mode, train_slide_ids, "train", master_df)
    val_ds   = make_dataset(mode, val_slide_ids,   "val",   master_df)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    # Model, optimiser, scheduler, loss
    model = build_model(mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS
    )
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    best_val_acc = -1.0
    patience_counter = 0
    best_metrics = {}

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── train ──
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            optimizer.zero_grad()

            if isinstance(batch[0], (list, tuple)):
                inputs = [x.to(device) for x in batch[0]]
                labels = batch[1].to(device)
                with autocast(dtype=torch.bfloat16):
                    outputs = model(*inputs)
            else:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                with autocast(dtype=torch.bfloat16):
                    outputs = model(inputs)

            with autocast(dtype=torch.bfloat16):
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total   += labels.size(0)

        scheduler.step()
        train_acc = correct / total * 100.0
        train_loss = running_loss / total

        # ── val ──
        val_metrics = evaluate(model, val_loader, device)
        val_acc = val_metrics["oa"]

        print(f"  Ep{epoch:02d}  train_loss={train_loss:.4f}"
              f"  train_acc={train_acc:.1f}%  val_acc={val_acc:.1f}%"
              f"  F1={val_metrics['macro_f1']:.3f}")

        # ── early stopping ──
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_metrics = val_metrics.copy()
            torch.save(model.state_dict(), out_dir / "best.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stop at epoch {epoch} (patience={PATIENCE})")
                break

    print(f"  Best val: OA={best_metrics['oa']:.2f}%"
          f"  F1={best_metrics['macro_f1']:.3f}"
          f"  Rec-I={best_metrics['rec_igneous']:.3f}"
          f"  Rec-M={best_metrics['rec_metamorphic']:.3f}"
          f"  Rec-S={best_metrics['rec_sedimentary']:.3f}")

    return best_metrics


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["exp01", "exp02"], required=True,
                        help="exp01 = single-frame PPL; exp02 = dual-stream PPL+XPL")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Mode: {args.mode}")

    # ── load per-image master table (used by make_dataset) ──
    master_df = pd.read_csv(DATASET_SUMMARY)
    master_df["uid"] = (
        master_df["rock_class"] + "_"
        + master_df["rock_id"].astype(str) + "_"
        + master_df["slide_id"].astype(str)
    )

    # ── load slide-level metadata ──
    df = pd.read_csv(SLIDE_SUMMARY)
    # Composite unique key: rock_class + rock_id + slide_id
    df["uid"] = df["rock_class"] + "_" + df["rock_id"].astype(str) + "_" + df["slide_id"].astype(str)

    # Exclude 3 known anomalous slides by composite key
    anomaly_mask = (
        ((df["rock_class"] == "sedimentary") & (df["rock_id"] == 12) & (df["slide_id"] == 3))
        | ((df["rock_class"] == "sedimentary") & (df["rock_id"] == 16) & (df["slide_id"] == 1))
        | ((df["rock_class"] == "sedimentary") & (df["rock_id"] == 9) & (df["slide_id"] == 1))
    )
    df = df[~anomaly_mask].reset_index(drop=True)
    assert len(df) == 321, f"Expected 321 slides after exclusion, got {len(df)}"

    # Encode class labels for StratifiedKFold
    class_map = {"igneous": 0, "metamorphic": 1, "sedimentary": 2}
    df["label"] = df["rock_class"].map(class_map)

    slide_ids = df["uid"].values
    labels    = df["label"].values

    # ── 5-fold stratified split (slide-level) ──
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    out_root = ROOT / "experiments" / f"kfold_{args.mode}"
    out_root.mkdir(parents=True, exist_ok=True)

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(slide_ids, labels)):
        fold_dir = out_root / f"fold{fold_idx}"
        fold_dir.mkdir(exist_ok=True)

        train_slides = slide_ids[train_idx].tolist()
        val_slides   = slide_ids[val_idx].tolist()

        # Sanity check: no leakage
        assert len(set(train_slides) & set(val_slides)) == 0

        metrics = train_one_fold(
            mode=args.mode,
            fold_idx=fold_idx,
            train_slide_ids=train_slides,
            val_slide_ids=val_slides,
            out_dir=fold_dir,
            device=device,
            master_df=master_df,
        )
        metrics["fold"] = fold_idx
        fold_results.append(metrics)

        # Save per-fold result
        with open(fold_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    # ── aggregate ──
    keys = ["oa", "macro_f1", "rec_igneous", "rec_metamorphic", "rec_sedimentary"]
    summary = {}
    for k in keys:
        vals = [r[k] for r in fold_results]
        summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=0))}

    summary["fold_results"] = fold_results
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── pretty print ──
    print(f"\n{'='*60}")
    print(f"  {args.mode.upper()}  5-fold CV Summary")
    print(f"{'='*60}")
    label_map = {
        "oa":               "OA    (%)",
        "macro_f1":         "F1        ",
        "rec_igneous":      "Rec-I     ",
        "rec_metamorphic":  "Rec-M     ",
        "rec_sedimentary":  "Rec-S     ",
    }
    for k, label in label_map.items():
        m = summary[k]["mean"]
        s = summary[k]["std"]
        unit = "%" if k == "oa" else ""
        print(f"  {label}: {m:.2f}{unit} ± {s:.2f}{unit}")
    print(f"{'='*60}\n")
    print(f"Full results saved to: {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
