r"""
小规模超参数搜索 + 敏感性分析 — exp02（双模态早期融合）

做法：随机搜索（random search），固定预算 N_TRIALS 组配置，每组只在 train/val 上训练
评估（不碰 held-out test，避免"用test调参"这种更严重的问题），记录每组的 val OA / F1，
搜索空间覆盖 lr / weight_decay / dropout / batch_size 四个待评估超参数。

为了让 N_TRIALS 组配置能在单卡 8GB 显存、合理时间内跑完，每组训练用了比正式实验更紧的
patience（3 而不是 5）、更低的 epoch 上限（20 而不是 50）——这是"搜索阶段"的正常做法，
最终选出的最优配置可以再单独用正式设置（patience=5）跑一次确认。

用法（PowerShell，项目根目录下，先激活 venv）：
    .\.venv\Scripts\python.exe scripts/hyperparam_search.py

输出：
    experiments/hparam_search_exp02/search_results.csv   （每组配置 + val 指标）
    experiments/hparam_search_exp02/search_results.json  （同上，程序读取用）

预计耗时：12 组配置，每组几分钟量级，总计约 30-60 分钟。
"""

import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mineral_id.data.dataset_dual import DualStreamDataset
from mineral_id.data.transforms import get_train_transform, get_eval_transform

SEED = 42
N_CLASSES = 3
SPLITS_DIR = ROOT / "data" / "processed" / "splits"
OUT_DIR = ROOT / "experiments" / "hparam_search_exp02"

# 搜索预算：随机抽 N_TRIALS 组，而不是跑满全网格（3^4=81 组太多）
N_TRIALS = 12
MAX_EPOCHS = 20
PATIENCE = 3

# 搜索空间：围绕参考设置给出合理邻域
SEARCH_SPACE = {
    "lr":           [5e-5, 1e-4, 2e-4],
    "weight_decay": [1e-5, 1e-4, 1e-3],
    "dropout":      [0.2, 0.3, 0.5],
    "batch_size":   [8, 16, 32],
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(dropout: float) -> nn.Module:
    import torchvision.models as models

    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    old_conv = backbone.conv1
    new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = old_conv.weight * 0.5
        new_conv.weight[:, 3:, :, :] = old_conv.weight * 0.5
    backbone.conv1 = new_conv
    in_features = backbone.fc.in_features
    backbone.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, N_CLASSES))
    return backbone


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with autocast("cuda", dtype=torch.bfloat16):
            outputs = model(images)
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    oa = (all_preds == all_labels).mean() * 100.0
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return {"oa": float(oa), "macro_f1": float(macro_f1)}


def run_one_trial(trial_idx: int, cfg: dict, device: torch.device) -> dict:
    set_seed(SEED)  # 每组配置用同一个 seed，公平对比（差异只来自超参数本身）

    print(f"\n{'='*70}")
    print(f"  Trial {trial_idx}: lr={cfg['lr']:.0e}  wd={cfg['weight_decay']:.0e}  "
          f"dropout={cfg['dropout']}  batch_size={cfg['batch_size']}")
    print(f"{'='*70}")

    train_ds = DualStreamDataset(SPLITS_DIR / "train.csv", mode="train",
                                  transform=get_train_transform(), align_augmentation=True)
    val_ds = DualStreamDataset(SPLITS_DIR / "val.csv", mode="val",
                                transform=get_eval_transform(), align_augmentation=False)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                               num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=2, pin_memory=True)

    model = build_model(cfg["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                   weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    best_val_acc = -1.0
    best_metrics = {}
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast("cuda", dtype=torch.bfloat16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device)
        print(f"  Ep{epoch:02d}  val_oa={val_metrics['oa']:.1f}%  "
              f"val_f1={val_metrics['macro_f1']:.3f}")

        if val_metrics["oa"] > best_val_acc:
            best_val_acc = val_metrics["oa"]
            best_metrics = val_metrics.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break

    result = dict(cfg)
    result.update({"val_oa": best_metrics["oa"], "val_macro_f1": best_metrics["macro_f1"]})
    print(f"  >>> Trial {trial_idx} best: val_oa={best_metrics['oa']:.2f}%  "
          f"val_f1={best_metrics['macro_f1']:.3f}")
    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 随机抽样 N_TRIALS 组不重复的配置（固定种子，保证可复现）
    rng = random.Random(SEED)
    all_combos = list(itertools.product(
        SEARCH_SPACE["lr"], SEARCH_SPACE["weight_decay"],
        SEARCH_SPACE["dropout"], SEARCH_SPACE["batch_size"],
    ))
    # 将参考配置（lr=1e-4, wd=1e-4, dropout=0.3, batch=16）固定纳入搜索，
    # 便于与其余候选配置直接比较。
    default_cfg = (1e-4, 1e-4, 0.3, 16)
    sampled = [default_cfg] + rng.sample(
        [c for c in all_combos if c != default_cfg], N_TRIALS - 1
    )

    results = []
    for i, (lr, wd, dropout, bs) in enumerate(sampled):
        cfg = {"lr": lr, "weight_decay": wd, "dropout": dropout, "batch_size": bs,
               "is_reference_config": (lr, wd, dropout, bs) == default_cfg}
        res = run_one_trial(i, cfg, device)
        res["is_reference_config"] = cfg["is_reference_config"]
        results.append(res)

        # 每跑完一组就存一次盘，中途中断也不会丢结果
        with open(OUT_DIR / "search_results.json", "w") as f:
            json.dump(results, f, indent=2)
        pd.DataFrame(results).to_csv(OUT_DIR / "search_results.csv", index=False)

    df = pd.DataFrame(results)
    print(f"\n{'='*70}")
    print("  超参数搜索结果汇总（按 val_oa 降序）")
    print(f"{'='*70}")
    print(df.sort_values("val_oa", ascending=False).to_string(index=False))

    # 简单敏感性分析：固定其余三个维度取搜索中位数附近的组，看单个维度的边际影响
    print(f"\n{'='*70}")
    print("  敏感性分析（每个超参数取值 → 平均 val_oa，用全部 trial 汇总）")
    print(f"{'='*70}")
    for param in ["lr", "weight_decay", "dropout", "batch_size"]:
        grouped = df.groupby(param)["val_oa"].agg(["mean", "std", "count"])
        print(f"\n-- {param} --")
        print(grouped.to_string())

    print(f"\n结果已保存到：{OUT_DIR / 'search_results.csv'}")
    print("Hyperparameter-search outputs are ready for inspection.")


if __name__ == "__main__":
    main()
