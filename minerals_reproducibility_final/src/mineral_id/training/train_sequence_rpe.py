"""
exp04: 8-frame sequence model WITH Rotational Position Encoding (RPE)
Rotational positional-encoding training entry point.
运行方式：python -m mineral_id.training.train_sequence_rpe
"""

import math, os, json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.models import ResNet50_Weights

# 复用已有的 SequenceDataset（支持 df= 参数）
from mineral_id.data.dataset_sequence import SequenceDataset
from mineral_id.data.transforms import get_train_transform, get_eval_transform

# ── 超参数（与 exp03b 保持一致，便于公平对比）────────────────────────────────
CFG = dict(
    csv_train  = "data/processed/splits/train.csv",
    csv_val    = "data/processed/splits/val.csv",
    csv_test   = "data/processed/splits/test.csv",
    num_classes= 3,
    embed_dim  = 2048,
    num_heads  = 8,
    dropout    = 0.5,           # 同 exp03b
    lr         = 5e-5,          # 同 exp03b
    weight_decay= 1e-4,
    batch_size = 8,
    max_epochs = 50,
    patience   = 5,
    seed       = 42,
    freeze_layers = ["conv1","bn1","layer1","layer2"],  # 同 exp03b
    save_dir   = "experiments/exp04_rpe",
    theta_max  = 90.0,          # RPE 角度归一化分母
    # photo_id -> 对应的 XPL 旋转角度（PPL用None）
    # photo_id: 1=PPL, 2=XPL@0, 3=XPL@15, 4=XPL@30, 5=XPL@45, 6=XPL@60, 7=XPL@75, 8=XPL@90
    photo_angles = {1: None, 2: 0.0, 3: 15.0, 4: 30.0,
                    5: 45.0, 6: 60.0, 7: 75.0, 8: 90.0},
)

# ── 工具函数 ─────────────────────────────────────────────────────────────────
def set_seed(seed):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("CPU mode")
    return dev

# ── RPE 模块 ─────────────────────────────────────────────────────────────────
class RotationalPositionEncoding(nn.Module):
    """
    Rotational Position Encoding for polarized microscopy extinction sequences.

    For XPL frames: uses sin/cos encoding based on physical rotation angle θ.
        PE(θ, 2k)   = sin(θ / θ_max * π)
        PE(θ, 2k+1) = cos(θ / θ_max * π)
        k = 0,..., embed_dim//2 - 1

    For PPL frame: uses a learnable embedding vector to distinguish
    the different imaging modality (morphology vs. interference optics).
    """
    def __init__(self, embed_dim: int, theta_max: float = 90.0):
        super().__init__()
        self.embed_dim  = embed_dim
        self.theta_max  = theta_max
        # 可学习的 PPL 嵌入
        self.ppl_embed  = nn.Parameter(torch.randn(embed_dim) * 0.02)
        # XPL 编码（确定性，不可学习）—— 在 forward 中动态生成

    def _xpl_encoding(self, theta_deg: float) -> torch.Tensor:
        """生成 2048 维的正弦-余弦角度编码"""
        theta_rad = (theta_deg / self.theta_max) * math.pi
        k         = torch.arange(self.embed_dim // 2, dtype=torch.float32)
        # 所有维度使用相同角度（这里 k 只是索引，不做频率缩放——与 NLP 不同）
        pe = torch.zeros(self.embed_dim)
        pe[0::2] = math.sin(theta_rad)   # even dims: sin
        pe[1::2] = math.cos(theta_rad)   # odd  dims: cos
        return pe

    def forward(self, frame_features: torch.Tensor,
                photo_ids: list[int]) -> torch.Tensor:
        """
        Args:
            frame_features: [B, T, embed_dim]
            photo_ids:       list of T photo_id values (1..8)
        Returns:
            Z: [B, T, embed_dim]  (features + position encoding)
        """
        B, T, D = frame_features.shape
        device  = frame_features.device

        pe_matrix = torch.zeros(T, D, device=device)  # [T, D]
        for i, pid in enumerate(photo_ids):
            angle = CFG["photo_angles"][pid]
            if angle is None:
                # PPL frame: learnable embedding
                pe_matrix[i] = self.ppl_embed.to(device)
            else:
                # XPL frame: deterministic sin/cos encoding
                pe_matrix[i] = self._xpl_encoding(angle).to(device)

        # broadcast: [T, D] -> [B, T, D]
        return frame_features + pe_matrix.unsqueeze(0)


# ── 主模型 ───────────────────────────────────────────────────────────────────
class SequenceRPEModel(nn.Module):
    """
    8-frame sequence model with Rotational Position Encoding.
    Architecture:
        Shared ResNet-50 (pretrained) → GAP → [B, T, 2048]
        + RPE (angle-aware position encoding)
        → MultiheadAttention(8 heads, 2048 dim)
        → LayerNorm → mean pool → Dropout → Linear(3)
    """
    def __init__(self, cfg: dict):
        super().__init__()
        D    = cfg["embed_dim"]
        nH   = cfg["num_heads"]
        nC   = cfg["num_classes"]
        drop = cfg["dropout"]

        # Shared backbone (pretrained ResNet-50, remove FC)
        backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # → [B,2048,1,1]

        # Freeze early layers (同 exp03b)
        for name, param in backbone.named_parameters():
            if any(name.startswith(l) for l in cfg["freeze_layers"]):
                param.requires_grad = False

        # RPE module
        self.rpe = RotationalPositionEncoding(D, cfg["theta_max"])

        # Temporal attention
        self.attn = nn.MultiheadAttention(D, nH, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(D)

        # Classifier
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(D, nC))

    def forward(self, x: torch.Tensor, photo_ids: list[int]) -> torch.Tensor:
        """
        Args:
            x: [B, T, 3, H, W]
            photo_ids: list of T int
        Returns:
            logits: [B, num_classes]
        """
        B, T, C, H, W = x.shape
        # Extract per-frame features via shared backbone
        x_flat  = x.reshape(B * T, C, H, W)                        # [B*T, 3, H, W]
        feats   = self.backbone(x_flat).squeeze(-1).squeeze(-1)     # [B*T, 2048]
        feats   = feats.reshape(B, T, -1)                           # [B, T, 2048]

        # Add rotational position encoding
        Z = self.rpe(feats, photo_ids)                              # [B, T, 2048]

        # Self-attention
        attn_out, _ = self.attn(Z, Z, Z)                           # [B, T, 2048]
        Z = self.norm(Z + attn_out)                                 # residual + norm

        # Pool over frame dimension
        slide_feat = Z.mean(dim=1)                                  # [B, 2048]

        return self.head(slide_feat)                                # [B, 3]


# ── 训练主程序 ───────────────────────────────────────────────────────────────
def main():
    set_seed(CFG["seed"])
    device = get_device()
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32

    save_dir = Path(CFG["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    train_ds = SequenceDataset(csv_path=CFG["csv_train"], mode="train",
                               transform=get_train_transform())
    val_ds   = SequenceDataset(csv_path=CFG["csv_val"],   mode="val",
                               transform=get_eval_transform())

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"],
                              shuffle=False, num_workers=4, pin_memory=True)

    # photo_id sequence for 8-frame (must match SequenceDataset output order)
    # SequenceDataset returns frames in order photo_id=1..8
    PHOTO_IDS = [1, 2, 3, 4, 5, 6, 7, 8]

    # Model
    model = SequenceRPEModel(CFG).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params/1e6:.1f}M  |  Trainable: {trainable/1e6:.1f}M")

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG["lr"], weight_decay=CFG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["max_epochs"]
    )
    criterion = nn.CrossEntropyLoss()
    scaler    = torch.amp.GradScaler(enabled=(device.type=="cuda"))

    best_val_acc = 0.0
    no_improve   = 0
    history      = []

    print(f"\n{'Ep':>4} {'TrainLoss':>10} {'TrainAcc':>10} {'ValAcc':>10}")
    print("─" * 42)

    for epoch in range(1, CFG["max_epochs"] + 1):
        # ── train ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch in train_loader:
            imgs, labels = batch
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, dtype=dtype):
                logits = model(imgs, PHOTO_IDS)
                loss   = criterion(logits, labels)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            train_loss    += loss.item() * labels.size(0)
            preds          = logits.argmax(1)
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)
        scheduler.step()

        train_acc = train_correct / train_total * 100

        # ── validate ──
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                imgs, labels = batch
                imgs   = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, dtype=dtype):
                    logits = model(imgs, PHOTO_IDS)
                preds       = logits.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)
        val_acc = val_correct / val_total * 100

        avg_loss = train_loss / train_total
        print(f"{epoch:>4d}  {avg_loss:>10.4f}  {train_acc:>9.2f}%  {val_acc:>9.2f}%")

        history.append({"epoch":epoch, "train_loss":avg_loss,
                        "train_acc":train_acc, "val_acc":val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_dir / "best.pth")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= CFG["patience"]:
                print(f"\nEarly stopping at epoch {epoch}. Best val acc: {best_val_acc:.2f}%")
                break

    # Save training log
    with open(save_dir / "train_log.json", "w") as f:
        json.dump({"config": CFG, "history": history,
                   "best_val_acc": best_val_acc}, f, indent=2)
    print(f"\nBest val acc: {best_val_acc:.2f}%")
    print(f"Checkpoint saved to: {save_dir / 'best.pth'}")
    print("Next: run evaluate.py --checkpoint experiments/exp04_rpe/best.pth --seq")


if __name__ == "__main__":
    main()
