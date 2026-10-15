r"""
Attention 权重可视化 — exp03 / exp03b / exp04（8 帧消光序列 Temporal Attention 模型）

Attention-weight visualization across the 8-frame sequence, including per-frame
and aggregate attention summaries across the test set.

做法：nn.MultiheadAttention 默认 forward 不返回权重矩阵，本脚本手动重放一遍
forward（backbone 部分权重照常加载、不改），额外传 need_weights=True,
average_attn_weights=False 拿到 [B, num_heads, 8, 8] 的完整权重矩阵，然后：
  1. 对 heads 取平均 -> [8, 8] 的"帧对帧"注意力矩阵，画热力图
  2. 对"谁被关注"这个维度（矩阵的列）取平均 -> 长度为 8 的向量，表示每一帧
     （PPL / XPL@0°…90°）平均获得了多少注意力权重，画成条形图——这个数字直接
     用于比较不同旋转角度帧获得的平均注意力权重
  3. 除了每个展示样例单独画一张，还会把测试集里全部样例的"每帧平均注意力"汇总
     一遍，给出一个总体结论（比单个样例更有说服力，可以直接写进 Discussion）

exp03 / exp03b 用的都是 mineral_id.models.TemporalAttentionModel（结构一致，
只有 dropout/是否冻结层的区别，这些在 eval 模式下不影响推理结果），checkpoint
是 dict 包装格式（跟 evaluate.py 用的一样）。exp04 用的是 train_sequence_rpe.py
里定义的 SequenceRPEModel（多了旋转位置编码 RPE），checkpoint 是裸 state_dict
格式——脚本里已经按各自的真实格式处理，不需要手动改。

用法（PowerShell，项目根目录下，先激活 venv）：
    .\.venv\Scripts\python.exe scripts/attention_viz.py --exp exp03
    .\.venv\Scripts\python.exe scripts/attention_viz.py --exp exp03b
    .\.venv\Scripts\python.exe scripts/attention_viz.py --exp exp04

输出：
    experiments/attn_viz_{exp}/example_<rockclass>_<id>_<slideid>.png   （单样例热力图+条形图）
    experiments/attn_viz_{exp}/aggregate_attention.png                  （全测试集平均，最重要的一张）
    experiments/attn_viz_{exp}/aggregate_attention.csv                  （对应的数字，方便写进论文）

预计耗时：几十秒到一两分钟（33 个测试薄片，只做前向）。
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mineral_id.models import TemporalAttentionModel
from mineral_id.data.transforms import get_eval_transform

CLASS_TO_IDX = {"igneous": 0, "metamorphic": 1, "sedimentary": 2}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
CLASS_NAMES_ZH = {"igneous": "火成岩", "metamorphic": "变质岩", "sedimentary": "沉积岩"}

FRAME_LABELS = ["PPL", "XPL@0°", "XPL@15°", "XPL@30°", "XPL@45°",
                "XPL@60°", "XPL@75°", "XPL@90°"]
PHOTO_IDS = [1, 2, 3, 4, 5, 6, 7, 8]


# ── exp04 专用：SequenceRPEModel（原样照抄 train_sequence_rpe.py 的定义，
#    因为这个类没有放进 mineral_id.models 包，只存在于训练脚本里）──────────────
RPE_CFG = dict(
    embed_dim=2048, num_heads=8, dropout=0.5, theta_max=90.0,
    freeze_layers=["conv1", "bn1", "layer1", "layer2"],
    photo_angles={1: None, 2: 0.0, 3: 15.0, 4: 30.0, 5: 45.0, 6: 60.0, 7: 75.0, 8: 90.0},
)


class RotationalPositionEncoding(nn.Module):
    def __init__(self, embed_dim, theta_max=90.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.theta_max = theta_max
        self.ppl_embed = nn.Parameter(torch.randn(embed_dim) * 0.02)

    def _xpl_encoding(self, theta_deg):
        theta_rad = (theta_deg / self.theta_max) * math.pi
        pe = torch.zeros(self.embed_dim)
        pe[0::2] = math.sin(theta_rad)
        pe[1::2] = math.cos(theta_rad)
        return pe

    def forward(self, frame_features, photo_ids):
        B, T, D = frame_features.shape
        device = frame_features.device
        pe_matrix = torch.zeros(T, D, device=device)
        for i, pid in enumerate(photo_ids):
            angle = RPE_CFG["photo_angles"][pid]
            pe_matrix[i] = (self.ppl_embed.to(device) if angle is None
                             else self._xpl_encoding(angle).to(device))
        return frame_features + pe_matrix.unsqueeze(0)


class SequenceRPEModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        import torchvision.models as models
        D, nH, drop = cfg["embed_dim"], cfg["num_heads"], cfg["dropout"]
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.rpe = RotationalPositionEncoding(D, cfg["theta_max"])
        self.attn = nn.MultiheadAttention(D, nH, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(D)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(D, 3))

    def forward_with_attention(self, x, photo_ids):
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)
        feats = self.backbone(x_flat).squeeze(-1).squeeze(-1)
        feats = feats.reshape(B, T, -1)
        Z = self.rpe(feats, photo_ids)
        attn_out, attn_weights = self.attn(Z, Z, Z, need_weights=True,
                                            average_attn_weights=False)
        Z2 = self.norm(Z + attn_out)
        pooled = Z2.mean(dim=1)
        logits = self.head(pooled)
        return logits, attn_weights  # attn_weights: [B, num_heads, T, T]


# ── exp03 / exp03b 专用：直接复用 mineral_id.models.TemporalAttentionModel，
#    但要手动重放一遍 forward 才能拿到 attention 权重 ─────────────────────────
def forward_with_attention_temporal(model: TemporalAttentionModel, x: torch.Tensor):
    B, T, C, H, W = x.shape
    x = x.view(B * T, C, H, W)
    feats = model.backbone(x)
    feats = feats.view(B, T, -1)
    attn_out, attn_weights = model.temporal_attn(
        feats, feats, feats, need_weights=True, average_attn_weights=False
    )
    pooled = attn_out.mean(dim=1)
    logits = model.classifier(pooled)
    return logits, attn_weights  # [B, num_heads, T, T]


def load_model_and_forward_fn(exp: str, device: torch.device):
    """返回 (model, forward_fn)，forward_fn(x, photo_ids) -> (logits, attn_weights[B,heads,T,T])"""
    if exp in ("exp03", "exp03b"):
        ckpt_name = "exp03_sequence" if exp == "exp03" else "exp03b_frozen"
        ckpt_path = ROOT / "experiments" / ckpt_name / "best.pth"
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = TemporalAttentionModel().to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        def forward_fn(x, photo_ids):
            return forward_with_attention_temporal(model, x)

        return model, forward_fn

    elif exp == "exp04":
        ckpt_path = ROOT / "experiments" / "exp04_rpe" / "best.pth"
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = SequenceRPEModel(RPE_CFG).to(device)
        model.load_state_dict(state_dict)  # 裸 state_dict，无外层 dict 包装
        model.eval()

        def forward_fn(x, photo_ids):
            return model.forward_with_attention(x, photo_ids)

        return model, forward_fn

    else:
        raise ValueError(f"未知 exp: {exp}")


def load_sequence(group: pd.DataFrame, transform) -> torch.Tensor:
    """group：某个薄片的全部 photo_id=1..8 行。返回 [1, 8, 3, 224, 224]。"""
    ordered = group[group["photo_id"].isin(PHOTO_IDS)].sort_values("photo_id")
    frames = []
    for _, row in ordered.iterrows():
        img = Image.open(row["filepath"]).convert("RGB")
        frames.append(transform(img))
    seq = torch.stack(frames, dim=0)  # [8, 3, H, W]
    return seq.unsqueeze(0)  # [1, 8, 3, H, W]


def plot_single_example(attn_matrix: np.ndarray, title: str, out_path: Path):
    """attn_matrix: [8,8]（已对 heads 取平均）"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    im = axes[0].imshow(attn_matrix, cmap="viridis", vmin=0)
    axes[0].set_xticks(range(8)); axes[0].set_xticklabels(FRAME_LABELS, rotation=45, ha="right")
    axes[0].set_yticks(range(8)); axes[0].set_yticklabels(FRAME_LABELS)
    axes[0].set_xlabel("被关注的帧（key）"); axes[0].set_ylabel("发起关注的帧（query）")
    axes[0].set_title("帧间注意力矩阵（8头平均）")
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    received = attn_matrix.mean(axis=0)  # 每一列（key帧）平均获得多少关注
    axes[1].bar(range(8), received, color="#4C72B0")
    axes[1].set_xticks(range(8)); axes[1].set_xticklabels(FRAME_LABELS, rotation=45, ha="right")
    axes[1].set_ylabel("平均获得的注意力权重")
    axes[1].set_title("各帧平均被关注程度")
    axes[1].axhline(1/8, color="gray", linestyle="--", linewidth=1, label="均匀分布基线(1/8)")
    axes[1].legend(fontsize=8)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["exp03", "exp03b", "exp04"], required=True)
    parser.add_argument("--splits-dir", default="data/processed/splits")
    parser.add_argument("--n-examples", type=int, default=6,
                         help="单独画图的样例数量（另外会对全部测试集做汇总统计）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  exp: {args.exp}")

    model, forward_fn = load_model_and_forward_fn(args.exp, device)
    transform = get_eval_transform()

    splits_dir = Path(args.splits_dir)
    test_df = pd.read_csv(splits_dir / "test.csv")

    out_dir = ROOT / "experiments" / f"attn_viz_{args.exp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_received = []  # 汇总全部测试薄片的"每帧平均注意力"，用来算总体结论
    all_meta = []
    n_plotted = 0

    groups = list(test_df.groupby(["rock_class", "rock_id", "slide_id"]))
    print(f"test 集共 {len(groups)} 个薄片，逐个跑推理并提取 attention...")

    for (rock_class, rock_id, slide_id), group in groups:
        present = set(group["photo_id"].tolist())
        if not all(pid in present for pid in PHOTO_IDS):
            print(f"跳过 {rock_class}_{rock_id}_{slide_id}：8帧不全")
            continue

        seq = load_sequence(group, transform).to(device)

        with torch.no_grad():
            logits, attn_weights = forward_fn(seq, PHOTO_IDS)
            pred_idx = logits.argmax(1).item()
            conf = torch.softmax(logits, dim=1)[0, pred_idx].item()

        attn_matrix = attn_weights[0].mean(dim=0).cpu().numpy()  # 对 heads 取平均 -> [8,8]
        received = attn_matrix.mean(axis=0)  # [8]，每帧平均获得的关注

        all_received.append(received)
        pred_class = IDX_TO_CLASS[pred_idx]
        is_correct = (pred_class == rock_class)
        all_meta.append({
            "rock_class": rock_class, "rock_id": rock_id, "slide_id": slide_id,
            "pred_class": pred_class, "conf": conf, "correct": is_correct,
        })

        # 只挑前 n_examples 个（且优先保留各类别都有代表）单独画图，避免刷屏
        if n_plotted < args.n_examples:
            true_zh = CLASS_NAMES_ZH[rock_class]
            pred_zh = CLASS_NAMES_ZH[pred_class]
            tag_zh = "预测正确" if is_correct else "预测错误"
            title = (f"{rock_class}_{rock_id}_{slide_id}  真实={true_zh}  预测={pred_zh} "
                      f"(置信度{conf:.2f})  [{tag_zh}]  —— {args.exp}")
            out_path = out_dir / f"example_{rock_class}_{rock_id}_{slide_id}.png"
            plot_single_example(attn_matrix, title, out_path)
            print(f"已保存单样例图：{out_path}")
            n_plotted += 1

    # ── 汇总：全部测试薄片的"每帧平均获得注意力"取平均 ──
    all_received = np.array(all_received)  # [N, 8]
    mean_received = all_received.mean(axis=0)
    std_received = all_received.std(axis=0)

    summary_df = pd.DataFrame({
        "frame": FRAME_LABELS,
        "mean_attention_received": mean_received,
        "std_attention_received": std_received,
    })
    summary_df.to_csv(out_dir / "aggregate_attention.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.bar(range(8), mean_received, yerr=std_received, capsize=3, color="#55A868")
    ax.set_xticks(range(8)); ax.set_xticklabels(FRAME_LABELS, rotation=45, ha="right")
    ax.set_ylabel("平均获得的注意力权重（全部测试薄片）")
    ax.axhline(1/8, color="gray", linestyle="--", linewidth=1, label="均匀分布基线(1/8)")
    ax.set_title(f"{args.exp}：各旋转角度帧的平均注意力权重（N={len(all_received)} 个测试薄片）")
    ax.legend(fontsize=8)
    plt.tight_layout()
    agg_path = out_dir / "aggregate_attention.png"
    plt.savefig(agg_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n{'='*70}")
    print(f"  {args.exp}  全测试集平均注意力权重")
    print(f"{'='*70}")
    print(summary_df.to_string(index=False))
    print(f"\n汇总图：{agg_path}")
    print(f"汇总数据：{out_dir / 'aggregate_attention.csv'}")
    print(f"单样例图共 {n_plotted} 张，保存在：{out_dir}")
    print("\nAttention summaries saved.")


if __name__ == "__main__":
    main()
