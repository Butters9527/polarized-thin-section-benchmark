"""TemporalAttentionModel：ResNet-50 骨干 + 帧间自注意力，用于消光序列分类。"""

import torch
import torch.nn as nn
from torchvision import models


class TemporalAttentionModel(nn.Module):
    """8 帧消光序列分类模型。

    架构：
        1. 共享 ResNet-50 backbone（去掉 fc，保留 avgpool）逐帧提取特征
           每帧 [B, 3, 224, 224] → [B, 2048]
        2. 8 帧 stack → [B, 8, 2048]
        3. MultiheadAttention 自注意力（Q=K=V）→ [B, 8, 2048]
        4. mean(dim=1) → [B, 2048]
        5. LayerNorm(2048) → Dropout(0.5) → Linear(2048, 3) → [B, 3]

    backbone 使用 ResNet50_Weights.IMAGENET1K_V1 预训练，全参数可训练。
    attention 与分类头随机初始化。
    """

    def __init__(self, dropout: float = 0.5) -> None:
        super().__init__()

        # ── backbone：ResNet-50 去掉 fc ──────────────────────────────
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # 保留 conv1 → bn1 → relu → maxpool → layer1-4 → avgpool
        # 去掉 fc；用 Identity 占位不改变 forward 路径
        resnet.fc = nn.Identity()
        self.backbone = resnet          # 输出 [B, 2048]（经 avgpool flatten）

        # ── 帧间自注意力 ─────────────────────────────────────────────
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=2048,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )

        # ── 分类头 ───────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(2048),
            nn.Dropout(dropout),
            nn.Linear(2048, 3),
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 8, 3, 224, 224]
        Returns:
            logits: [B, 3]
        """
        B, T, C, H, W = x.shape  # T == 8

        # 逐帧过 backbone：合并 B×T 批次一次前向，避免 Python 循环
        x = x.view(B * T, C, H, W)           # [B*8, 3, 224, 224]
        feats = self.backbone(x)              # [B*8, 2048]
        feats = feats.view(B, T, -1)          # [B, 8, 2048]

        # 帧间自注意力（Q=K=V）
        attn_out, _ = self.temporal_attn(feats, feats, feats)  # [B, 8, 2048]

        # 时序聚合
        pooled = attn_out.mean(dim=1)         # [B, 2048]

        return self.classifier(pooled)        # [B, 3]


# ----------------------------------------------------------------------
if __name__ == "__main__":
    model = TemporalAttentionModel()
    model.eval()

    dummy = torch.zeros(2, 8, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)

    print(f"输入 shape：{dummy.shape}")
    print(f"输出 shape：{out.shape}")
    assert out.shape == (2, 3), f"shape 异常：{out.shape}"
    print("smoke test PASSED")
