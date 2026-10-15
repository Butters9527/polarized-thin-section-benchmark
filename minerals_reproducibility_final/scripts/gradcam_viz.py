r"""
Grad-CAM 可视化 — exp01（单偏光基线）/ exp02（双模态早期融合）

Grad-CAM visualization for checking spatial attribution in the single-modal and dual-modal models.

做法：标准 Grad-CAM（Selvaraju et al. 2017，论文参考文献[14]），hook 在 ResNet-50
最后一个卷积块 layer4 上。exp02 是 6 通道输入（PPL 3ch + XPL 3ch 拼接），层4的特征图
本身就是两个模态融合之后的产物，没法把"这块热力图来自 PPL 还是 XPL"精确拆开——
所以脚本对每张图只算一次 CAM，然后同时叠加显示在 PPL 和 XPL 原图上（左右并排），方便
直接判断模型关注的空间位置是否落在地质学上有意义的颗粒/组构上（比如云母定向排列、
个别高突起矿物颗粒等）。

选图逻辑：默认从 test.csv 每个真实类别里，各挑 1 张预测正确的 + 1 张预测错误的（如果
该类别有错分样本的话），一共最多 6 张，兼顾"模型关注对不对"和"模型出错时在看哪里"两种
诊断价值。也可以用 --slide-keys 手动指定要看哪几个薄片。

用法（PowerShell，项目根目录下，先激活 venv）：
    .\.venv\Scripts\python.exe scripts/gradcam_viz.py --exp exp02
    .\.venv\Scripts\python.exe scripts/gradcam_viz.py --exp exp01

输出：
    experiments/gradcam_{exp}/gradcam_<rockclass>_<rockid>_<slideid>_<correct|wrong>.png
    每张图：原图 + CAM 热力图叠加图，并排显示，标题写明真实类/预测类/置信度

预计耗时：几十秒（只做前向+一次反向，不训练）。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mineral_id.data.transforms import get_eval_transform

CLASS_TO_IDX = {"igneous": 0, "metamorphic": 1, "sedimentary": 2}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
CLASS_NAMES_ZH = {"igneous": "火成岩", "metamorphic": "变质岩", "sedimentary": "沉积岩"}

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD = np.array([0.229, 0.224, 0.225])


# ── 模型构建（与训练脚本完全一致）───────────────────────────────────────────
def build_model_exp01() -> nn.Module:
    import torchvision.models as models
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    return model


def build_model_exp02() -> nn.Module:
    import torchvision.models as models
    base = models.resnet50(weights=None)
    base.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
    base.fc = nn.Linear(base.fc.in_features, 3)
    return base


# ── Grad-CAM 核心：hook layer4，算加权特征图 ─────────────────────────────────
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """input_tensor: [1, C, H, W]。返回 [H, W] 的归一化热力图（0-1）。"""
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()

        # 全局平均池化梯度 -> 每个通道的权重
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, h, w]
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear",
                             align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 1e-8:
            cam = cam / cam.max()
        return cam, torch.softmax(output, dim=1)[0].detach().cpu().numpy()


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """[3,H,W] 归一化张量 -> [H,W,3] uint8 图像"""
    arr = tensor.permute(1, 2, 0).cpu().numpy()
    arr = arr * _IMAGENET_STD + _IMAGENET_MEAN
    arr = np.clip(arr, 0, 1)
    return (arr * 255).astype(np.uint8)


def overlay_cam(image_uint8: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heatmap = cm.jet(cam)[:, :, :3]  # RGBA -> RGB, 0-1
    heatmap = (heatmap * 255).astype(np.uint8)
    blended = (image_uint8.astype(np.float32) * (1 - alpha)
               + heatmap.astype(np.float32) * alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def pick_examples(test_csv: Path, checkpoint_preds: dict, max_per_class: int = 2) -> list[dict]:
    """从 test.csv 里为每个类别挑 1 张预测正确 + 1 张预测错误（若存在）的薄片。"""
    df = pd.read_csv(test_csv)
    slides = df.groupby(["rock_class", "rock_id", "slide_id"]).first().reset_index()

    picked = []
    for rock_class in CLASS_TO_IDX:
        subset = slides[slides["rock_class"] == rock_class]
        correct_done, wrong_done = False, False
        for _, row in subset.iterrows():
            key = (row["rock_class"], row["rock_id"], row["slide_id"])
            if key not in checkpoint_preds:
                continue
            pred_class, conf = checkpoint_preds[key]
            is_correct = (pred_class == rock_class)
            if is_correct and not correct_done:
                picked.append({**row.to_dict(), "pred_class": pred_class,
                               "conf": conf, "tag": "correct"})
                correct_done = True
            elif not is_correct and not wrong_done:
                picked.append({**row.to_dict(), "pred_class": pred_class,
                               "conf": conf, "tag": "wrong"})
                wrong_done = True
            if correct_done and wrong_done:
                break
    return picked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["exp01", "exp02"], required=True)
    parser.add_argument("--checkpoint", default=None,
                         help="默认 experiments/{exp01_baseline|exp02_dual_stream}/best.pth")
    parser.add_argument("--splits-dir", default="data/processed/splits")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint) if args.checkpoint else (
        ROOT / "experiments" / ("exp01_baseline" if args.exp == "exp01" else "exp02_dual_stream")
        / "best.pth"
    )
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = (build_model_exp01() if args.exp == "exp01" else build_model_exp02()).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    gradcam = GradCAM(model, target_layer=model.layer4[-1])
    transform = get_eval_transform()

    splits_dir = Path(args.splits_dir)
    test_df = pd.read_csv(splits_dir / "test.csv")

    # ── 先跑一遍完整推理，拿到每个薄片的预测类别，用来挑"对/错各一张"的样例 ──
    print("先做一遍完整推理，用于挑选展示样例...")
    checkpoint_preds = {}
    for (rock_class, rock_id, slide_id), group in test_df.groupby(
        ["rock_class", "rock_id", "slide_id"]
    ):
        if args.exp == "exp01":
            row = group[group["photo_id"] == 1]
            if row.empty:
                continue
            img = Image.open(row.iloc[0]["filepath"]).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
        else:
            ppl_row = group[group["photo_id"] == 1]
            xpl_row = group[group["photo_id"] == 2]
            if ppl_row.empty or xpl_row.empty:
                continue
            ppl = transform(Image.open(ppl_row.iloc[0]["filepath"]).convert("RGB"))
            xpl = transform(Image.open(xpl_row.iloc[0]["filepath"]).convert("RGB"))
            x = torch.cat([ppl, xpl], dim=0).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(x)
            pred_idx = out.argmax(1).item()
            conf = torch.softmax(out, dim=1)[0, pred_idx].item()
        checkpoint_preds[(rock_class, rock_id, slide_id)] = (IDX_TO_CLASS[pred_idx], conf)

    examples = pick_examples(splits_dir / "test.csv", checkpoint_preds)
    print(f"选出 {len(examples)} 个展示样例。")

    out_dir = ROOT / "experiments" / f"gradcam_{args.exp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ex in examples:
        rock_class, rock_id, slide_id = ex["rock_class"], ex["rock_id"], ex["slide_id"]
        group = test_df[
            (test_df["rock_class"] == rock_class)
            & (test_df["rock_id"] == rock_id)
            & (test_df["slide_id"] == slide_id)
        ]
        ppl_row = group[group["photo_id"] == 1].iloc[0]
        ppl_img = Image.open(ppl_row["filepath"]).convert("RGB")
        ppl_tensor = transform(ppl_img)

        if args.exp == "exp01":
            x_for_hook = ppl_tensor.unsqueeze(0).to(device)
        else:
            xpl_row = group[group["photo_id"] == 2].iloc[0]
            xpl_img = Image.open(xpl_row["filepath"]).convert("RGB")
            xpl_tensor = transform(xpl_img)
            x_for_hook = torch.cat([ppl_tensor, xpl_tensor], dim=0).unsqueeze(0).to(device)

        target_idx = CLASS_TO_IDX[ex["pred_class"]]  # 对模型自己预测出的类别做 CAM
        cam, probs = gradcam.generate(x_for_hook, target_idx)

        ppl_uint8 = denormalize(ppl_tensor)
        ppl_overlay = overlay_cam(ppl_uint8, cam)

        if args.exp == "exp02":
            xpl_uint8 = denormalize(xpl_tensor)
            xpl_overlay = overlay_cam(xpl_uint8, cam)
            fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
            axes[0].imshow(ppl_uint8); axes[0].set_title("PPL 原图"); axes[0].axis("off")
            axes[1].imshow(ppl_overlay); axes[1].set_title("PPL + Grad-CAM"); axes[1].axis("off")
            axes[2].imshow(xpl_uint8); axes[2].set_title("XPL 原图"); axes[2].axis("off")
            axes[3].imshow(xpl_overlay); axes[3].set_title("XPL + Grad-CAM"); axes[3].axis("off")
        else:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
            axes[0].imshow(ppl_uint8); axes[0].set_title("PPL 原图"); axes[0].axis("off")
            axes[1].imshow(ppl_overlay); axes[1].set_title("PPL + Grad-CAM"); axes[1].axis("off")

        true_zh = CLASS_NAMES_ZH[rock_class]
        pred_zh = CLASS_NAMES_ZH[ex["pred_class"]]
        tag_zh = "预测正确" if ex["tag"] == "correct" else "预测错误"
        fig.suptitle(
            f"{rock_class}_{rock_id}_{slide_id}  |  真实={true_zh}  预测={pred_zh} "
            f"(置信度{ex['conf']:.2f})  [{tag_zh}]",
            fontsize=11,
        )
        plt.tight_layout()

        out_path = out_dir / f"gradcam_{rock_class}_{rock_id}_{slide_id}_{ex['tag']}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"已保存：{out_path}")

    print(f"\n全部完成，{len(examples)} 张图保存在：{out_dir}")
    print("Grad-CAM outputs saved.")


if __name__ == "__main__":
    main()
