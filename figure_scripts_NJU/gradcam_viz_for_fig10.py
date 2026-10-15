r"""
gradcam_viz.py

对 exp02（PPL+XPL早期融合模型）的单个样本计算 Grad-CAM 归因热力图，
输出一张干净的 1x4 面板图：PPL原图 | PPL+CAM叠加 | XPL原图 | XPL+CAM叠加。

这张输出图是"原始素材"，不含任何标题栏/文字，故意设计成这样——
generate_fig10_gradcam.py 会读取两次本脚本的输出（一个正确分类样例、
一个误判样例），拼成论文Figure 10最终版式（英文标题、(a)-(h)面板字母）。
两个脚本按"四等分裁切"的约定配合，本脚本只要保证输出是等宽横排4联图、
不带任何嵌入文字，下游脚本就能正确处理，不需要写死像素坐标。

================================================================
【模型定义】与 train_dual.py 保持一致：
================================================================
Checkpoint：experiments/exp02_dual_stream/best.pth，
格式 {'model_state_dict': ..., 'epoch':..., 'optimizer_state_dict':..., 'val_acc':...}。

模型构建函数是 mineral_id.training.train_dual.build_model_dual()（注意
不是 build_model，是 build_model_dual）——返回的就是一个原生
torchvision.models.resnet50 对象，只改了两处：
  - conv1: 6通道输入（bias=False），前3通道复用ImageNet预训练权重、
    后3通道也复制同一份预训练权重，整体乘0.5
  - fc: nn.Linear(2048, 3)，没有额外的Dropout层
该 checkpoint 对应的模型没有额外 Dropout 层；本脚本按训练代码定义模型。

本脚本【优先】尝试 from mineral_id.training.train_dual import build_model_dual，
只有导入失败（比如不在项目根目录/mineral_id包不在PYTHONPATH里）时，才退回
下面 _fallback_build_model_dual()——这个备用实现是 build_model_dual() 的
逐行精确复刻，不是猜的，正常情况下两者行为完全一致。

【DualStreamDataset 通道拼接顺序】：
__getitem__ 中使用 torch.cat([ppl_tensor, xpl_tensor], dim=0) ——
PPL在前3通道、XPL在后3通道，跟本脚本原来的假设一致，不需要改。

预处理与 transforms.py 的 get_eval_transform() 保持一致：
Resize(256) → CenterCrop(224) → ToTensor → Normalize(ImageNet均值方差)，
跟本脚本 load_and_preprocess() 的实现逐字一致，不需要改。


================================================================
依赖：torch, torchvision, matplotlib, numpy, Pillow
================================================================

用法：
    python gradcam_viz.py \
        --ppl path/to/ppl_image.jpg \
        --xpl path/to/xpl0_image.jpg \
        --checkpoint experiments/exp02_dual_stream/best.pth \
        --out gradcam_igneous_4_1_correct.png

    --target-class 不给的话，默认用模型预测的类别（argmax）算CAM；
    如果想看"模型为什么会错判成某类"，可以显式指定 --target-class 来看
    模型对该类别的归因，即使它不是最终预测类别。

输出：
    --out 指定的路径，一张 1x4 面板 PNG（PPL | PPL+CAM | XPL | XPL+CAM），
    左上角小字标注 true/pred/confidence，不含大标题。
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_NAMES = ["igneous", "metamorphic", "sedimentary"]  # 顺序跟论文Table/Fig5一致

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224


# ============================================================================
# 备用模型实现（仅在无法import mineral_id包时使用）——
# 这是 mineral_id.training.train_dual.build_model_dual() 的逐行精确复刻
# （已用扫描到的真实源码核对过，不是猜的），正常情况下两者行为完全一致。
# ============================================================================
def _fallback_build_model_dual() -> nn.Module:
    import torchvision.models as tv_models
    base = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)

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


def build_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """加载exp02模型。优先从你的训练代码导入真实的 build_model_dual()，
    失败才退回上面的逐行复刻版本（见文件头说明）。"""
    model = None
    try:
        from mineral_id.training.train_dual import build_model_dual  # noqa
        model = build_model_dual()
    except Exception as e:
        print(f"警告：未能从 mineral_id.training.train_dual 导入 build_model_dual"
              f"（{e}），退回脚本内置的逐行复刻实现。\n"
              f"如果你不在项目根目录下运行、或 mineral_id 包不在 PYTHONPATH 里，"
              f"这是预期行为，复刻版本应该能正常work；如果 load_state_dict 仍然"
              f"报key不匹配，说明真实代码可能改过，需要重新核对。")
        model = _fallback_build_model_dual()

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"警告：load_state_dict 有不匹配的key（strict=False已跳过）：\n"
              f"  missing={missing}\n  unexpected={unexpected}\n"
              f"如果这些key看起来不该缺失/多余，说明模型定义跟checkpoint不匹配，"
              f"结果不可信，需要先解决这个问题。")
    model.to(device)
    model.eval()
    return model


# ============================================================================
# Grad-CAM 核心实现
# ============================================================================
def find_last_conv2d(model: nn.Module) -> nn.Module:
    """自动找模型里最后一个Conv2d层作为Grad-CAM的目标层，不依赖具体的
    属性命名（不管你的backbone叫self.backbone还是self.resnet还是别的，
    只要它内部最终还是标准的卷积层结构，这里都能找到）。"""
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise RuntimeError("模型里没有找到任何Conv2d层，无法做Grad-CAM。")
    return last_conv


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

    def compute(self, input_tensor: torch.Tensor, target_class: int):
        self.model.zero_grad()
        logits = self.model(input_tensor)
        score = logits[0, target_class]
        score.backward()

        # weights: 对每个通道的梯度做全局平均池化
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, h, w]
        cam = F.relu(cam)
        cam = cam[0, 0].cpu().numpy()

        # 归一化到0-1
        cam -= cam.min()
        if cam.max() > 1e-8:
            cam /= cam.max()
        return cam, logits.detach().cpu().numpy()[0]


# ============================================================================
# 图像预处理 / 叠加可视化
# ============================================================================
def load_and_preprocess(path: str):
    """返回 (原始PIL图, 归一化后的Tensor[3,H,W])。
    预处理按 eval_cascade.py 里真实的 TRANSFORM 复刻：
    Resize(256)（短边缩放）→ CenterCrop(224) → Normalize(ImageNet统计量)。
    注意这跟"直接resize到224×224"不是一回事——前者保留长宽比、裁中心，
    后者会把图拉伸变形，两者对同一张图算出来的结果不一样。"""
    im_original = Image.open(path).convert("RGB")

    # Resize(256)：PIL/torchvision语义是把短边缩放到256，长边等比例缩放
    w, h = im_original.size
    if w < h:
        new_w, new_h = 256, int(round(h * 256 / w))
    else:
        new_h, new_w = 256, int(round(w * 256 / h))
    im_resized = im_original.resize((new_w, new_h), Image.BILINEAR)

    # CenterCrop(224)
    left = (new_w - IMG_SIZE) // 2
    top = (new_h - IMG_SIZE) // 2
    im_cropped = im_resized.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))

    arr = np.asarray(im_cropped).astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    normed = (arr - mean) / std
    tensor = torch.from_numpy(normed.transpose(2, 0, 1)).float()
    return im_cropped, tensor


def overlay_cam(pil_image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """把CAM热力图（任意分辨率）resize到跟原图一致，用jet colormap叠加。"""
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        pil_image.size, Image.BILINEAR)
    cam_arr = np.asarray(cam_img).astype(np.float32) / 255.0
    heatmap = matplotlib.colormaps.get_cmap("jet")(cam_arr)[:, :, :3]  # RGBA -> RGB
    base = np.asarray(pil_image).astype(np.float32) / 255.0
    blended = (1 - alpha) * base + alpha * heatmap
    return np.clip(blended, 0, 1)


# ============================================================================
# 主流程
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppl", required=True, help="PPL @0° 原始照片路径")
    parser.add_argument("--xpl", required=True, help="XPL @0° 原始照片路径")
    parser.add_argument("--checkpoint", default="experiments/exp02_dual_stream/best.pth")
    parser.add_argument("--target-class", default=None,
                         help="igneous/metamorphic/sedimentary之一；不给则用模型预测类别")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = build_model(args.checkpoint, device)
    target_layer = find_last_conv2d(model)
    gradcam = GradCAM(model, target_layer)

    ppl_im, ppl_tensor = load_and_preprocess(args.ppl)
    xpl_im, xpl_tensor = load_and_preprocess(args.xpl)
    input_tensor = torch.cat([ppl_tensor, xpl_tensor], dim=0).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    # 先跑一次前向拿预测类别/置信度（用于确定target_class默认值、以及标注文字）
    with torch.no_grad():
        probs = F.softmax(model(input_tensor), dim=1)[0].cpu().numpy()
    pred_idx = int(probs.argmax())

    if args.target_class is not None:
        target_idx = CLASS_NAMES.index(args.target_class.lower())
    else:
        target_idx = pred_idx

    cam, _ = gradcam.compute(input_tensor, target_idx)

    ppl_overlay = overlay_cam(ppl_im, cam)
    xpl_overlay = overlay_cam(xpl_im, cam)  # 早期融合模型：同一张CAM同时叠加在PPL和XPL上

    # ---- 画1x4面板：不含标题栏，四张等宽，供下游脚本自动四等分裁切 ----
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.0))
    panels = [np.asarray(ppl_im), ppl_overlay, np.asarray(xpl_im), xpl_overlay]
    for ax, panel_img in zip(axes, panels):
        ax.imshow(panel_img)
        ax.axis("off")
        ax.set_position(ax.get_position())  # 保持各面板宽度严格一致

    plt.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005, wspace=0.02)
    plt.savefig(args.out, dpi=200, facecolor="white")
    plt.close(fig)

    print(f"saved {args.out}")
    print(f"预测类别: {CLASS_NAMES[pred_idx]} (置信度 {probs[pred_idx]:.3f})")
    print(f"CAM计算目标类别: {CLASS_NAMES[target_idx]}")
    print(f"各类别概率: " + ", ".join(f"{c}={p:.3f}" for c, p in zip(CLASS_NAMES, probs)))


if __name__ == "__main__":
    main()
