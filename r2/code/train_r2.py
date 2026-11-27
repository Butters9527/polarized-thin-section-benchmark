# -*- coding: utf-8 -*-
"""
train_r2.py — Minerals R2 实验第 2 步：统一设置下训练和评估全部模型（在 AutoDL 上运行）

所有模型用同一套训练设置（回应审稿人4“训练设置混杂”的意见）：
  ResNet-50（ImageNet-1K V1 预训练）全参数微调；AdamW lr 1e-4、weight decay 1e-4；余弦退火 T_max=30；
  最多 30 个 epoch；早停 patience 5（看早停验证集 OA，OA 相同时看 loss）；Dropout 0.3；batch 16；bf16 自动混合精度；
  增强：RandomCrop 224、水平/竖直翻转、±90° 旋转，同一样本的所有帧共用一组随机参数（dual_mis 除外）。

模型：
  ppl        单帧 PPL                         xpl0      单帧 XPL@0°
  dual       PPL+XPL@0° 六通道早期融合（Eq. 1）  dual_mis  同上，但训练时两帧独立增强（错位消融）
  seq_mean   8 帧共享骨干 + 帧特征均值（无注意力）
  seq_attn   8 帧 + 多头自注意力（无位置编码，对应原 exp03）
  seq_rpe90  8 帧 + 90° 周期旋转位置编码 + 自注意力（修正后的 exp04）
  xpl7_attn  7 帧纯 XPL + 自注意力

阶段（--stage）：
  A  按岩石名称分组的 5 折 CV（每折内再分组划出约 15% 作早停验证），rep 0–2，全部 7 个模型   ← 主结果
  D  原固定划分（256/32/33），种子 0–4，7 个模型 + dual_mis；种子 0 保存权重（供 Grad-CAM/作图）
  B  同 A，但用分辨率归一化缓存（norm），ppl/xpl0/dual，rep 0–2      ← 分辨率偏差对照
  C  按薄片随机分 5 折（不分组，原论文口径），ppl/xpl0/dual，rep 0–2  ← 量化同类型泄漏的影响
  E  同 A，ppl/xpl0/dual，rep 3–9（与 A 合起来为 10×5 重复 CV）
  F  同 A，4 个序列模型，rep 3–9（全部 7 个模型均为 10×5）
  G  同 B，rep 3–9（分辨率对照 10×5）
  H  同 C，rep 3–9（按薄片随机分折对照 10×5）
  I  同 D，种子 5–9（固定划分共 10 个种子）
  J  按岩石名称分组 CV 10×5：EfficientNet-B0、ViT-B/16、Swin-T 各跑 PPL 和 PPL+XPL（ResNet-50 见 A/E）
  K  按岩石名称分组 CV 10×5：RPE90 固定增益 1.0 与 0.3（与 seq_rpe90 可学习增益 0.1、seq_attn 无编码配对比较）
  L  按“产地 + 岩石名称”连通分组 CV 10×5（同产地或同名称的薄片在同一侧），ppl/xpl0/dual；划分见 folds.csv 中 protocol=locality
  all  依次 A → D → B → C → E → F → G → H → I → J → K → L

每次运行写一个 JSON（训练曲线、最佳 epoch、每片测试预测：中心裁剪、五裁剪平均、通道遮挡）。已存在的 JSON 自动跳过，可随时中断后续跑。
"""
import argparse
import copy
import csv
import json
import math
import os
import pickle
import random
import re
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CROP = 224
ANGLES = {1: None, 2: 0, 3: 15, 4: 30, 5: 45, 6: 60, 7: 75, 8: 90}

FRAMES = {"ppl": [1], "xpl0": [2], "dual": [1, 2], "dual_mis": [1, 2],
          "seq_mean": list(range(1, 9)), "seq_attn": list(range(1, 9)),
          "seq_rpe90": list(range(1, 9)), "xpl7_attn": list(range(2, 9))}
SEQ_MODELS = {"seq_mean", "seq_attn", "seq_rpe90", "xpl7_attn", "seq_rpe90_g03", "seq_rpe90_g1"}
RPE_GAIN = {"seq_rpe90": (0.1, True), "seq_rpe90_g03": (0.3, False), "seq_rpe90_g1": (1.0, False)}  # (初值, 是否可学习)
FRAMES["seq_rpe90_g03"] = list(range(1, 9))
FRAMES["seq_rpe90_g1"] = list(range(1, 9))
BACKBONES = {"effb0": "efficientnet_b0", "vitb16": "vit_b_16", "swint": "swin_t"}
BB_MODELS = [f"{b}_{i}" for b in BACKBONES for i in ("ppl", "dual")]
for _m in BB_MODELS:
    FRAMES[_m] = [1] if _m.endswith("_ppl") else [1, 2]
DUAL_MODELS = {"dual", "dual_mis"} | {m for m in BB_MODELS if m.endswith("_dual")}
ALL7 = ["ppl", "xpl0", "dual", "seq_mean", "seq_attn", "seq_rpe90", "xpl7_attn"]
BASE3 = ["ppl", "xpl0", "dual"]
SEQ4 = ["seq_mean", "seq_attn", "seq_rpe90", "xpl7_attn"]

STAGES = {
    "A": dict(protocol="grouped", prep="orig", reps=range(0, 3), models=ALL7),
    "D": dict(protocol="fixed", prep="orig", seeds=range(0, 5), models=ALL7 + ["dual_mis"]),
    "B": dict(protocol="grouped", prep="norm", reps=range(0, 3), models=BASE3),
    "C": dict(protocol="slide", prep="orig", reps=range(0, 3), models=BASE3),
    "E": dict(protocol="grouped", prep="orig", reps=range(3, 10), models=BASE3),
    "F": dict(protocol="grouped", prep="orig", reps=range(3, 10), models=SEQ4),
    "G": dict(protocol="grouped", prep="norm", reps=range(3, 10), models=BASE3),
    "H": dict(protocol="slide", prep="orig", reps=range(3, 10), models=BASE3),
    "I": dict(protocol="fixed", prep="orig", seeds=range(5, 10), models=ALL7 + ["dual_mis"]),
    "J": dict(protocol="grouped", prep="orig", reps=range(0, 10), models=BB_MODELS),
    "K": dict(protocol="grouped", prep="orig", reps=range(0, 10), models=["seq_rpe90_g1", "seq_rpe90_g03"]),
    "L": dict(protocol="locality", prep="orig", reps=range(0, 10), models=BASE3),
}

CACHE = {}


# ---------------------------------------------------------------- 数据
def load_cache(index_dir, prep):
    if prep not in CACHE:
        t = time.time()
        import io
        with open(os.path.join(index_dir, f"cache_{prep}.pkl"), "rb") as f:
            raw = pickle.load(f)
        CACHE[prep] = {k: (np.asarray(Image.open(io.BytesIO(v)).convert("RGB")) if isinstance(v, bytes) else v)
                       for k, v in raw.items()}
        print(f"  载入缓存 {prep}：{len(CACHE[prep])} 张，{time.time()-t:.1f}s")
    return CACHE[prep]


def to_tensor(img):
    return TF.normalize(TF.to_tensor(img), MEAN, STD)


def sample_params(w, h):
    i = int(torch.randint(0, h - CROP + 1, (1,)))
    j = int(torch.randint(0, w - CROP + 1, (1,)))
    hf = bool(torch.rand(1) < 0.5)
    vf = bool(torch.rand(1) < 0.5)
    ang = float(torch.empty(1).uniform_(-90.0, 90.0))
    return i, j, hf, vf, ang


def augment(img, p):
    i, j, hf, vf, ang = p
    img = TF.crop(img, i, j, CROP, CROP)
    if hf:
        img = TF.hflip(img)
    if vf:
        img = TF.vflip(img)
    return TF.rotate(img, ang, interpolation=InterpolationMode.NEAREST, expand=False, fill=0)


class SectionDS(Dataset):
    """mode: train / val（中心裁剪）/ test（五裁剪，第 5 个即中心裁剪）"""

    def __init__(self, secs, model, mode, cache):
        self.secs, self.model, self.mode, self.cache = secs, model, mode, cache
        self.frames = FRAMES[model]

    def __len__(self):
        return len(self.secs)

    def __getitem__(self, idx):
        s = self.secs[idx]
        imgs = [Image.fromarray(self.cache[re.split(r"[\\/]", s[f"f{p}"])[-1]]) for p in self.frames]
        if self.mode == "train":
            if self.model == "dual_mis":
                ts = [to_tensor(augment(im, sample_params(*im.size))) for im in imgs]
            else:
                p = sample_params(*imgs[0].size)
                ts = [to_tensor(augment(im, p)) for im in imgs]
            x = self._pack(ts)
        elif self.mode == "val":
            x = self._pack([to_tensor(TF.center_crop(im, CROP)) for im in imgs])
        else:
            crops = [TF.five_crop(im, CROP) for im in imgs]  # 每帧 5 个裁剪
            x = torch.stack([self._pack([to_tensor(c[k]) for c in crops]) for k in range(5)])
        return x, int(s["label"]), idx

    def _pack(self, ts):
        if self.model in SEQ_MODELS:
            return torch.stack(ts)          # [T,3,H,W]
        return torch.cat(ts, dim=0)         # [3 或 6,H,W]


# ---------------------------------------------------------------- 模型
def resnet50_backbone(in_ch=3):
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    if in_ch == 6:  # Equation (1)：预训练权重复制到两组通道再乘 0.5
        w = m.conv1.weight.data.clone()
        conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            conv.weight[:, :3] = w
            conv.weight[:, 3:] = w
            conv.weight *= 0.5
        m.conv1 = conv
    m.fc = nn.Identity()
    return m


def _expand_conv(conv):
    """Equation (1) applied to any first conv / patch-embedding layer: copy pretrained weights to both channel groups, ×0.5"""
    w = conv.weight.data.clone()
    new = nn.Conv2d(6, conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding,
                    bias=conv.bias is not None)
    with torch.no_grad():
        new.weight[:, :3] = w
        new.weight[:, 3:] = w
        new.weight *= 0.5
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


def other_backbone(arch, in_ch=3):
    """torchvision ImageNet-1K V1 权重；去掉原分类头，返回 (模块, 特征维度)"""
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        if in_ch == 6:
            m.features[0][0] = _expand_conv(m.features[0][0])
        m.classifier = nn.Identity()
        return m, 1280
    if arch == "vit_b_16":
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        if in_ch == 6:
            m.conv_proj = _expand_conv(m.conv_proj)
        m.heads = nn.Identity()
        return m, 768
    if arch == "swin_t":
        m = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        if in_ch == 6:
            m.features[0][0] = _expand_conv(m.features[0][0])
        m.head = nn.Identity()
        return m, 768
    raise ValueError(arch)


class SingleNet(nn.Module):
    def __init__(self, in_ch, dropout=0.3, arch="resnet50"):
        super().__init__()
        if arch == "resnet50":
            self.backbone, dim = resnet50_backbone(in_ch), 2048
        else:
            self.backbone, dim = other_backbone(arch, in_ch)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, 3))

    def forward(self, x):
        return self.head(self.backbone(x))


class RPE90(nn.Module):
    """90° 周期旋转位置编码。消光强度 ∝ sin²(2θ)，周期 90°，0° 与 90° 光学等价，编码相同。
    第 k 对维度用谐波 m_k = 1 + (k mod 3)：PE(θ,2k)=sin(2π m_k θ/90°)，PE(θ,2k+1)=cos(2π m_k θ/90°)。
    15° 采样在一个 90° 周期内有 6 个不同相位，最多可分辨 3 次谐波。
    编码乘以可学习标量 gain（初值 0.1），训练后报告 ‖gain·PE‖ 与帧特征 ‖F‖ 之比。PPL 帧用可学习向量。"""

    def __init__(self, angles, dim=2048, gain0=0.1, learn_gain=True):
        super().__init__()
        pe = torch.zeros(len(angles), dim)
        k = torch.arange(dim // 2)
        harm = (1 + (k % 3)).float()
        self.ppl_index = [i for i, a in enumerate(angles) if a is None]
        for i, a in enumerate(angles):
            if a is None:
                continue
            ph = 2 * math.pi * harm * a / 90.0
            pe[i, 0::2] = torch.sin(ph)
            pe[i, 1::2] = torch.cos(ph)
        self.register_buffer("pe", pe)
        if learn_gain:
            self.gain = nn.Parameter(torch.tensor(float(gain0)))
        else:  # 固定增益：不参与优化（阶段 K 的强度对照）
            self.register_buffer("gain", torch.tensor(float(gain0)))
        self.ppl_embed = nn.Parameter(torch.randn(dim) * 0.02)

    def code(self):
        c = self.gain * self.pe
        if self.ppl_index:
            c = c.clone()
            for i in self.ppl_index:
                c[i] = self.ppl_embed
        return c  # [T,D]

    def forward(self, f):
        return f + self.code().unsqueeze(0)


class SeqNet(nn.Module):
    def __init__(self, kind, dropout=0.3):
        super().__init__()
        self.kind = kind
        self.backbone = resnet50_backbone(3)
        angles = [ANGLES[p] for p in FRAMES[kind]]
        self.rpe = RPE90(angles, gain0=RPE_GAIN[kind][0], learn_gain=RPE_GAIN[kind][1]) if kind in RPE_GAIN else None
        self.attn = None if kind == "seq_mean" else nn.MultiheadAttention(2048, 8, dropout=0.1, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(2048), nn.Dropout(dropout), nn.Linear(2048, 3))
        self.last_feat_norm = None
        self.feat_norms = []  # 测试时（未遮挡输入）各 batch 的帧特征平均 L2 范数

    def forward(self, x):
        B, T = x.shape[:2]
        f = self.backbone(x.flatten(0, 1)).view(B, T, -1)
        self.last_feat_norm = f.detach().float().norm(dim=-1).mean().item()
        if self.rpe is not None:
            f = self.rpe(f)
        if self.attn is not None:
            f, _ = self.attn(f, f, f, need_weights=False)
        return self.head(f.mean(1))


def build_model(name):
    if name in SEQ_MODELS:
        return SeqNet(name)
    arch = BACKBONES[name.split("_")[0]] if name in BB_MODELS else "resnet50"
    return SingleNet(6 if name in DUAL_MODELS else 3, arch=arch)


# ---------------------------------------------------------------- 训练与评估
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, dev, opt=None, accum=1, max_batches=None):
    train = opt is not None
    model.train(train)
    crit = nn.CrossEntropyLoss()
    tot, n, correct = 0.0, 0, 0
    if train:
        opt.zero_grad(set_to_none=True)
    for b, (x, y, _) in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.set_grad_enabled(train), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out = model(x)
            loss = crit(out.float(), y)
        if train:
            (loss / accum).backward()
            if (b + 1) % accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
        tot += loss.item() * len(y)
        n += len(y)
        correct += (out.argmax(1) == y).sum().item()
    if train and n and (b + 1) % accum != 0:
        opt.step()
        opt.zero_grad(set_to_none=True)
    return tot / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def predict(model, loader, dev, name, max_batches=None):
    model.eval()
    out = {}
    for b, (x, y, idx) in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        x = x.to(dev)                       # [B,5,...]
        B = x.shape[0]
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            p = torch.softmax(model(x.flatten(0, 1)).float(), -1).view(B, 5, 3)
            if getattr(model, "last_feat_norm", None) is not None:
                model.feat_norms.append(model.last_feat_norm)
            xc = x[:, 4]
            occ = {}
            if name in DUAL_MODELS:
                for tag, sl in (("no_ppl", slice(0, 3)), ("no_xpl", slice(3, 6))):
                    z = xc.clone()
                    z[:, sl] = 0
                    occ[tag] = torch.softmax(model(z).float(), -1)
            elif name in SEQ_MODELS and name != "xpl7_attn":
                z = xc.clone()
                z[:, 0] = 0
                occ["no_ppl"] = torch.softmax(model(z).float(), -1)
        pc, pm = p[:, 4], p.mean(1)
        for i in range(B):
            d = dict(label=int(y[i]), prob_center=[round(v, 5) for v in pc[i].tolist()],
                     prob_multi=[round(v, 5) for v in pm[i].tolist()],
                     pred_center=int(pc[i].argmax()), pred_multi=int(pm[i].argmax()))
            for tag, q in occ.items():
                d[f"pred_{tag}"] = int(q[i].argmax())
            out[int(idx[i])] = d
    return out


def run_job(job, secs_all, splits, a, dev):
    name, protocol, prep = job["model"], job["protocol"], job["prep"]
    cache = load_cache(a.index, prep)
    roles = splits(job)
    by_role = {r: [s for s in secs_all if roles.get(s["section_key"]) == r] for r in ("train", "val", "test")}
    set_seed(job["seed"])
    seq = name in SEQ_MODELS
    bs = a.batch
    g = torch.Generator()
    g.manual_seed(job["seed"])
    kw = dict(num_workers=a.workers, pin_memory=dev.type == "cuda", persistent_workers=a.workers > 0)
    tl = DataLoader(SectionDS(by_role["train"], name, "train", cache), batch_size=bs, shuffle=True,
                    drop_last=False, generator=g, **kw)
    vl = DataLoader(SectionDS(by_role["val"], name, "val", cache), batch_size=bs, **kw)
    te = DataLoader(SectionDS(by_role["test"], name, "test", cache), batch_size=max(2, bs // 4 if seq else bs // 2), **kw)

    model = build_model(name).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    mb = 2 if a.smoke else None
    hist, best, best_state, wait = [], None, None, 0
    t0 = time.time()
    for ep in range(1, (1 if a.smoke else a.epochs) + 1):
        lr = opt.param_groups[0]["lr"]
        trl, tra = run_epoch(model, tl, dev, opt, a.accum, mb)
        vll, vla = run_epoch(model, vl, dev, None, 1, mb)
        sched.step()
        hist.append(dict(epoch=ep, lr=lr, train_loss=round(trl, 5), train_acc=round(tra, 5),
                         val_loss=round(vll, 5), val_acc=round(vla, 5), sec=round(time.time() - t0, 1)))
        if getattr(model, "rpe", None) is not None:
            hist[-1]["rpe_gain"] = round(float(model.rpe.gain), 6)
        better = best is None or vla > best[0] + 1e-9 or (abs(vla - best[0]) < 1e-9 and vll < best[1])
        if better:
            best, wait = (vla, vll, ep), 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= a.patience:
                break
    model.load_state_dict(best_state)
    preds = predict(model, te, dev, name, mb)
    test_secs = by_role["test"]
    rec = dict(job=job, n_train=len(by_role["train"]), n_val=len(by_role["val"]), n_test=len(test_secs),
               best_epoch=best[2], best_val_acc=best[0], best_val_loss=best[1], epochs_run=len(hist),
               history=hist, train_seconds=round(time.time() - t0, 1),
               settings=dict(lr=1e-4, wd=1e-4, epochs=a.epochs, patience=a.patience, batch=bs, accum=a.accum,
                             dropout=0.3, amp="bf16" if dev.type == "cuda" else "fp32"),
               test=[dict(section_key=test_secs[i]["section_key"], **d) for i, d in sorted(preds.items())])
    if name in RPE_GAIN:
        rpe = model.rpe
        code = rpe.code().detach().float()
        xpl = [i for i in range(code.shape[0]) if i not in rpe.ppl_index]
        fn = float(np.mean(model.feat_norms)) if model.feat_norms else float("nan")
        rec["rpe"] = dict(gain=float(rpe.gain), pe_norm_xpl=float(code[xpl].norm(dim=-1).mean()),
                          ppl_embed_norm=float(rpe.ppl_embed.norm()), feat_norm_test=fn)
        rec["rpe"]["ratio_pe_to_feat"] = rec["rpe"]["pe_norm_xpl"] / fn
    if a.save_ckpt and job.get("save_ckpt"):
        torch.save(best_state, job["out"].replace(".json", ".pth"))
    return rec


def build_jobs(stage_names, a):
    jobs = []
    for st in stage_names:
        S = STAGES[st]
        for m in S["models"]:
            if S["protocol"] == "fixed":
                for sd in S["seeds"]:
                    jobs.append(dict(stage=st, model=m, protocol="fixed", prep=S["prep"], seed=sd, rep=None, fold=None,
                                     save_ckpt=(sd == 0),
                                     out=os.path.join(a.results, f"fixed_{S['prep']}", m, f"s{sd}.json")))
            else:
                for rep in S["reps"]:
                    for fo in range(5):
                        jobs.append(dict(stage=st, model=m, protocol=S["protocol"], prep=S["prep"],
                                         seed=10000 + 100 * rep + fo, rep=rep, fold=fo,
                                         out=os.path.join(a.results, f"{S['protocol']}_{S['prep']}", m,
                                                          f"r{rep}_f{fo}.json")))
    if a.models:
        keep = set(a.models.split(","))
        jobs = [j for j in jobs if j["model"] in keep]
    if a.smoke:
        seen, sj = set(), []
        for j in jobs:
            if j["model"] not in seen:
                seen.add(j["model"])
                j = dict(j, out=j["out"].replace(a.results, a.results + "_smoke"))
                sj.append(j)
        jobs = sj
    # 同一缓存的任务排在一起，减少反复载入
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="index")
    ap.add_argument("--results", default="results")
    ap.add_argument("--stage", default="all", help="A–L 中的一个或逗号组合，或 all")
    ap.add_argument("--models", default="", help="只跑这些模型（逗号分隔），默认阶段内全部")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1, help="显存不够时：--batch 8 --accum 2，等效 16")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save-ckpt", action="store_true", default=True)
    ap.add_argument("--smoke", action="store_true", help="冒烟测试：每个模型 1 个任务、1 个 epoch、每段 2 个 batch")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("设备：", torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU")
    torch.backends.cudnn.benchmark = True

    with open(os.path.join(a.index, "sections.csv"), encoding="utf-8-sig") as f:
        secs = list(csv.DictReader(f))
    folds = {}
    with open(os.path.join(a.index, "folds.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            folds.setdefault((r["protocol"], int(r["rep"]), int(r["fold"])), {})[r["section_key"]] = \
                {"train": "train", "inner_val": "val", "test": "test"}[r["role"]]
    fixed = {s["section_key"]: s["fixed_split"] for s in secs}

    def splits(job):
        if job["protocol"] == "fixed":
            return fixed
        return folds[(job["protocol"], job["rep"], job["fold"])]

    stages = list("ADBCEFGHIJKL") if a.stage == "all" else a.stage.split(",")
    jobs = build_jobs(stages, a)
    todo = [j for j in jobs if not os.path.exists(j["out"])]
    print(f"阶段 {stages}：共 {len(jobs)} 个任务，待跑 {len(todo)} 个")
    fails = []
    t_all = time.time()
    for k, job in enumerate(todo, 1):
        os.makedirs(os.path.dirname(job["out"]), exist_ok=True)
        tag = f"[{k}/{len(todo)}] {job['stage']} {job['protocol']}_{job['prep']} {job['model']} " + \
              (f"seed{job['seed']}" if job["protocol"] == "fixed" else f"r{job['rep']} f{job['fold']}")
        print(tag, "…", flush=True)
        try:
            rec = run_job(job, secs, splits, a, dev)
            tmp = job["out"] + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False)
            os.replace(tmp, job["out"])
            acc = np.mean([t["pred_center"] == t["label"] for t in rec["test"]]) if rec["test"] else float("nan")
            print(f"    完成：best epoch {rec['best_epoch']}，早停验证 OA {rec['best_val_acc']:.3f}，"
                  f"测试 OA {acc:.3f}，{rec['train_seconds']:.0f}s；累计 {(time.time()-t_all)/3600:.2f} h", flush=True)
        except Exception as e:
            fails.append((tag, repr(e)))
            print("    失败：", repr(e))
            traceback.print_exc()
            if "out of memory" in repr(e).lower():
                print("    显存不足：请改用 --batch 8 --accum 2 重跑（已完成的任务会自动跳过）")
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    print(f"\n全部结束，用时 {(time.time()-t_all)/3600:.2f} h；失败 {len(fails)} 个")
    for t, e in fails:
        print("  ", t, e)


if __name__ == "__main__":
    main()
