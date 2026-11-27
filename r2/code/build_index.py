# -*- coding: utf-8 -*-
"""
build_index.py — Minerals R2 实验第 1 步：建索引、分折、缓存缩放后的图像（在 AutoDL 上运行）

输入：
  --data   NJU 数据集根目录（解压后的“南京大学三大岩类数据集”文件夹）
  --fixed  原论文固定划分 fixed_frame_assignments.csv（随包提供）
输出（--out，默认 ./index）：
  sections.csv        321 片可用薄片：薄片号、岩类、rock_id、slide_id、岩石名称、分组键、分辨率、8 帧文件路径、原固定划分
  folds.csv           全部交叉验证划分：protocol(grouped/slide) × rep(0–9) × fold(0–4) × 薄片 → role(train/inner_val/test)
  index_report.md     核对报告（请发回）
  cache_orig.pkl      每张图按原流程 Resize(256)（短边 256、保持长宽比）后的图像（PNG 无损字节）
  cache_norm.pkl      分辨率归一化：先居中裁成 5:4，再缩到 1280×1024，最后 Resize(256)
可以在本机运行（推荐：只需上传 index 文件夹，不必上传原图），也可以在 AutoDL 上运行。

分组口径（用户 2026-11-20 决定）：按岩石名称分组（去掉薄片号中的数字），同名不同 rock_id 的岩石归入同一组。
"""
import argparse
import csv
import os
import pickle
import random
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
from PIL import Image

CLS = {"火": "igneous", "变": "metamorphic", "沉": "sedimentary"}
LABEL = {"igneous": 0, "metamorphic": 1, "sedimentary": 2}
STEM_RE = re.compile(r"^([火变沉])(\d+)(.+?)(\d+)-(\d+)$")
N_REPS = 10
N_FOLDS = 5
INNER_FRAC = 0.15


# ---------------------------------------------------------------- 扫描
def scan(data_root):
    secs = {}
    bad = []
    for dp, _, fns in os.walk(data_root):
        for fn in fns:
            if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".bmp")):
                continue
            stem = os.path.splitext(fn)[0].strip()
            m = STEM_RE.match(stem)
            if not m:
                bad.append(os.path.join(dp, fn))
                continue
            c, rid, name, sid, pid = m.groups()
            cls = CLS[c]
            key = f"{c}{rid}{name}{sid}"
            s = secs.setdefault(key, dict(section_key=key, rock_class=cls, label=LABEL[cls],
                                          rock_id=int(rid), slide_id=int(sid), rock_name=name,
                                          type_key=f"{cls}:{name}", frames={}))
            s["frames"][int(pid)] = os.path.join(dp, fn)
    return secs, bad


# ---------------------------------------------------------------- 分折
def assign_folds(items, n_folds, rng):
    """items: list of (group_key, cls, size)。按岩类分层、按组分配，使每折每类的薄片数尽量均衡。"""
    by_cls = defaultdict(list)
    for g, c, n in items:
        by_cls[c].append((g, n))
    fold_of = {}
    for c in sorted(by_cls):
        groups = by_cls[c][:]
        rng.shuffle(groups)
        groups.sort(key=lambda x: -x[1])  # 稳定排序：大组先分
        cnt = [0] * n_folds
        for g, n in groups:
            f = min(range(n_folds), key=lambda i: (cnt[i], i))
            fold_of[g] = f
            cnt[f] += n
    return fold_of


def pick_inner(items, frac, rng):
    """从外层训练集中按组、按岩类分层选出约 frac 的薄片作早停用验证集。"""
    by_cls = defaultdict(list)
    for g, c, n in items:
        by_cls[c].append((g, n))
    chosen = set()
    for c in sorted(by_cls):
        groups = by_cls[c][:]
        rng.shuffle(groups)
        target = round(frac * sum(n for _, n in groups))
        got = 0
        for g, n in groups:
            if got >= target:
                break
            chosen.add(g)
            got += n
    return chosen


def make_folds(sections):
    rows = []
    for protocol in ("grouped", "slide"):
        gkey = (lambda s: s["type_key"]) if protocol == "grouped" else (lambda s: s["section_key"])
        groups = defaultdict(list)
        for s in sections:
            groups[gkey(s)].append(s)
        items = [(g, v[0]["rock_class"], len(v)) for g, v in groups.items()]
        for rep in range(N_REPS):
            rng = random.Random(20261120 + 1000 * (protocol == "slide") + rep)
            fold_of = assign_folds(items, N_FOLDS, rng)
            for f in range(N_FOLDS):
                train_items = [it for it in items if fold_of[it[0]] != f]
                inner = pick_inner(train_items, INNER_FRAC, random.Random(rng.random()))
                for g, v in groups.items():
                    role = "test" if fold_of[g] == f else ("inner_val" if g in inner else "train")
                    for s in v:
                        rows.append(dict(protocol=protocol, rep=rep, fold=f,
                                         section_key=s["section_key"], role=role))
    return rows


# ---------------------------------------------------------------- 缓存
# 只用 PIL 实现，与 torchvision 对 PIL 图像的 Resize(256) / CenterCrop 逐像素一致：
#   Resize(256)：短边 256，长边 int(256×长/短)，PIL BILINEAR（PIL 缩小时自带抗锯齿）
# 缓存以 PNG（无损）字节保存，与 numpy 版本无关，体积约为原始数组的一半左右。
def _resize_shorter(img, size=256):
    w, h = img.size
    if w <= h:
        nw, nh = size, int(size * h / w)
    else:
        nw, nh = int(size * w / h), size
    return img.resize((nw, nh), Image.BILINEAR)


def _center_crop(img, ch, cw):
    w, h = img.size
    top = int(round((h - ch) / 2.0))
    left = int(round((w - cw) / 2.0))
    return img.crop((left, top, left + cw, top + ch))


def _load(args):
    import io
    path, prep = args
    img = Image.open(path).convert("RGB")
    if prep == "norm":
        w, h = img.size
        if w / h > 1.25 + 1e-3:
            img = _center_crop(img, h, int(round(h * 1.25)))
        elif w / h < 1.25 - 1e-3:
            img = _center_crop(img, int(round(w / 1.25)), w)
        if img.size[1] > 1024:
            img = img.resize((1280, 1024), Image.BILINEAR)
    img = _resize_shorter(img, 256)
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=6)
    return os.path.basename(path), buf.getvalue(), img.size


def build_cache(sections, prep, out_dir, workers):
    jobs = [(s["frames"][p], prep) for s in sections for p in range(1, 9)]
    with Pool(workers) as pool:
        res = pool.map(_load, jobs, chunksize=16)
    cache = {k: b for k, b, _ in res}
    path = os.path.join(out_dir, f"cache_{prep}.pkl")
    with open(path, "wb") as f:
        pickle.dump(cache, f, protocol=4)
    shapes = Counter(sz for _, _, sz in res)
    return path, shapes


# ---------------------------------------------------------------- 主程序
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--fixed", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "fixed_frame_assignments.csv"))
    ap.add_argument("--out", default="index")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-cache", action="store_true", help="只建索引和分折，不做图像缓存")
    ap.add_argument("--zip", default="", help="完成后把输出文件夹打包成这个 zip（供上传 AutoDL）")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = []

    secs, bad = scan(a.data)
    R.append("# R2 索引核对报告\n")
    R.append(f"- 数据目录：`{a.data}`")
    R.append(f"- 解析到的薄片：{len(secs)}；文件名无法解析的图像：{len(bad)}")
    for b in bad[:10]:
        R.append(f"  - `{b}`")
    complete = [s for s in secs.values() if all(p in s["frames"] for p in range(1, 9))]
    R.append(f"- 帧 1–8 齐全的薄片：**{len(complete)}**（应为 321）")

    # 分辨率（以 PPL 帧为准）
    for s in complete:
        w, h = Image.open(s["frames"][1]).size
        s["width"], s["height"] = w, h
        s["res_group"] = "high" if max(w, h) >= 3000 else "standard"

    # 原固定划分
    fixed = {}
    with open(a.fixed, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            fixed[(r["rock_class"], int(r["rock_id"]), int(r["slide_id"]))] = (r["split"], r["rock_name"])
    miss, name_mismatch = 0, []
    for s in complete:
        k = (s["rock_class"], s["rock_id"], s["slide_id"])
        if k in fixed:
            s["fixed_split"] = fixed[k][0]
            if fixed[k][1] != s["rock_name"]:
                name_mismatch.append((s["section_key"], fixed[k][1]))
        else:
            s["fixed_split"] = ""
            miss += 1
    fx = Counter(s["fixed_split"] for s in complete)
    R.append(f"- 原固定划分：{dict(fx)}（应为 train 256 / val 32 / test 33）；未匹配 {miss}；名称不一致 {len(name_mismatch)}")
    for k, n in name_mismatch[:10]:
        R.append(f"  - {k} ↔ {n}")

    tk = Counter(s["type_key"] for s in complete)
    rid = Counter((s["rock_class"], s["rock_id"]) for s in complete)
    R.append(f"- 分组（岩石名称）：{len(tk)} 组；每组薄片数分布 {dict(sorted(Counter(tk.values()).items()))}")
    R.append(f"- rock_id：{len(rid)} 种")
    hi = Counter((s["rock_class"], s["res_group"]) for s in complete)
    R.append("- 高分辨率薄片：" + "，".join(f"{c} {hi[(c,'high')]}/{hi[(c,'high')]+hi[(c,'standard')]}"
                                     for c in LABEL))

    complete.sort(key=lambda s: (s["label"], s["rock_id"], s["slide_id"]))
    with open(os.path.join(a.out, "sections.csv"), "w", encoding="utf-8-sig", newline="") as f:
        cols = ["section_key", "rock_class", "label", "rock_id", "slide_id", "rock_name", "type_key",
                "width", "height", "res_group", "fixed_split"] + [f"f{p}" for p in range(1, 9)]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for s in complete:
            row = {c: s.get(c, "") for c in cols[:11]}
            row.update({f"f{p}": os.path.relpath(s["frames"][p], a.data) for p in range(1, 9)})
            w.writerow(row)

    rows = make_folds(complete)
    with open(os.path.join(a.out, "folds.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["protocol", "rep", "fold", "section_key", "role"])
        w.writeheader()
        w.writerows(rows)

    # 分折核对：每折各岩类薄片数；grouped 协议下测试折与训练折共享的岩石名称数（应为 0）
    sk = {s["section_key"]: s for s in complete}
    R.append("\n## 分折核对（rep 0）\n")
    for protocol in ("grouped", "slide"):
        R.append(f"**{protocol}**\n")
        R.append("| 折 | 测试 ign/met/sed | 早停验证 | 训练 | 测试中高分辨率 | 测试与训练共享的岩石名称 |")
        R.append("|---|---|---:|---:|---:|---:|")
        for fo in range(N_FOLDS):
            sel = [r for r in rows if r["protocol"] == protocol and r["rep"] == 0 and r["fold"] == fo]
            te = [sk[r["section_key"]] for r in sel if r["role"] == "test"]
            tr = [sk[r["section_key"]] for r in sel if r["role"] != "test"]
            iv = sum(1 for r in sel if r["role"] == "inner_val")
            cc = Counter(s["rock_class"] for s in te)
            shared = {s["type_key"] for s in te} & {s["type_key"] for s in tr}
            R.append(f"| {fo} | {cc['igneous']}/{cc['metamorphic']}/{cc['sedimentary']} | {iv} | "
                     f"{len(tr)-iv} | {sum(1 for s in te if s['res_group']=='high')} | {len(shared)} |")
        R.append("")
    # grouped 协议所有 rep 的泄漏检查
    leak = 0
    for rep in range(N_REPS):
        for fo in range(N_FOLDS):
            sel = [r for r in rows if r["protocol"] == "grouped" and r["rep"] == rep and r["fold"] == fo]
            te = {sk[r["section_key"]]["type_key"] for r in sel if r["role"] == "test"}
            tr = {sk[r["section_key"]]["type_key"] for r in sel if r["role"] != "test"}
            iv = {sk[r["section_key"]]["type_key"] for r in sel if r["role"] == "inner_val"}
            trn = {sk[r["section_key"]]["type_key"] for r in sel if r["role"] == "train"}
            leak += len(te & tr) + len(iv & trn)
    R.append(f"grouped 协议 10×5 折中，测试/早停验证与训练共享岩石名称的总次数：**{leak}**（应为 0）\n")

    if not a.no_cache:
        for prep in ("orig", "norm"):
            print(f"缓存 {prep} …")
            path, shapes = build_cache(complete, prep, a.out, a.workers)
            R.append(f"- 缓存 `{os.path.basename(path)}`：{sum(shapes.values())} 张，{os.path.getsize(path)/1024**2:.0f} MB；尺寸（宽×高）"
                     + "，".join(f"{k[0]}×{k[1]}: {v}" for k, v in shapes.most_common()))

    with open(os.path.join(a.out, "index_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(R))
    print("\n".join(R))
    if a.zip:
        import zipfile
        os.makedirs(os.path.dirname(os.path.abspath(a.zip)), exist_ok=True)
        top = os.path.basename(os.path.normpath(a.out))
        with zipfile.ZipFile(a.zip, "w", zipfile.ZIP_STORED) as z:
            for fn in sorted(os.listdir(a.out)):
                z.write(os.path.join(a.out, fn), f"{top}/{fn}")
        print(f"\n已打包：{a.zip}（{os.path.getsize(a.zip)/1024**2:.0f} MB）")


if __name__ == "__main__":
    main()
