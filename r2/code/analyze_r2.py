# -*- coding: utf-8 -*-
"""
analyze_r2.py — Minerals R2 实验第 3 步：汇总统计（AutoDL 或本地均可，只需 numpy）

用法：python analyze_r2.py --results results --index index --out analysis
输出：
  analysis/report_R2.md        主报告（请发回）
  analysis/cv_summary.csv      各协议×模型的汇总指标
  analysis/pairwise.csv        模型两两比较（修正重采样 t 检验、按岩石名称分组的 bootstrap 置信区间）
  analysis/fixed_summary.csv   原固定划分多种子结果与 McNemar 检验
  analysis/curves_fixed.csv    固定划分各种子的训练曲线（作图用）
  analysis/oof_predictions.csv 每片的折外预测（各协议×模型×rep）

统计口径：
  - 均值 ± 样本标准差（ddof=1）
  - 修正重采样 t 检验（Nadeau & Bengio 2003；重复 k 折见 Bouckaert & Frank 2004）：
      t = mean(d) / sqrt((1/(k·r) + n_test/n_train) · var(d))，df = k·r − 1，d 为同一 rep、同一折上两模型的 OA 差
  - bootstrap：以岩石名称为单位整组重抽样（2000 次），对各 rep 折外预测的 OA 差取平均后求 2.5%/97.5% 分位数
  - McNemar：精确二项检验（双侧），用于原固定划分 33 片测试集
"""
import argparse
import csv
import glob
import json
import math
import os
import random
from collections import Counter, defaultdict

import numpy as np

CLS = ["igneous", "metamorphic", "sedimentary"]
NAMES = {"ppl": "PPL", "xpl0": "XPL@0°", "dual": "PPL+XPL", "dual_mis": "PPL+XPL（错位增强）",
         "seq_mean": "8帧均值池化", "seq_attn": "8帧注意力", "seq_rpe90": "8帧注意力+RPE90",
         "xpl7_attn": "7帧XPL注意力"}
ORDER = ["ppl", "xpl0", "dual", "dual_mis", "seq_mean", "seq_attn", "seq_rpe90", "xpl7_attn",
         "effb0_ppl", "effb0_dual", "vitb16_ppl", "vitb16_dual", "swint_ppl", "swint_dual",
         "seq_rpe90_g03", "seq_rpe90_g1"]
NAMES.update({"seq_rpe90_g03": "8帧注意力+RPE90（固定增益0.3）", "seq_rpe90_g1": "8帧注意力+RPE90（固定增益1.0）"})
NAMES.update({"effb0_ppl": "EfficientNet-B0 PPL", "effb0_dual": "EfficientNet-B0 PPL+XPL",
              "vitb16_ppl": "ViT-B/16 PPL", "vitb16_dual": "ViT-B/16 PPL+XPL",
              "swint_ppl": "Swin-T PPL", "swint_dual": "Swin-T PPL+XPL"})
PAIRS = [("dual", "ppl"), ("dual", "xpl0"), ("xpl0", "ppl"), ("seq_attn", "dual"), ("seq_mean", "dual"),
         ("seq_attn", "seq_mean"), ("seq_rpe90", "seq_attn"), ("seq_rpe90", "dual"), ("xpl7_attn", "seq_attn"),
         ("effb0_dual", "effb0_ppl"), ("vitb16_dual", "vitb16_ppl"), ("swint_dual", "swint_ppl"),
         ("effb0_dual", "dual"), ("vitb16_dual", "dual"), ("swint_dual", "dual"),
         ("effb0_ppl", "ppl"), ("vitb16_ppl", "ppl"), ("swint_ppl", "ppl"),
         ("seq_rpe90_g03", "seq_attn"), ("seq_rpe90_g1", "seq_attn"),
         ("seq_rpe90_g1", "seq_rpe90"), ("seq_rpe90_g03", "seq_rpe90"), ("seq_rpe90_g1", "seq_rpe90_g03")]


# ---------------------------------------------------------------- 统计工具
def betacf(a, b, x):
    MAXIT, EPS, FPMIN = 300, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > FPMIN else FPMIN)
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > FPMIN else FPMIN)
        c = 1 + aa / c; c = c if abs(c) > FPMIN else FPMIN
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > FPMIN else FPMIN)
        c = 1 + aa / c; c = c if abs(c) > FPMIN else FPMIN
        de = d * c
        h *= de
        if abs(de - 1) < EPS:
            break
    return h


def betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return bt * betacf(a, b, x) / a
    return 1 - bt * betacf(b, a, 1 - x) / b


def t_two_sided_p(t, df):
    return betai(df / 2, 0.5, df / (df + t * t))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    out = dict(oa=float(np.mean(y == p)))
    f1s = []
    for k, c in enumerate(CLS):
        tp = np.sum((y == k) & (p == k)); fn = np.sum((y == k) & (p != k)); fp = np.sum((y != k) & (p == k))
        rec = tp / (tp + fn) if tp + fn else float("nan")
        pre = tp / (tp + fp) if tp + fp else 0.0
        f1s.append(0.0 if (pre + (rec if rec == rec else 0)) == 0 else 2 * pre * rec / (pre + rec))
        out[f"rec_{c}"] = float(rec)
    out["macro_f1"] = float(np.mean(f1s))
    return out


def ms(v, nd=2, pct=True):
    v = [x for x in v if x == x]
    if not v:
        return "—"
    f = 100 if pct else 1
    if len(v) == 1:
        return f"{v[0]*f:.{nd}f}"
    return f"{np.mean(v)*f:.{nd}f} ± {np.std(v, ddof=1)*f:.{nd}f}"


# ---------------------------------------------------------------- 主程序
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--index", default="index")
    ap.add_argument("--out", default="analysis")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    with open(os.path.join(a.index, "sections.csv"), encoding="utf-8-sig") as f:
        secs = {r["section_key"]: r for r in csv.DictReader(f)}
    locp = os.path.join(a.index, "section_locality.csv")
    if os.path.exists(locp):  # 产地 + 名称连通分组（阶段 L 的 bootstrap 抽样单元）
        with open(locp, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                secs[r["section_key"]]["loc_group"] = r["loc_group"]
                secs[r["section_key"]]["locality"] = r["locality"]

    runs = defaultdict(list)  # (setting, model) -> list of records
    for p in glob.glob(os.path.join(a.results, "*", "*", "*.json")):
        setting = os.path.basename(os.path.dirname(os.path.dirname(p)))
        model = os.path.basename(os.path.dirname(p))
        with open(p, encoding="utf-8") as f:
            runs[(setting, model)].append(json.load(f))
    R = ["# Minerals R2 实验统计报告\n"]
    R.append("口径：OA 等为薄片级；“±”为样本标准差；未注明时为中心裁剪预测。\n")
    R.append("已读取的结果：" + "，".join(f"{s}/{m} {len(v)}" for (s, m), v in sorted(runs.items())) + "\n")

    cv_rows, pair_rows, oof_rows = [], [], []
    settings = sorted({s for s, _ in runs if not s.startswith("fixed")})
    for setting in settings:
        models_here = [m for m in ORDER if (setting, m) in runs]
        R.append(f"\n## {setting}\n")
        R.append("| 模型 | 运行数 | 折 OA | 折 macro-F1 | 折外合并 OA（各 rep） | 火成岩召回 | 变质岩召回 | 沉积岩召回 | 五裁剪 OA | 去 PPL OA | 去 XPL OA | 最佳 epoch |")
        R.append("|---|---:|---|---|---|---|---|---|---|---|---|---|")
        per_fold = {}
        oof = {}
        rpe_note = []
        for m in models_here:
            rs = runs[(setting, m)]
            fm = [metrics([t["label"] for t in r["test"]], [t["pred_center"] for t in r["test"]]) for r in rs]
            fmm = [metrics([t["label"] for t in r["test"]], [t["pred_multi"] for t in r["test"]]) for r in rs]
            per_fold[m] = {(r["job"]["rep"], r["job"]["fold"]): (fm[i]["oa"], r["n_test"], r["n_train"] + r["n_val"])
                           for i, r in enumerate(rs)}
            by_rep = defaultdict(dict)
            for r in rs:
                for t in r["test"]:
                    by_rep[r["job"]["rep"]][t["section_key"]] = t
            oof[m] = by_rep
            pooled = [metrics([t["label"] for t in d.values()], [t["pred_center"] for t in d.values()])
                      for rep, d in sorted(by_rep.items()) if len(d) == len(secs)]
            occ = {}
            for tag in ("no_ppl", "no_xpl"):
                v = [np.mean([t[f"pred_{tag}"] == t["label"] for t in r["test"]]) for r in rs if f"pred_{tag}" in r["test"][0]]
                occ[tag] = ms(v) if v else "—"
            row = dict(setting=setting, model=m, runs=len(rs),
                       fold_oa=ms([x["oa"] for x in fm]), fold_f1=ms([x["macro_f1"] for x in fm], 3, False),
                       pooled_oa=ms([x["oa"] for x in pooled]),
                       rec_ign=ms([x["rec_igneous"] for x in pooled], 3, False),
                       rec_met=ms([x["rec_metamorphic"] for x in pooled], 3, False),
                       rec_sed=ms([x["rec_sedimentary"] for x in pooled], 3, False),
                       multi_oa=ms([x["oa"] for x in fmm]), no_ppl=occ["no_ppl"], no_xpl=occ["no_xpl"],
                       best_epoch=ms([r["best_epoch"] for r in rs], 1, False))
            cv_rows.append(row)
            R.append(f"| {NAMES[m]} | {row['runs']} | {row['fold_oa']} | {row['fold_f1']} | {row['pooled_oa']} | "
                     f"{row['rec_ign']} | {row['rec_met']} | {row['rec_sed']} | {row['multi_oa']} | {row['no_ppl']} | "
                     f"{row['no_xpl']} | {row['best_epoch']} |")
            for rep, d in by_rep.items():
                for k, t in d.items():
                    oof_rows.append(dict(setting=setting, model=m, rep=rep, section_key=k, label=t["label"],
                                         pred_center=t["pred_center"], pred_multi=t["pred_multi"]))
            if m.startswith("seq_rpe90"):
                rp = [r["rpe"] for r in rs if "rpe" in r]
                if rp:
                    rpe_note.append(f"\n{NAMES[m]}：gain {ms([x['gain'] for x in rp],3,False)}；‖gain·PE‖ {ms([x['pe_norm_xpl'] for x in rp],2,False)}；"
                             f"帧特征 ‖F‖ {ms([x['feat_norm_test'] for x in rp],2,False)}；比值 {ms([x['ratio_pe_to_feat'] for x in rp],3,False)}\n")

        R.extend(rpe_note)
        # 分辨率分组
        R.append(f"\n**分辨率分组（折外合并，各 rep 平均）**\n")
        R.append("| 模型 | 标准分辨率 OA | 高分辨率 OA | 变质岩召回（标准） | 变质岩召回（高） | 高分辨率薄片被判为变质岩的比例 |")
        R.append("|---|---|---|---|---|---|")
        for m in models_here:
            vals = defaultdict(list)
            for rep, d in oof[m].items():
                for g in ("standard", "high"):
                    ks = [k for k in d if secs[k]["res_group"] == g]
                    if not ks:
                        continue
                    vals[f"oa_{g}"].append(np.mean([d[k]["pred_center"] == d[k]["label"] for k in ks]))
                    km = [k for k in ks if secs[k]["rock_class"] == "metamorphic"]
                    if km:
                        vals[f"met_{g}"].append(np.mean([d[k]["pred_center"] == 1 for k in km]))
                    if g == "high":
                        vals["hi_as_met"].append(np.mean([d[k]["pred_center"] == 1 for k in ks]))
            R.append(f"| {NAMES[m]} | {ms(vals['oa_standard'])} | {ms(vals['oa_high'])} | "
                     f"{ms(vals['met_standard'],3,False)} | {ms(vals['met_high'],3,False)} | {ms(vals['hi_as_met'])} |")

        # 原固定测试集 33 片在 CV 中的难度
        fx = [k for k, s in secs.items() if s["fixed_split"] == "test"]
        R.append(f"\n**原固定测试集 33 片在本协议折外预测中的准确率（判断该测试集是否偏难）**\n")
        R.append("| 模型 | 33 片 | 其余 288 片 |")
        R.append("|---|---|---|")
        for m in models_here:
            a33, aoth = [], []
            for rep, d in oof[m].items():
                a33.append(np.mean([d[k]["pred_center"] == d[k]["label"] for k in fx if k in d]))
                aoth.append(np.mean([d[k]["pred_center"] == d[k]["label"] for k in d if k not in fx]))
            R.append(f"| {NAMES[m]} | {ms(a33)} | {ms(aoth)} |")

        # 两两比较
        R.append(f"\n**模型比较（OA 差 = 前者 − 后者）**\n")
        R.append("| 比较 | 配对折数 | 折 OA 差均值 | 修正 t | df | p | 朴素配对 t 的 p（仅供对照） | 分组 bootstrap 95% CI |")
        R.append("|---|---:|---|---:|---:|---:|---:|---|")
        groups = defaultdict(list)
        gfield = "loc_group" if setting.startswith("locality") and "loc_group" in next(iter(secs.values())) else "type_key"
        for k, s in secs.items():
            groups[s[gfield]].append(k)
        gkeys = sorted(groups)
        for m1, m2 in PAIRS:
            if m1 not in per_fold or m2 not in per_fold:
                continue
            common = sorted(set(per_fold[m1]) & set(per_fold[m2]))
            if len(common) < 2:
                continue
            d = np.array([per_fold[m1][c][0] - per_fold[m2][c][0] for c in common])
            ratio = np.mean([per_fold[m1][c][1] / per_fold[m1][c][2] for c in common])
            reps_n = len({c[0] for c in common})
            k = len(common) / reps_n
            var = np.var(d, ddof=1)
            df = len(common) - 1
            if var > 0:
                tc = d.mean() / math.sqrt((1 / (k * reps_n) + ratio) * var)
                tn = d.mean() / math.sqrt(var / len(common))
                pc, pn = t_two_sided_p(tc, df), t_two_sided_p(tn, df)
            else:
                tc = tn = float("nan"); pc = pn = 1.0 if d.mean() == 0 else 0.0
            reps_c = sorted(set(oof[m1]) & set(oof[m2]))
            rng = random.Random(1)
            diffs = []
            for _ in range(a.boot):
                samp = [k2 for g in (rng.choice(gkeys) for _ in gkeys) for k2 in groups[g]]
                dd = []
                for rep in reps_c:
                    o1, o2 = oof[m1][rep], oof[m2][rep]
                    if len(o1) < len(secs) or len(o2) < len(secs):
                        continue
                    dd.append(np.mean([(o1[x]["pred_center"] == o1[x]["label"]) - (o2[x]["pred_center"] == o2[x]["label"]) for x in samp]))
                if dd:
                    diffs.append(np.mean(dd))
            ci = f"[{np.percentile(diffs,2.5)*100:+.2f}, {np.percentile(diffs,97.5)*100:+.2f}] pp" if diffs else "—"
            pair_rows.append(dict(setting=setting, m1=m1, m2=m2, n=len(common), mean_diff_pp=round(d.mean() * 100, 3),
                                  t_corrected=round(tc, 3), df=df, p_corrected=round(pc, 4), p_naive=round(pn, 4), ci=ci))
            R.append(f"| {NAMES[m1]} vs {NAMES[m2]} | {len(common)} | {d.mean()*100:+.2f} pp | {tc:.2f} | {df} | "
                     f"{pc:.3f} | {pn:.3f} | {ci} |")

    # 跨协议对比
    R.append("\n## 跨协议对比（同一模型，按 rep×折配对的 OA 均值）\n")
    for s1, s2, lab in (("grouped_orig", "slide_orig", "按名称分组 vs 按薄片随机（泄漏影响）"),
                        ("locality_orig", "grouped_orig", "按产地+名称分组 vs 按名称分组"),
                        ("grouped_norm", "grouped_orig", "分辨率归一化 vs 原流程")):
        R.append(f"**{lab}**\n")
        R.append(f"| 模型 | {s1} | {s2} | 差 |")
        R.append("|---|---|---|---|")
        for m in ORDER:
            if (s1, m) in runs and (s2, m) in runs:
                v1 = [np.mean([t["pred_center"] == t["label"] for t in r["test"]]) for r in runs[(s1, m)]]
                v2 = [np.mean([t["pred_center"] == t["label"] for t in r["test"]]) for r in runs[(s2, m)]]
                R.append(f"| {NAMES[m]} | {ms(v1)} | {ms(v2)} | {(np.mean(v1)-np.mean(v2))*100:+.2f} pp |")
        R.append("")

    # 固定划分
    fixed_rows, curves = [], []
    fsettings = sorted({s for s, _ in runs if s.startswith("fixed")})
    for setting in fsettings:
        R.append(f"\n## {setting}（原固定划分：训练 256 / 早停验证 32 / 测试 33）\n")
        R.append("| 模型 | 种子数 | 测试 OA | macro-F1 | 火成岩召回 | 变质岩召回 | 沉积岩召回 | 五裁剪 OA | 最佳 epoch |")
        R.append("|---|---:|---|---|---|---|---|---|---|")
        preds = {}
        for m in ORDER:
            rs = runs.get((setting, m))
            if not rs:
                continue
            mm = [metrics([t["label"] for t in r["test"]], [t["pred_center"] for t in r["test"]]) for r in rs]
            mu = [metrics([t["label"] for t in r["test"]], [t["pred_multi"] for t in r["test"]]) for r in rs]
            preds[m] = {r["job"]["seed"]: {t["section_key"]: t for t in r["test"]} for r in rs}
            row = dict(setting=setting, model=m, seeds=len(rs), oa=ms([x["oa"] for x in mm]),
                       f1=ms([x["macro_f1"] for x in mm], 3, False), rec_ign=ms([x["rec_igneous"] for x in mm], 3, False),
                       rec_met=ms([x["rec_metamorphic"] for x in mm], 3, False),
                       rec_sed=ms([x["rec_sedimentary"] for x in mm], 3, False), multi=ms([x["oa"] for x in mu]),
                       best_epoch=ms([r["best_epoch"] for r in rs], 1, False))
            fixed_rows.append(row)
            R.append(f"| {NAMES[m]} | {row['seeds']} | {row['oa']} | {row['f1']} | {row['rec_ign']} | {row['rec_met']} | "
                     f"{row['rec_sed']} | {row['multi']} | {row['best_epoch']} |")
            for r in rs:
                for h in r["history"]:
                    curves.append(dict(setting=setting, model=m, seed=r["job"]["seed"], best_epoch=r["best_epoch"], **h))
        R.append("\n**McNemar 精确检验（每个种子单独检验；b = 前者对后者错，c = 前者错后者对）**\n")
        R.append("| 比较 | 各种子 b/c | 各种子 p | p<0.05 的种子数 |")
        R.append("|---|---|---|---:|")
        for m1, m2 in [("dual", "ppl"), ("dual", "xpl0"), ("xpl0", "ppl"), ("dual", "dual_mis")] + PAIRS[3:]:
            if m1 not in preds or m2 not in preds:
                continue
            bc, ps = [], []
            for sd in sorted(set(preds[m1]) & set(preds[m2])):
                p1, p2 = preds[m1][sd], preds[m2][sd]
                b = sum(1 for k in p1 if p1[k]["pred_center"] == p1[k]["label"] and p2[k]["pred_center"] != p2[k]["label"])
                c = sum(1 for k in p1 if p1[k]["pred_center"] != p1[k]["label"] and p2[k]["pred_center"] == p2[k]["label"])
                bc.append(f"{b}/{c}")
                ps.append(mcnemar_exact(b, c))
            R.append(f"| {NAMES[m1]} vs {NAMES[m2]} | {' '.join(bc)} | {' '.join(f'{p:.3f}' for p in ps)} | "
                     f"{sum(p < 0.05 for p in ps)}/{len(ps)} |")

    def wcsv(name, rows):
        if rows:
            with open(os.path.join(a.out, name), "w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    wcsv("cv_summary.csv", cv_rows)
    wcsv("pairwise.csv", pair_rows)
    wcsv("fixed_summary.csv", fixed_rows)
    wcsv("curves_fixed.csv", curves)
    wcsv("oof_predictions.csv", oof_rows)
    with open(os.path.join(a.out, "report_R2.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(R))
    print("\n".join(R))


if __name__ == "__main__":
    main()
