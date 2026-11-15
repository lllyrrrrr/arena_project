#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze.py —— 统一口径下计算报告所需全部数字

主口径: 严格单轮 + 提问与两侧回答均非空 + 投票标签有效
  - 2024期: 全量(近似英文)
  - 2025期: language == 'en'
对照口径在各表内注明。

输入: results/battles_analysis.parquet  (prepare_data.py 产物)
输出: results/numbers_sheet.md        正文引用的数字清单
      results/numbers.json            同一份数字的机器可读版

用法: .venv/bin/python scripts/analyze.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TBL = ROOT / "results" / "battles_analysis.parquet"
OUT_MD = ROOT / "results" / "numbers_sheet.md"
OUT_JSON = ROOT / "results" / "numbers.json"

RNG = np.random.default_rng(20261114)
LINES = []
R = {}  # numbers.json


def w(s=""):
    LINES.append(s)


def pct(x, d=1):
    return f"{100*x:.{d}f}%"


def cluster_boot_rate(df, value_col, cluster_col, reps=2000):
    """按会话(或票)聚类重抽样,返回比例均值的95%区间"""
    g = df.groupby(cluster_col)[value_col].agg(["sum", "count"])
    sums, counts = g["sum"].to_numpy(), g["count"].to_numpy()
    n = len(g)
    idx = RNG.multinomial(n, np.ones(n) / n, size=reps)
    num = idx @ sums
    den = idx @ counts
    rates = num / den
    return np.nanpercentile(rates, [2.5, 97.5])


def longer_win_table(dec):
    """dec: 分胜负且不等长子集 -> pooled, 同对阵均值, CI(按session或行)"""
    lw = dec["longer_won"].astype(float)
    pooled = lw.mean()
    if "session_id" in dec and dec["session_id"].notna().any():
        ci = cluster_boot_rate(dec, "longer_won", "session_id")
    else:
        b = RNG.binomial(len(lw), lw.mean(), size=2000) / len(lw)
        ci = np.nanpercentile(b, [2.5, 97.5])
    pair_key = dec[["model_a", "model_b"]].apply(lambda r: tuple(sorted(r)), axis=1)
    ps = lw.groupby(pair_key).agg(["mean", "count"])
    big = ps[ps["count"] >= 30]
    return pooled, ci, big["mean"].mean(), len(big)


def main():
    df = pd.read_parquet(TBL)
    df["normdiff"] = (df.a_chars - df.b_chars) / (df.a_chars + df.b_chars)
    df["longer_won"] = ((df.a_chars > df.b_chars) & (df.winner == "model_a")) | \
                       ((df.b_chars > df.a_chars) & (df.winner == "model_b"))
    df["unresolved"] = ~df.decisive
    ok = df[df.single_turn & ~df.empty_side & (df.prompt_chars > 0)].copy()
    en25 = ok[(ok.period == "2025") & (ok.language == "en")].copy()
    y24 = ok[ok.period == "2024"].copy()
    all25 = ok[ok.period == "2025"].copy()
    dec = lambda d: d[d.decisive & ~d.equal_length] if "equal_length" in d else \
        d[d.decisive & (d.a_chars != d.b_chars)]

    d24, d25, dall = dec(y24), dec(en25), dec(all25)

    w("# Numbers sheet(主口径: 严格单轮 + 非空 + 有效标签;2025=英文)\n")

    # ---------- 0. 样本流转 ----------
    w("## 0. 样本流转\n")
    w("| 步骤 | 2024 | 2025英文 | 2025全语言 |")
    w("|---|---|---|---|")
    rows = [
        ("原始票", len(df[df.period == '2024']), len(df[df.period == '2025']), len(df[df.period == '2025'])),
        ("严格单轮", len(df[(df.period == '2024') & df.single_turn]),
         len(df[(df.period == '2025') & df.single_turn & (df.language == 'en')]),
         len(df[(df.period == '2025') & df.single_turn])),
        ("提问+两侧回答非空", len(y24), len(en25), len(all25)),
        ("分胜负", int(d24.size and d24.shape[0] and d24.shape[0] or d24.shape[0]) if False else len(y24[y24.decisive]),
         len(en25[en25.decisive]), len(all25[all25.decisive])),
        ("分胜负且不等长(胜率分母)", len(d24), len(d25), len(dall)),
    ]
    for r in rows:
        w(f"| {r[0]} | {r[1]:,} | {r[2]:,} | {r[3]:,} |")
    R["flow"] = {r[0]: r[1:] for r in rows}
    w("")

    # ---------- 1. 描述:回答形态两期漂移 ----------
    w("## 1. 回答形态(回答级,合并A/B两侧)\n")
    resp = lambda d: pd.concat([d["a_chars"], d["b_chars"]])
    ratio = lambda d: pd.concat([d.a_chars / d.prompt_chars, d.b_chars / d.prompt_chars])
    has = lambda d, c: pd.concat([d[f"a_{c}"], d[f"b_{c}"]]).gt(0).mean()
    for name, d in [("2024", y24), ("2025英文", en25), ("2025全语言", all25)]:
        R[f"desc_{name}"] = {
            "resp_median": float(resp(d).median()),
            "ratio_median": float(ratio(d).median()),
            "bold_share": float(has(d, "bold")),
            "head_share": float(has(d, "head")),
            "list_share": float(has(d, "list")),
            "prompt_median": float(d.prompt_chars.median()),
        }
        w(f"| {name} | 回答中位字符 {resp(d).median():,.0f} | 答/问比中位 {ratio(d).median():.1f} | "
          f"加粗 {pct(has(d,'bold'))} | 标题 {pct(has(d,'head'))} | 列表 {pct(has(d,'list'))} | 提问中位 {d.prompt_chars.median():.0f} |")
    w("")

    # 提问长度分档的增长倍数
    w("### 同档提问下回答增长倍数(2025英文/2024 中位数比)\n")
    w("| 提问字符档 | 2024回答中位 | 2025英文回答中位 | 倍数 |")
    w("|---|---|---|---|")
    bins = [(1, 30), (31, 100), (101, 300), (301, 1000), (1001, 10**9)]
    R["prompt_bins"] = []
    for lo, hi in bins:
        m24 = resp(y24[(y24.prompt_chars >= lo) & (y24.prompt_chars < hi)]).median()
        m25 = resp(en25[(en25.prompt_chars >= lo) & (en25.prompt_chars < hi)]).median()
        w(f"| {lo}–{hi if hi<10**9 else '∞'} | {m24:,.0f} | {m25:,.0f} | {m25/m24:.2f} |")
        R["prompt_bins"].append({"bin": f"{lo}-{hi}", "m2024": m24, "m2025en": m25, "ratio": m25 / m24})
    w("")

    # ---------- 2. 长度优势 ----------
    w("## 2. 更长一方胜率(分胜负且不等长)\n")
    w("| 口径 | 胜率分母 | 更长方胜率 | 95%CI | 同对阵均值(≥30票) |")
    w("|---|---|---|---|---|")
    for name, d, sess in [("2024", d24, False), ("2025英文", d25, True), ("2025全语言", dall, True)]:
        pooled, ci, within, npair = longer_win_table(d.assign(session_id=d.session_id.fillna(d.id) if sess else d.id))
        w(f"| {name} | {len(d):,} | {pct(pooled,2)} | [{pct(ci[0],2)},{pct(ci[1],2)}] | {pct(within,2)} ({npair}对) |")
        R[f"longer_win_{name}"] = {"n": len(d), "pooled": pooled, "ci": ci.tolist(), "within_pair": within}
    w("")

    # 剂量反应
    w("### 剂量反应:归一化长度差分档 vs 更长方胜率\n")
    w("| 长度差档(从小到大五档) | 2024 | 2025英文 |")
    w("|---|---|---|")
    R["dose"] = []
    q = [0, .2, .4, .6, .8, 1.0]
    edges25 = d25.normdiff.abs().quantile(q).to_numpy().copy(); edges25[-1] += 1e-9
    edges24 = d24.normdiff.abs().quantile(q).to_numpy().copy(); edges24[-1] += 1e-9
    for i in range(5):
        s25 = d25[(d25.normdiff.abs() >= edges25[i]) & (d25.normdiff.abs() < edges25[i+1])]
        s24 = d24[(d24.normdiff.abs() >= edges24[i]) & (d24.normdiff.abs() < edges24[i+1])]
        w(f"| Q{i+1} | {pct(s24.longer_won.mean(),2)} | {pct(s25.longer_won.mean(),2)} |")
        R["dose"].append({"q": i+1, "y2024": s24.longer_won.mean(), "en2025": s25.longer_won.mean()})
    w("")

    # 近等长时的格式效应
    w("### 长度接近(|归一化差|<0.1)时,格式更多一方的胜率\n")
    for name, d in [("2024", d24), ("2025英文", d25)]:
        close = d[d.normdiff.abs() < 0.1]
        fmt_a = close.a_bold + close.a_head + close.a_list
        fmt_b = close.b_bold + close.b_head + close.b_list
        diff = close[(fmt_a != fmt_b)]
        fa, fb = fmt_a[diff.index], fmt_b[diff.index]
        won = ((fa > fb) & (diff.winner == "model_a")) | ((fb > fa) & (diff.winner == "model_b"))
        w(f"| {name} | n={len(diff):,} | 格式更多方胜率 {pct(won.mean(),2)} |")
        R[f"format_only_{name}"] = {"n": len(diff), "rate": float(won.mean())}
    w("")

    # ---------- 3. 边界:任务/语言/会话 ----------
    w("## 3. 边界条件\n### 任务(2025英文)\n")
    w("| 分组 | n | 更长方胜率 |")
    w("|---|---|---|")
    for name, sub in [("数学", d25[d25.tag_math == True]), ("非数学", d25[d25.tag_math != True]),
                      ("涉及代码", d25[d25.is_code == True]), ("非代码", d25[d25.is_code != True]),
                      ("创意写作", d25[d25.tag_creative == True]), ("非创意写作", d25[d25.tag_creative != True]),
                      ("指令遵循", d25[d25.tag_if == True]), ("非指令遵循", d25[d25.tag_if != True])]:
        w(f"| {name} | {len(sub):,} | {pct(sub.longer_won.mean(),2)} |")
        R[f"task_{name}"] = {"n": len(sub), "rate": float(sub.longer_won.mean())}
    w("")

    w("### 语言(全语言单轮,各语言内部比较)\n")
    w("| 语言 | n | 更长方胜率 |")
    w("|---|---|---|")
    R["lang"] = {}
    for lang, g in dall.groupby("language"):
        if len(g) > 500:
            w(f"| {lang} | {len(g):,} | {pct(g.longer_won.mean(),2)} |")
            R["lang"][lang] = {"n": len(g), "rate": float(g.longer_won.mean())}
    w("")

    w("### 会话进程(≥3票会话内部)\n")
    w("| 口径 | order=1 | order=3 | 未分胜负率1→3 |")
    w("|---|---|---|---|")
    for name, base in [("英文", en25), ("全语言", all25)]:
        vpn = base.groupby("session_id").size()
        multi_ids = vpn[vpn >= 3].index
        m = base[base.session_id.isin(multi_ids)]
        md = m[m.decisive & (m.a_chars != m.b_chars)]
        o1 = md[md.eval_order == 1]; o3 = md[md.eval_order == 3]
        u1 = m[m.eval_order == 1].unresolved.mean(); u3 = m[m.eval_order == 3].unresolved.mean()
        w(f"| {name} | n={len(o1):,} {pct(o1.longer_won.mean(),2)} | n={len(o3):,} {pct(o3.longer_won.mean(),2)} | {pct(u1)}→{pct(u3)} |")
        R[f"session_{name}"] = {"o1": [len(o1), float(o1.longer_won.mean())],
                               "o3": [len(o3), float(o3.longer_won.mean())],
                               "unres13": [float(u1), float(u3)]}
    w("")

    # ---------- 4. 未分胜负票 ----------
    w("## 4. 未分胜负票机制(2025英文,按长度差五档)\n")
    w("| 档 | tie | both_bad |")
    w("|---|---|---|")
    R["unres_bins"] = []
    for i in range(5):
        s = en25[(en25.normdiff.abs() >= edges25[i]) & (en25.normdiff.abs() < edges25[i+1])]
        tie = (s.winner == "tie").mean(); bb = (s.winner == "both_bad").mean()
        w(f"| Q{i+1} | {pct(tie,2)} | {pct(bb,2)} |")
        R["unres_bins"].append({"q": i+1, "tie": tie, "both_bad": bb})
    bb_prompt = en25[en25.winner == "both_bad"].prompt_chars.median()
    dec_prompt = en25[en25.decisive].prompt_chars.median()
    w(f"\nboth_bad 提问中位字符 {bb_prompt:.0f} vs 分胜负 {dec_prompt:.0f}")
    R["bothbad_prompt"] = {"both_bad": float(bb_prompt), "decisive": float(dec_prompt)}
    w("")

    # ---------- 5. 预测对照 ----------
    w("## 5. 预测对照:模型身份 vs +形式特征(2025英文,按会话留出)\n")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, log_loss
        from sklearn.model_selection import GroupShuffleSplit
        from sklearn.preprocessing import OneHotEncoder

        dd = d25.copy()
        dd["y"] = (dd.winner == "model_a").astype(int)
        top = dd.model_a.value_counts().head(40).index.union(dd.model_b.value_counts().head(40).index)
        dd["ma"] = dd.model_a.where(dd.model_a.isin(top), "other")
        dd["mb"] = dd.model_b.where(dd.model_b.isin(top), "other")
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        Xid = enc.fit_transform(dd[["ma", "mb"]])
        Xf = np.c_[dd.normdiff.fillna(0),
                   np.log1p(dd.a_bold) - np.log1p(dd.b_bold),
                   np.log1p(dd.a_head) - np.log1p(dd.b_head),
                   np.log1p(dd.a_list) - np.log1p(dd.b_list),
                   np.log1p(dd.prompt_chars)]
        gss = GroupShuffleSplit(1, test_size=0.25, random_state=20261114)
        tr, te = next(gss.split(Xid, dd.y, groups=dd.session_id))
        res = {}
        for name, X in [("仅模型身份", Xid), ("身份+形式", np.hstack([Xid, Xf])), ("仅形式", Xf)]:
            m = LogisticRegression(max_iter=1000, C=1.0).fit(X[tr], dd.y.iloc[tr])
            p = m.predict_proba(X[te])[:, 1]
            res[name] = {"acc": float(accuracy_score(dd.y.iloc[te], p > .5)),
                         "logloss": float(log_loss(dd.y.iloc[te], p))}
            w(f"| {name} | acc {pct(res[name]['acc'],2)} | logloss {res[name]['logloss']:.4f} |")
        R["prediction"] = res
    except Exception as e:
        w(f"(预测对照失败: {e})")
    w("")

    # ---------- 6. 附:位置偏差 ----------
    pa = (d25.winner == "model_a").mean()
    w(f"## 6. 附:位置\nP(model_a胜 | 2025英文分胜负) = {pct(pa,2)} (n={len(d25):,})\n")
    R["position"] = float(pa)

    OUT_MD.write_text("\n".join(LINES), encoding="utf-8")
    OUT_JSON.write_text(json.dumps(R, ensure_ascii=False, indent=2, default=float))
    print("\n".join(LINES[:60]))
    print(f"\n... 已写出 {OUT_MD.name} / {OUT_JSON.name}")


if __name__ == "__main__":
    main()
