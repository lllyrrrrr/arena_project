import pandas as pd
import numpy as np
from pathlib import Path

DF = pd.read_parquet("results/features.parquet")
OUT = []
def p(s=""):
    print(s); OUT.append(str(s))


def cluster_se_mean(vals, clusters):
    """按簇残差求和的夹心标准误（允许同簇多票相关）"""
    v = np.asarray(vals, dtype=float)
    n = len(v)
    if n == 0:
        return np.nan
    mu = v.mean()
    codes, uniques = pd.factorize(clusters, sort=False)
    g = len(uniques)
    if g <= 1:
        return np.nan
    u = np.bincount(codes, weights=v - mu, minlength=g)
    return float(np.sqrt((g / (g - 1)) * np.sum(u ** 2)) / n)


def wr(vals, clusters):
    """胜率% + 聚类 95% CI(±半宽) 紧凑打印"""
    v = np.asarray(vals, dtype=float)
    n = len(v)
    if n == 0:
        return "n=0"
    r = v.mean()
    se = cluster_se_mean(v, clusters)
    hw = 1.96 * se
    g = pd.Series(clusters).nunique()
    return f"{r*100:.1f}%±{hw*100:.1f} (n={n:,}, G={g:,})"


# 主样本口径: 同月窗口(6.20-7.24) ∩ 英文 ∩ 严格单轮 ∩ 文本非空
W = {"t2_100k": ("2024-06-20", "2024-07-25"), "t3_140k": ("2025-06-20", "2025-07-25")}
def pmask(src, lo, hi):
    return (DF.source == src) & (DF.timestamp >= lo) & (DF.timestamp < hi)


base, dec = {}, {}
p("== 样本流 (逐级严格闭合) ==")
for src, (lo, hi) in W.items():
    d = DF[pmask(src, lo, hi) & (DF.language == "en") & (DF.turns == 1) &
           (DF.q_chars > 0) & (DF.a_chars > 0) & (DF.b_chars > 0)]
    dv = d[d.winner.isin(["model_a", "model_b"])]
    e = dv[dv.a_chars != dv.b_chars]
    base[src], dec[src] = d, e
    n_tie = int((d.winner == "tie").sum())
    n_bad = int((d.winner == "both_bad").sum())
    eq_dec = int((dv.a_chars == dv.b_chars).sum())
    assert len(d) - n_tie - n_bad == len(dv) and len(dv) - eq_dec == len(e)
    p(f"{src[:2]} 主样本{len(d):,} (G={d.cluster_id.nunique():,}) "
      f"→ 剔未分胜负{n_tie+n_bad:,}(平局{n_tie:,}/双差{n_bad:,}) "
      f"→ 分胜负{len(dv):,} → 剔分胜负等长{eq_dec:,} → 胜率分母{len(e):,}")

# 1. 漂移: 回答/提问长度中位数 + 去代码块后三类Markdown标记使用率(两侧回答合并算)
p("\n== 1.形态漂移 (去代码块Markdown口径) ==")
for src, d in base.items():
    c = pd.concat([d.a_chars, d.b_chars])
    f_clean = lambda col: (pd.concat([d[f"clean_a_{col}"], d[f"clean_b_{col}"]]) > 0).mean() * 100
    f_raw = lambda col: (pd.concat([d[f"a_{col}"], d[f"b_{col}"]]) > 0).mean() * 100
    p(f"{src[:2]} 答{c.median():.0f}字 问{d.q_chars.median():.0f}字 | "
      f"加粗{f_clean('bold'):.0f}% 标题{f_clean('head'):.0f}% 列表{f_clean('list'):.0f}% "
      f"(未清洗标题{f_raw('head'):.0f}%)")

# 2. 主效应: 分胜负票中更长方胜率 (聚类 95% CI)
p("\n== 2.更长方胜率 (聚类 95% CI) ==")
for src, e in dec.items():
    longer = np.where(e.a_chars > e.b_chars, "a", "b")
    win = (e.winner == ("model_" + longer)).astype(int)
    p(f"{src[:2]} 字符口径 {wr(win, e.cluster_id)}")
    # 同票字符 vs 官方token稳健性核验
    tok_ok = (e.a_tok_off > 0) & (e.b_tok_off > 0) & (e.a_tok_off != e.b_tok_off)
    ep = e[tok_ok]
    w_char = (ep.a_chars > ep.b_chars).eq(ep.winner == "model_a").astype(int)
    w_tok = (ep.a_tok_off > ep.b_tok_off).eq(ep.winner == "model_a").astype(int)
    flip = (ep.a_chars > ep.b_chars).ne(ep.a_tok_off > ep.b_tok_off).mean() * 100
    diff = w_tok - w_char
    se_d = cluster_se_mean(diff, ep.cluster_id)
    p(f"   同票token对照(n={len(ep):,}): 换边率{flip:.1f}% | "
      f"token胜率{w_tok.mean()*100:.1f}% vs 字符{w_char.mean()*100:.1f}% "
      f"(差{diff.mean()*100:+.2f}±{1.96*se_d*100:.2f}pp)")

# 3. 剂量反应: 按 |a-b|/均值 分5档,看更长方胜率梯度 (聚类 95% CI)
p("\n== 3.剂量反应 (聚类 95% CI) ==")
for src, e in dec.items():
    diff = (e.a_chars - e.b_chars).abs() / ((e.a_chars + e.b_chars) / 2)
    longer = np.where(e.a_chars > e.b_chars, "a", "b")
    win = (e.winner == ("model_" + longer)).astype(int)
    tmp = pd.DataFrame({"bin": pd.qcut(diff, 5, duplicates="drop"),
                        "w": win, "cl": e.cluster_id.to_numpy()})
    parts = []
    for i, g in tmp.groupby("bin", observed=True):
        m = g.w.mean()
        hw = 1.96 * cluster_se_mean(g.w, g.cl)
        parts.append(f"差≤{i.right:.2f}:{m*100:.1f}%±{hw*100:.1f}(n={len(g):,})")
    p(f"{src[:2]} " + " | ".join(parts))

# 4. 任务边界: 该类任务 vs 其他 的更长方胜率 (聚类 95% CI)
p("\n== 4.任务边界 (聚类 95% CI) ==")
def wr2(d, col):
    e = d[d.winner.isin(["model_a", "model_b"]) & (d.a_chars != d.b_chars)]
    longer = np.where(e.a_chars > e.b_chars, "a", "b")
    w = (e.winner == ("model_" + longer)).astype(int)
    m = e[col] == True
    return f"{wr(w[m], e.cluster_id[m])} vs 其他 {wr(w[~m], e.cluster_id[~m])}"


for src, d in base.items():
    p(f"{src[:2]} 数学 {wr2(d, 'tag_math')}")
    p(f"{src[:2]} 指令 {wr2(d, 'tag_if')}")
    p(f"{src[:2]} 代码 {wr2(d, 'is_code')}")
p(f"t3 写作 {wr2(base['t3_140k'], 'tag_writing')} (仅期内)")

# 5. 未分胜负构成 + 位置公平性(A侧应≈50%, 聚类 95% CI)
p("\n== 5.平局与位置 (聚类 95% CI) ==")
for src, d in base.items():
    dv = d[d.winner.isin(["model_a", "model_b"])]
    a_win = (dv.winner == "model_a").astype(int)
    p(f"{src[:2]} tie{(d.winner=='tie').mean()*100:.1f}% bothbad{(d.winner=='both_bad').mean()*100:.1f}% | "
      f"A侧胜率 {wr(a_win, dv.cluster_id)}")

# 6. 语言边界: 全语言口径(不限英文), top8语言的更长方胜率 (聚类 95% CI)
p("\n== 6.语言(全语言口径, 聚类 95% CI) ==")
for src, (lo, hi) in W.items():
    d = DF[pmask(src, lo, hi) & (DF.turns == 1) & (DF.q_chars > 0) &
           (DF.a_chars > 0) & (DF.b_chars > 0) &
           DF.winner.isin(["model_a", "model_b"]) & (DF.a_chars != DF.b_chars)]
    longer = np.where(d.a_chars > d.b_chars, "a", "b")
    d = d.assign(wl=(d.winner == ("model_" + longer)).astype(int))
    top_langs = d.language.value_counts().head(8).index
    parts = []
    for lang in top_langs:
        g = d[d.language == lang]
        m = g.wl.mean()
        hw = 1.96 * cluster_se_mean(g.wl, g.cluster_id)
        parts.append(f"{lang}:{m*100:.1f}%±{hw*100:.1f}(n={len(g):,})")
    p(f"{src[:2]} " + " | ".join(parts))

Path("results/main_stats.md").write_text("\n".join(OUT), encoding="utf-8")
print("\n→ results/main_stats.md")
