import pandas as pd
import numpy as np
from scipy import sparse
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from pathlib import Path

DF = pd.read_parquet("results/features.parquet")
OUT = []
def p(s=""):
    print(s); OUT.append(str(s))


MIN_VOTES, C_BIG = 50, 1e8
W = {"t2": ("t2_100k", "2024-06-20", "2024-07-25"),
     "t3": ("t3_140k", "2025-06-20", "2025-07-25")}
CRIT_NAMES = [
    "complexity", "creativity", "domain_knowledge", "problem_solving",
    "real_world", "specificity", "technical_accuracy",
]
CRIT_COLS = [f"crit_{c}" for c in CRIT_NAMES]
TASK_COLS = ["tag_math", "is_code", "tag_if"]
MD = ["head", "bold", "list"]


def cluster_diff_means(y1, cl1, y2, cl2):
    """两组均值差 mean(y1)-mean(y2) 的按簇合并夹心标准误"""
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    m1, m2 = y1.mean(), y2.mean()
    groups = pd.Index(pd.unique(np.concatenate([np.asarray(cl1), np.asarray(cl2)])))
    g = len(groups)
    inf = (np.bincount(groups.get_indexer(cl1), weights=(y1 - m1) / len(y1), minlength=g)
           - np.bincount(groups.get_indexer(cl2), weights=(y2 - m2) / len(y2), minlength=g))
    se = float(np.sqrt((g / (g - 1)) * np.sum(inf ** 2)))
    diff = float(m1 - m2)
    pv = float(2 * norm.sf(abs(diff) / se)) if se > 0 else 1.0
    return diff, se, pv, g


def build_undirected_model_matrix(ma, mb, active):
    """无方向模型暴露矩阵 (X_A + X_B), 删去基准组(other)防止与截距项完全共线"""
    n = len(ma)
    midx = {m: i for i, m in enumerate(active)}
    ia = pd.Series(ma).map(midx).to_numpy()
    ib = pd.Series(mb).map(midx).to_numpy()
    rows, cols, vals = [], [], []
    for arr in (ia, ib):
        mask = ~np.isnan(arr)
        if mask.any():
            rows.append(np.nonzero(mask)[0])
            cols.append(arr[mask].astype(int))
            vals.append(np.ones(int(mask.sum())))
    return sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, len(active)))


def fit_logit_cluster(X, y, clusters, c_reg=C_BIG):
    """满秩逻辑回归 + Liang-Zeger 聚类稳健夹心标准误"""
    X_csr = sparse.csr_matrix(X)
    lr = LogisticRegression(fit_intercept=False, C=c_reg,
                            tol=1e-6, max_iter=2000).fit(X_csr, y)
    beta = lr.coef_[0]
    prob = lr.predict_proba(X_csr)[:, 1]
    resid = y - prob
    w = prob * (1.0 - prob)
    H = (X_csr.T @ X_csr.multiply(w[:, None])).toarray() + (1.0 / c_reg) * np.eye(X_csr.shape[1])
    H_inv = np.linalg.inv(H)
    codes, uniques = pd.factorize(clusters, sort=False)
    g = len(uniques)
    n, k = X_csr.shape
    C_mat = sparse.csr_matrix((np.ones(n), (codes, np.arange(n))), shape=(g, n))
    S = (C_mat @ X_csr.multiply(resid[:, None])).toarray()
    scale = (g / (g - 1)) * ((n - 1) / (n - k))
    cov = scale * (H_inv @ (S.T @ S) @ H_inv)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return beta, se


def or_row(period, outcome, spec, term, b, s, n, g):
    pv = float(2 * norm.sf(abs(b) / s)) if s > 0 else 1.0
    return {
        "period": period, "outcome": outcome, "spec": spec, "term": term,
        "beta": float(b), "se": float(s),
        "or": float(np.exp(b)),
        "or_lo": float(np.exp(b - 1.96 * s)),
        "or_hi": float(np.exp(b + 1.96 * s)),
        "p": pv, "n": int(n), "clusters": int(g),
    }


grid_rows = []
or_rows = []
crit_rate_rows = []

for tag, (src, lo, hi) in W.items():
    # 主样本四票型全量纳入: 同月窗口 ∩ 英文 ∩ 单轮 ∩ 三段非空
    d = DF[(DF.source == src) & (DF.timestamp >= lo) & (DF.timestamp < hi) &
           (DF.language == "en") & (DF.turns == 1) &
           (DF.q_chars > 0) & (DF.a_chars > 0) & (DF.b_chars > 0)].reset_index(drop=True)
    p(f"\n{'='*12} {tag} 全四票型主样本 n={len(d):,} (G={d.cluster_id.nunique():,}) {'='*12}")

    # ===== [1] 近组 vs 远组的四票型分流 (聚类 Wald 检验) =====
    gap = (d.a_chars - d.b_chars).abs() / ((d.a_chars + d.b_chars) / 2)
    near = d[gap <= 0.10]
    far = d[gap > 0.60]
    p("\n[1] 相对长度差近组(≤10%) vs 远组(>60%)四票型构成:")
    for lab, sub in [("近组 ≤10%", near), ("远组 >60%", far)]:
        p(f"  {lab:10s} n={len(sub):6,} | A胜{(sub.winner=='model_a').mean()*100:5.1f}% "
          f"B胜{(sub.winner=='model_b').mean()*100:5.1f}% "
          f"平局{(sub.winner=='tie').mean()*100:5.1f}% "
          f"双差{(sub.winner=='both_bad').mean()*100:5.1f}%")
    for lab, fn in [
        ("平局率 (远−近)", lambda x: (x.winner == "tie").astype(int)),
        ("双差率 (远−近)", lambda x: (x.winner == "both_bad").astype(int)),
        ("平局减双差 (远−近)", lambda x: (x.winner == "tie").astype(int) - (x.winner == "both_bad").astype(int)),
    ]:
        diff, se, pv, g_u = cluster_diff_means(fn(far), far.cluster_id, fn(near), near.cluster_id)
        p(f"  {lab:18s}: {diff*100:+6.2f}pp [{(diff-1.96*se)*100:+6.2f}, {(diff+1.96*se)*100:+6.2f}] "
          f"p={pv:.2e} (G={g_u:,})")

    # ===== [2] 长度差 × 用词重合度(Jaccard) 双终点分层网格 =====
    p("\n[2] 长度差 × 用词重合度(Jaccard) 双终点网格 (tie% / both_bad%):")
    dw = d[d.word_overlap.notna()].copy()
    dw["gap_bin"] = pd.cut(
        (dw.a_chars - dw.b_chars).abs() / ((dw.a_chars + dw.b_chars) / 2),
        bins=[-0.01, 0.10, 0.60, 2.01], labels=["≤10%", "10%–60%", ">60%"])
    dw["ov_bin"] = pd.cut(
        dw.word_overlap, bins=[-0.01, 0.15, 0.30, 0.50, 1.01],
        labels=["≤0.15", "0.15–0.30", "0.30–0.50", ">0.50"])
    for ov_lab in ["≤0.15", "0.15–0.30", "0.30–0.50", ">0.50"]:
        cells = []
        for g_lab in ["≤10%", "10%–60%", ">60%"]:
            sub = dw[(dw.ov_bin == ov_lab) & (dw.gap_bin == g_lab)]
            n_c = len(sub)
            r_t = float((sub.winner == "tie").mean()) if n_c else np.nan
            r_b = float((sub.winner == "both_bad").mean()) if n_c else np.nan
            grid_rows.append({
                "period": tag, "overlap_bin": ov_lab, "gap_bin": g_lab,
                "n": n_c, "tie_rate": r_t, "both_bad_rate": r_b,
            })
            cells.append(f"{g_lab}: tie {r_t*100:4.1f}% / bad {r_b*100:4.1f}% (n={n_c:,})")
        p(f"  Jaccard {ov_lab:9s} | " + " ; ".join(cells))

    # ===== [3] 双终点(tie vs both_bad) 聚类稳健 Logistic 回归 =====
    req = CRIT_COLS + TASK_COLS + ["word_overlap"]
    g = d.loc[d[req].notna().all(axis=1)].reset_index(drop=True)
    cl = g.cluster_id.to_numpy()
    n_g = len(g)
    n_cl = int(g.cluster_id.nunique())

    # 七维提问属性出现率汇总 (供雷达图)
    for cname, col in zip(CRIT_NAMES, CRIT_COLS):
        crit_rate_rows.append({
            "period": tag, "criterion": cname,
            "rate": float(g[col].astype(float).mean()), "n": n_g,
        })

    # 构造满秩基础设计矩阵: 截距 + 无方向模型暴露(删基准列) + 长度/提问/任务控制项
    vc = pd.concat([g.model_a, g.model_b]).value_counts()
    keep = set(vc[vc >= MIN_VOTES].index)
    ma = g.model_a.where(g.model_a.isin(keep), "other").to_numpy()
    mb = g.model_b.where(g.model_b.isin(keep), "other").to_numpy()
    models = sorted(set(ma) | set(mb))
    ref = "other" if "other" in models else models[0]
    active = [m for m in models if m != ref]
    Xm_u = build_undirected_model_matrix(ma, mb, active)

    gap_g = ((g.a_chars - g.b_chars).abs() / ((g.a_chars + g.b_chars) / 2)).to_numpy()
    log_avg = np.log1p(((g.a_chars + g.b_chars) / 2).to_numpy())
    log_q = np.log1p(g.q_chars.to_numpy())
    tasks = g[TASK_COLS].astype(float).to_numpy()
    X_base = sparse.hstack([
        sparse.csr_matrix(np.ones((n_g, 1))),
        Xm_u,
        sparse.csr_matrix(np.column_stack([gap_g, log_avg, log_q, tasks])),
    ]).tocsr()

    flags = g[CRIT_COLS].astype(float).to_numpy()
    hard_cnt = flags.sum(axis=1)
    # 去代码块后三类格式计数的 log1p 绝对差均值
    fmt_dist = np.mean([
        np.abs(np.log1p(g[f"clean_a_{k}"].to_numpy(dtype=float))
               - np.log1p(g[f"clean_b_{k}"].to_numpy(dtype=float)))
        for k in MD
    ], axis=0)
    jac10 = g.word_overlap.to_numpy(dtype=float) * 10.0  # 每增加 0.10 Jaccard

    # 需求属性数分组 (0 为参照组)
    bands = pd.cut(hard_cnt, bins=[-1, 0, 2, 4, 7], labels=["0", "1–2", "3–4", "5–7"])
    p(f"\n[3] 完整协变量回归样本 n={n_g:,} (G={n_cl:,}); 需求属性数分层观察率:")
    for blab, sub in g.assign(band=bands).groupby("band", observed=True):
        p(f"  属性数 {str(blab):3s} (n={len(sub):6,}): "
          f"平局{(sub.winner=='tie').mean()*100:5.1f}% | 双差{(sub.winner=='both_bad').mean()*100:5.1f}%")

    band_dum = pd.get_dummies(bands).drop(columns="0").to_numpy(dtype=float)

    for outcome, olab in [("tie", "平局"), ("both_bad", "双差")]:
        y_out = (g.winner == outcome).astype(int).to_numpy()
        p(f"\n  --- 结局: {olab} ({outcome}) [聚类稳健 OR 与 95% CI] ---")

        # 规格 A: 7 项提问属性 + 3 项任务
        X_a = sparse.hstack([X_base, sparse.csr_matrix(flags)]).tocsr()
        ba, sea = fit_logit_cluster(X_a, y_out, cl)
        p("  [A] 七维提问属性 (控制模型暴露、长度差、篇幅与任务):")
        for j, cname in enumerate(CRIT_NAMES):
            row = or_row(tag, outcome, "criteria", cname, ba[-7 + j], sea[-7 + j], n_g, n_cl)
            or_rows.append(row)
            p(f"    {cname:20s} OR={row['or']:.3f} [{row['or_lo']:.3f}, {row['or_hi']:.3f}] p={row['p']:.2e}")
        for j, tname in enumerate(TASK_COLS):
            row = or_row(tag, outcome, "task", tname, ba[-10 + j], sea[-10 + j], n_g, n_cl)
            or_rows.append(row)

        # 规格 B: 用词重合度(每+0.1) + 格式距离 (控制属性总数与上述全部基准项)
        X_b = sparse.hstack([
            X_base, sparse.csr_matrix(np.column_stack([hard_cnt, jac10, fmt_dist]))
        ]).tocsr()
        bb, seb = fit_logit_cluster(X_b, y_out, cl)
        p("  [B] 文本相似度与格式距离 (控制模型暴露、长度、任务与属性总数):")
        for term, idx_c in [("jaccard_per_0.1", -2), ("format_dist", -1)]:
            row = or_row(tag, outcome, "similarity", term, bb[idx_c], seb[idx_c], n_g, n_cl)
            or_rows.append(row)
            p(f"    {term:20s} OR={row['or']:.3f} [{row['or_lo']:.3f}, {row['or_hi']:.3f}] p={row['p']:.2e}")

        # 规格 C: 需求属性总数分档 (相对 0 项)
        X_c = sparse.hstack([X_base, sparse.csr_matrix(band_dum)]).tocsr()
        bc, sec = fit_logit_cluster(X_c, y_out, cl)
        for j, blab in enumerate(["1–2", "3–4", "5–7"]):
            row = or_row(tag, outcome, "criteria_band", f"band_{blab}", bc[-3 + j], sec[-3 + j], n_g, n_cl)
            or_rows.append(row)

pd.DataFrame(grid_rows).to_csv("results/undecided_grid.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(or_rows).to_csv("results/undecided_or.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(crit_rate_rows).to_csv("results/undecided_criteria_rates.csv", index=False, encoding="utf-8-sig")
Path("results/undecided_votes.md").write_text("\n".join(OUT), encoding="utf-8")
print("\n→ results/undecided_votes.md + results/undecided_grid/or/criteria_rates.csv")
