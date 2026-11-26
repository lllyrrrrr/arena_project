import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from pathlib import Path

DF = pd.read_parquet("results/features.parquet")
OUT = []
def p(s=""):
    print(s); OUT.append(str(s))


MIN_VOTES, C = 50, 1e8
W = {"t2": ("t2_100k", "2024-06-20", "2024-07-25"),
     "t3": ("t3_140k", "2025-06-20", "2025-07-25")}


def zdiff(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    s = a + b
    v = np.where(s > 0, (a - b) / s, 0.0)
    sd = v.std()
    return v / sd if sd > 0 else v


def build_model_matrix(ma, mb, active):
    n = len(ma)
    midx = {m: i for i, m in enumerate(active)}
    ia = pd.Series(ma).map(midx).to_numpy()
    ib = pd.Series(mb).map(midx).to_numpy()
    rows, cols, vals = [], [], []
    mask_a = ~np.isnan(ia)
    if mask_a.any():
        rows.append(np.nonzero(mask_a)[0])
        cols.append(ia[mask_a].astype(int))
        vals.append(np.ones(int(mask_a.sum())))
    mask_b = ~np.isnan(ib)
    if mask_b.any():
        rows.append(np.nonzero(mask_b)[0])
        cols.append(ib[mask_b].astype(int))
        vals.append(-np.ones(int(mask_b.sum())))
    return sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, len(active)))


def se_mean_iid_vs_cluster(vals, clusters):
    v = np.asarray(vals, dtype=float)
    n = len(v)
    mu = v.mean()
    se_iid = float(np.sqrt(mu * (1 - mu) / n))
    codes, uniques = pd.factorize(clusters, sort=False)
    g = len(uniques)
    u = np.bincount(codes, weights=v - mu, minlength=g)
    se_cl = float(np.sqrt((g / (g - 1)) * np.sum(u ** 2)) / n)
    return mu, se_iid, se_cl


def se_logit_iid_vs_cluster(X, y, clusters):
    """解析对比: 经典MLE信息阵逆(i.i.d.) vs Liang-Zeger聚类夹心标准误"""
    lr = LogisticRegression(fit_intercept=False, C=C,
                            tol=1e-6, max_iter=2000).fit(X, y)
    beta = lr.coef_[0]
    prob = lr.predict_proba(X)[:, 1]
    resid = y - prob
    w = prob * (1.0 - prob)
    H = (X.T @ X.multiply(w[:, None])).toarray() + (1.0 / C) * np.eye(X.shape[1])
    H_inv = np.linalg.inv(H)
    se_iid = np.sqrt(np.maximum(np.diag(H_inv), 0.0))
    codes, uniques = pd.factorize(clusters, sort=False)
    g = len(uniques)
    n, k = X.shape
    C_mat = sparse.csr_matrix(
        (np.ones(n), (codes, np.arange(n))), shape=(g, n))
    S = (C_mat @ X.multiply(resid[:, None])).toarray()
    scale = (g / (g - 1)) * ((n - 1) / (n - k))
    cov_cl = scale * (H_inv @ (S.T @ S) @ H_inv)
    se_cl = np.sqrt(np.maximum(np.diag(cov_cl), 0.0))
    return beta, se_iid, se_cl


for tag, (src, lo, hi) in W.items():
    d = DF[(DF.source == src) & (DF.timestamp >= lo) & (DF.timestamp < hi) &
           (DF.language == "en") & (DF.turns == 1) &
           (DF.q_chars > 0) & (DF.a_chars > 0) & (DF.b_chars > 0) &
           DF.winner.isin(["model_a", "model_b"])].reset_index(drop=True)
    cl_sizes = d.cluster_id.value_counts()
    p(f"\n===== {tag} n={len(d):,} clusters={len(cl_sizes):,} "
      f"mean_size={cl_sizes.mean():.2f} max_cluster={cl_sizes.max()} =====")

    # ---- 估计量1: 更长方胜率 (解析对比, 无Bootstrap噪声) ----
    e = d[d.a_chars != d.b_chars]
    wins = (e.a_chars.gt(e.b_chars)
            .eq(e.winner == "model_a")).astype(int).to_numpy()
    mu, se_i, se_c = se_mean_iid_vs_cluster(wins, e.cluster_id.to_numpy())
    p(f"更长方胜率 = {mu:.4f} (n={len(e):,})")
    p(f"  iid  SE={se_i:.4f} CI [{mu-1.96*se_i:.4f}, {mu+1.96*se_i:.4f}]")
    p(f"  聚类 SE={se_c:.4f} CI [{mu-1.96*se_c:.4f}, {mu+1.96*se_c:.4f}] design={se_c/se_i:.2f}")

    # ---- 估计量2: M3 全部四个风格系数 (满秩 + 去代码块口径, 解析对比) ----
    vc = pd.concat([d.model_a, d.model_b]).value_counts()
    keep = set(vc[vc >= MIN_VOTES].index)
    ma = d.model_a.where(d.model_a.isin(keep), "other").to_numpy()
    mb = d.model_b.where(d.model_b.isin(keep), "other").to_numpy()
    models = sorted(set(ma) | set(mb))
    ref = "other" if "other" in models else models[0]
    active = [m for m in models if m != ref]
    Xm = build_model_matrix(ma, mb, active)
    both_tok = (d.a_tok_off > 0) & (d.b_tok_off > 0)
    ta = np.where(both_tok, d.a_tok_off, d.a_chars)
    tb = np.where(both_tok, d.b_tok_off, d.b_chars)
    zcols = np.c_[zdiff(ta, tb), zdiff(d.clean_a_head, d.clean_b_head),
                  zdiff(d.clean_a_bold, d.clean_b_bold),
                  zdiff(d.clean_a_list, d.clean_b_list)]
    X = sparse.hstack([Xm, sparse.csr_matrix(zcols)]).tocsr()
    y = (d.winner == "model_a").astype(int).to_numpy()
    beta, se_i_all, se_c_all = se_logit_iid_vs_cluster(X, y, d.cluster_id.to_numpy())
    p("M3 风格系数 i.i.d. SE vs 聚类稳健 SE:")
    for i, name in enumerate(["len", "head", "bold", "list"]):
        b = beta[-4 + i]
        si, sc = se_i_all[-4 + i], se_c_all[-4 + i]
        p(f"  β_{name:4s}={b:+.3f} | iid SE={si:.4f} | 聚类 SE={sc:.4f} "
          f"CI[{b-1.96*sc:+.3f}, {b+1.96*sc:+.3f}] design={sc/si:.2f}")

Path("results/cluster_check.md").write_text("\n".join(OUT), encoding="utf-8")
print("\n→ results/cluster_check.md")
