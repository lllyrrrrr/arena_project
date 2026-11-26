import pandas as pd
import numpy as np
from scipy import sparse
from scipy.stats import norm, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path

DF = pd.read_parquet("results/features.parquet")
OUT = []
def p(s=""):
    print(s); OUT.append(str(s))


MIN_VOTES = 50
C_BIG = 1e8
B = 500                     # 聚类 Bootstrap 次数 (用于模型排名区间)
SEED = 42
rng = np.random.default_rng(SEED)

W = {"t2": ("t2_100k", "2024-06-20", "2024-07-25"),
     "t3": ("t3_140k", "2025-06-20", "2025-07-25")}


def rel_diff(a, b):
    """官方口径有符号相对差: (A-B)/(A+B), 分母0记0"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    s = a + b
    return np.where(s > 0, (a - b) / s, 0.0)


def scale_sd(v, train_idx=None):
    sd = np.std(v if train_idx is None else v[train_idx])
    return v / sd if sd > 0 else v


def build_model_matrix(ma, mb, active):
    """满秩 BT 模型矩阵: 删去基准组(other)消除严格共线性"""
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


def fit_logit_cluster(X, y, clusters, c_reg=C_BIG):
    """满秩逻辑回归 + Liang-Zeger 聚类稳健夹心标准误"""
    lr = LogisticRegression(fit_intercept=False, C=c_reg,
                            tol=1e-6, max_iter=2000).fit(X, y)
    beta = lr.coef_[0]
    prob = lr.predict_proba(X)[:, 1]
    resid = y - prob
    w = prob * (1.0 - prob)
    # Bread: H = X^T W X + (1/C) I (满秩严格正定)
    Xw = X.multiply(w[:, None])
    H = (X.T @ Xw).toarray() + (1.0 / c_reg) * np.eye(X.shape[1])
    H_inv = np.linalg.inv(H)
    # Meat: 按簇求和得分 S_g = sum_{i in g} r_i x_i
    codes, uniques = pd.factorize(clusters, sort=False)
    g = len(uniques)
    n, k = X.shape
    C_mat = sparse.csr_matrix(
        (np.ones(n), (codes, np.arange(n))), shape=(g, n))
    S = (C_mat @ X.multiply(resid[:, None])).toarray()
    scale = (g / (g - 1)) * ((n - 1) / (n - k))
    cov = scale * (H_inv @ (S.T @ S) @ H_inv)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return lr, beta, se


def cluster_se_mean(vals, clusters):
    v = np.asarray(vals, dtype=float)
    n = len(v)
    mu = v.mean()
    codes, uniques = pd.factorize(clusters, sort=False)
    g = len(uniques)
    u = np.bincount(codes, weights=v - mu, minlength=g)
    return float(np.sqrt((g / (g - 1)) * np.sum(u ** 2)) / n)


def restore_theta(coef, active, models):
    """还原含基准组(0)的完整实力向量并均值中心化"""
    th = pd.Series(0.0, index=models)
    th.loc[active] = coef[:len(active)]
    return th - th.mean()


for tag, (src, lo, hi) in W.items():
    d = DF[(DF.source == src) & (DF.timestamp >= lo) & (DF.timestamp < hi) &
           (DF.language == "en") & (DF.turns == 1) &
           (DF.q_chars > 0) & (DF.a_chars > 0) & (DF.b_chars > 0) &
           DF.winner.isin(["model_a", "model_b"])].reset_index(drop=True)
    cl = d.cluster_id.to_numpy()
    n = len(d)
    p(f"\n===== {tag}: n={n:,} (clusters={d.cluster_id.nunique():,}) =====")

    # 模型列 (删去基准组消除共线性)
    vc = pd.concat([d.model_a, d.model_b]).value_counts()
    keep = set(vc[vc >= MIN_VOTES].index)
    ma = d.model_a.where(d.model_a.isin(keep), "other").to_numpy()
    mb = d.model_b.where(d.model_b.isin(keep), "other").to_numpy()
    models = sorted(set(ma) | set(mb))
    ref = "other" if "other" in models else models[0]
    active = [m for m in models if m != ref]
    Xm = build_model_matrix(ma, mb, active)
    y = (d.winner == "model_a").astype(int).to_numpy()

    # 风格特征: 成对优先官方token(双侧有效才用token,否则双侧同时回填字符,避免单侧量纲混拼)
    both_tok = (d.a_tok_off > 0) & (d.b_tok_off > 0)
    ta = np.where(both_tok, d.a_tok_off, d.a_chars)
    tb = np.where(both_tok, d.b_tok_off, d.b_chars)
    raw_r = {
        "len":  rel_diff(ta, tb),
        "head": rel_diff(d.clean_a_head, d.clean_b_head),
        "bold": rel_diff(d.clean_a_bold, d.clean_b_bold),
        "list": rel_diff(d.clean_a_list, d.clean_b_list),
    }
    z = {k: scale_sd(v) for k, v in raw_r.items()}
    def feats(*ks):
        return sparse.csr_matrix(np.c_[tuple(z[k] for k in ks)]) if ks else None

    # 四套消融 (满秩 + 聚类稳健SE)
    fits = {}
    for name, ks in [("M0", ()), ("M1", ("len",)),
                     ("M2", ("head", "bold", "list")),
                     ("M3", ("len", "head", "bold", "list"))]:
        X = Xm if not ks else sparse.hstack([Xm, feats(*ks)]).tocsr()
        lr, beta, se = fit_logit_cluster(X, y, cl)
        fits[name] = (lr, beta, se)

    # M3 系数 + 精确聚类 95% CI + OR + Wald p
    _, c3, se3 = fits["M3"]
    p("M3 标准化风格系数 (去代码块清洗口径, 聚类稳健 95% CI):")
    for i, k in enumerate(["len", "head", "bold", "list"]):
        b = c3[-4 + i]
        s = se3[-4 + i]
        lo_, hi_ = b - 1.96 * s, b + 1.96 * s
        pv = 2 * norm.sf(abs(b) / s) if s > 0 else 1.0
        p(f"  β_{k:4s} = {b:+.3f} (SE={s:.4f}) [{lo_:+.3f}, {hi_:+.3f}] "
          f"OR={np.exp(b):.3f} p={pv:.2e}")

    # 未清洗原始正则口径敏感性对照 (一行核对)
    z_unclean = np.c_[z["len"],
                      scale_sd(rel_diff(d.a_head, d.b_head)),
                      scale_sd(rel_diff(d.a_bold, d.b_bold)),
                      scale_sd(rel_diff(d.a_list, d.b_list))]
    _, c_unc, _ = fit_logit_cluster(
        sparse.hstack([Xm, sparse.csr_matrix(z_unclean)]).tocsr(), y, cl)
    p("  (未去代码块对照: " + " ".join(
        f"{k}={v:+.3f}" for k, v in zip(["len", "head", "bold", "list"], c_unc[-4:])) + ")")

    # 纯字符长度口径敏感性对照 (核对系数与 M3 模型排名秩相关)
    z_char = np.c_[scale_sd(rel_diff(d.a_chars, d.b_chars)),
                   z["head"], z["bold"], z["list"]]
    _, c_chr, se_chr = fit_logit_cluster(
        sparse.hstack([Xm, sparse.csr_matrix(z_char)]).tocsr(), y, cl)
    th3_tok = restore_theta(c3, active, models).drop("other", errors="ignore")
    th3_chr = restore_theta(c_chr, active, models).drop("other", errors="ignore")
    rho_chr = spearmanr(th3_tok.rank(ascending=False), th3_chr.rank(ascending=False)).statistic
    p("  (纯字符长度对照: " + " ".join(
        f"{k}={v:+.3f}±{1.96*s:.3f}" for k, v, s in zip(
            ["len", "head", "bold", "list"], c_chr[-4:], se_chr[-4:]))
      + f" | M3排名ρ={rho_chr:.4f}, token双侧有效率={both_tok.mean()*100:.2f}%)")

    # 四阶消融排名 (均值中心化后排名)
    th = {name: restore_theta(fits[name][1], active, models) for name in ["M0", "M1", "M2", "M3"]}
    rk_s = {name: th[name].rank(ascending=False) for name in ["M0", "M1", "M2", "M3"]}
    r0, r1, r2, r3 = rk_s["M0"], rk_s["M1"], rk_s["M2"], rk_s["M3"]
    abl = pd.DataFrame({"r0": r0, "r1": r1, "r2": r2, "r3": r3})
    movers = (r0 - r3).abs().drop("other", errors="ignore").nlargest(8).index
    p("\n消融排名: M0无控制→M1控长→M2控格式→M3都控")
    for m in sorted(movers, key=lambda m: r0[m]):
        r = abl.loc[m]
        p(f"  {m[:36]:36s} {int(r.r0):>2d} → {int(r.r1):>2d} → "
          f"{int(r.r2):>2d} → {int(r.r3):>2d}")

    # 整簇留出预测检验 (GroupShuffleSplit 75%/25%, 解释—预测悖论)
    tr, te = next(GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=SEED).split(d, y, cl))
    assert set(cl[tr]).isdisjoint(set(cl[te]))
    vc_tr = pd.concat([d.model_a.iloc[tr], d.model_b.iloc[tr]]).value_counts()
    keep_tr = set(vc_tr[vc_tr >= MIN_VOTES].index)
    ma_tr = d.model_a.where(d.model_a.isin(keep_tr), "other").to_numpy()
    mb_tr = d.model_b.where(d.model_b.isin(keep_tr), "other").to_numpy()
    models_tr = sorted(set(ma_tr[tr]) | set(mb_tr[tr]))
    ref_tr = "other" if "other" in models_tr else models_tr[0]
    active_tr = [m for m in models_tr if m != ref_tr]
    Xm_ho = build_model_matrix(ma_tr, mb_tr, active_tr)
    z_ho = {k: scale_sd(v, train_idx=tr) for k, v in raw_r.items()}
    den_ho = {
        k: scale_sd(rel_diff(d[f"clean_a_{k}"] / d.a_chars * 1000,
                             d[f"clean_b_{k}"] / d.b_chars * 1000), train_idx=tr)
        for k in ["head", "bold", "list"]
    }
    def test_logloss(extra_cols):
        X_all = Xm_ho if not extra_cols else sparse.hstack(
            [Xm_ho, sparse.csr_matrix(np.column_stack(extra_cols))]).tocsr()
        clf = LogisticRegression(fit_intercept=False, C=1e4,
                                 tol=1e-6, max_iter=1000).fit(X_all[tr], y[tr])
        pr = np.clip(clf.predict_proba(X_all[te])[:, 1], 1e-10, 1 - 1e-10)
        return -np.where(y[te] == 1, np.log(pr), np.log1p(-pr))

    ll_m0 = test_logloss([])
    ll_m1 = test_logloss([z_ho["len"]])
    ll_m3 = test_logloss([z_ho["len"], z_ho["head"], z_ho["bold"], z_ho["list"]])
    ll_m3d = test_logloss([z_ho["len"], den_ho["head"], den_ho["bold"], den_ho["list"]])
    cl_te = cl[te]
    p(f"\n整簇留出预测检验 (test n={len(te):,}, G={pd.Series(cl_te).nunique():,}):")
    p(f"  平均LogLoss: M0={ll_m0.mean():.5f} | M1={ll_m1.mean():.5f} | "
      f"M3计数={ll_m3.mean():.5f} | M3密度={ll_m3d.mean():.5f}")
    for lab, diff_ll in [("M0→M1(加长度)", ll_m0 - ll_m1),
                         ("M1→M3(再加排版)", ll_m1 - ll_m3),
                         ("M3计数→密度", ll_m3 - ll_m3d)]:
        mu_d = diff_ll.mean()
        se_d = cluster_se_mean(diff_ll, cl_te)
        pv_d = 2 * norm.sf(abs(mu_d) / se_d) if se_d > 0 else 1.0
        p(f"  {lab:16s} ΔLogLoss={mu_d:+.5f} "
          f"[{mu_d-1.96*se_d:+.5f}, {mu_d+1.96*se_d:+.5f}] p={pv_d:.2e}")

    # 聚类 Bootstrap (B=500, 仅用于模型排名区间, 每次抽样内部重算标准化sd)
    cl_groups = list(d.groupby("cluster_id").indices.values())
    n_cl = len(cl_groups)
    raw_mat = np.column_stack([raw_r["len"], raw_r["head"], raw_r["bold"], raw_r["list"]])
    ths = []
    for _ in range(B):
        pick = rng.integers(0, n_cl, n_cl)
        idx = np.concatenate([cl_groups[i] for i in pick])
        rm_b = raw_mat[idx]
        sd_b = rm_b.std(axis=0)
        z_b = rm_b / np.where(sd_b > 0, sd_b, 1.0)
        Xb = sparse.hstack([Xm[idx], sparse.csr_matrix(z_b)]).tocsr()
        lb = LogisticRegression(fit_intercept=False, C=C_BIG,
                                max_iter=1500).fit(Xb, y[idx])
        ths.append(restore_theta(lb.coef_[0], active, models).to_numpy())
    ths = np.asarray(ths)

    # 名次变动 + 聚类 Bootstrap 95% 区间
    rk_boot = np.apply_along_axis(lambda v: pd.Series(-v).rank().to_numpy(), 1, ths)
    rlo = np.percentile(rk_boot, 2.5, axis=0)
    rhi = np.percentile(rk_boot, 97.5, axis=0)
    m_pos = {m: i for i, m in enumerate(models)}
    top = set(r0.nsmallest(15).index) | set(r3.nsmallest(15).index)
    p(f"\n头部模型: 原排名→控风格排名 [聚类Bootstrap 95%区间, B={B}] (变动)")
    for m in sorted(top, key=lambda m: r0[m]):
        if m == "other":
            continue
        i = m_pos[m]
        p(f"  {m[:36]:36s} {int(r0[m]):>2d}→{int(r3[m]):>2d} "
          f"[{rlo[i]:.0f}-{rhi[i]:.0f}] ({r0[m]-r3[m]:+.0f})")

    # 存四阶实力与排名供脚本 5 与绘图使用
    rk_df = pd.DataFrame({
        "model": models,
        "theta0": th["M0"].to_numpy(), "theta1": th["M1"].to_numpy(),
        "theta2": th["M2"].to_numpy(), "theta3": th["M3"].to_numpy(),
        "rank0": r0.astype(int).to_numpy(), "rank1": r1.astype(int).to_numpy(),
        "rank2": r2.astype(int).to_numpy(), "rank3": r3.astype(int).to_numpy(),
        "rank3_lo": rlo, "rank3_hi": rhi,
    })
    rk_df.to_parquet(f"results/ranks_{tag}.parquet", index=False)

Path("results/bradley_terry_model_stats.md").write_text("\n".join(OUT), encoding="utf-8")
print("\n→ results/bradley_terry_model_stats.md + results/ranks_t2/t3.parquet")