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


MIN_VOTES, MIN_VOTES_SUB, C = 50, 20, 1e8
W = {"t2": ("t2_100k", "2024-06-20", "2024-07-25"),
     "t3": ("t3_140k", "2025-06-20", "2025-07-25")}
MD = ["head", "bold", "list"]


def base(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    s = a + b
    return np.where(s > 0, (a - b) / s, 0.0)


def std(v):
    sd = np.std(v)
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


def fit_logit_cluster(X, y, clusters, c_reg=C):
    """满秩逻辑回归 + Liang-Zeger 聚类稳健夹心标准误"""
    lr = LogisticRegression(fit_intercept=False, C=c_reg,
                            tol=1e-6, max_iter=2000).fit(X, y)
    beta = lr.coef_[0]
    prob = lr.predict_proba(X)[:, 1]
    resid = y - prob
    w = prob * (1.0 - prob)
    H = (X.T @ X.multiply(w[:, None])).toarray() + (1.0 / c_reg) * np.eye(X.shape[1])
    H_inv = np.linalg.inv(H)
    codes, uniques = pd.factorize(clusters, sort=False)
    g = len(uniques)
    n, k = X.shape
    C_mat = sparse.csr_matrix(
        (np.ones(n), (codes, np.arange(n))), shape=(g, n))
    S = (C_mat @ X.multiply(resid[:, None])).toarray()
    scale = (g / (g - 1)) * ((n - 1) / (n - k))
    cov = scale * (H_inv @ (S.T @ S) @ H_inv)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return beta, se


def restore_theta(coef, active, models):
    th = pd.Series(0.0, index=models)
    th.loc[active] = coef[:len(active)]
    return th


def fmt_ci(b, s):
    pv = 2 * norm.sf(abs(b) / s) if s > 0 else 1.0
    return f"{b:+.3f} [{b-1.96*s:+.3f}, {b+1.96*s:+.3f}] (p={pv:.2e})"


for tag, (src, lo, hi) in W.items():
    d = DF[(DF.source == src) & (DF.timestamp >= lo) & (DF.timestamp < hi) &
           (DF.language == "en") & (DF.turns == 1) &
           (DF.q_chars > 0) & (DF.a_chars > 0) & (DF.b_chars > 0) &
           DF.winner.isin(["model_a", "model_b"])].reset_index(drop=True)
    cl = d.cluster_id.to_numpy()
    p(f"\n{'='*10} {tag} n={len(d):,} (G={d.cluster_id.nunique():,}, 聚类稳健SE + 去代码块口径) {'='*10}")

    vc = pd.concat([d.model_a, d.model_b]).value_counts()
    keep = set(vc[vc >= MIN_VOTES].index)
    ma = d.model_a.where(d.model_a.isin(keep), "other").to_numpy()
    mb = d.model_b.where(d.model_b.isin(keep), "other").to_numpy()
    models = sorted(set(ma) | set(mb))
    ref = "other" if "other" in models else models[0]
    active = [m for m in models if m != ref]
    Xm = build_model_matrix(ma, mb, active)
    y = (d.winner == "model_a").astype(int).to_numpy()

    def fit(cols):
        X = sparse.hstack([Xm] + [sparse.csr_matrix(c.reshape(-1, 1))
                                  for c in cols]).tocsr() if cols else Xm
        return fit_logit_cluster(X, y, cl)

    both_tok = (d.a_tok_off > 0) & (d.b_tok_off > 0)
    ta = np.where(both_tok, d.a_tok_off, d.a_chars)
    tb = np.where(both_tok, d.b_tok_off, d.b_chars)
    zlen = std(base(ta, tb))

    # ===== [1] 计数口径 vs 密度口径 (去代码块清洗版, 聚类 95% CI) =====
    p("\n[1] 格式特征: 计数(官方) vs 密度(每千字) [聚类稳健 95% CI]")
    cnt = {k: std(base(d[f"clean_a_{k}"], d[f"clean_b_{k}"])) for k in MD}
    den = {k: std(base(d[f"clean_a_{k}"] / d.a_chars * 1000,
                       d[f"clean_b_{k}"] / d.b_chars * 1000)) for k in MD}
    bc, sec = fit([zlen] + [cnt[k] for k in MD])
    bd, sed = fit([zlen] + [den[k] for k in MD])
    for lab, bb, ss in [("计数", bc[-4:], sec[-4:]), ("密度", bd[-4:], sed[-4:])]:
        p(f"  {lab}: " + " | ".join(
            f"{k}={v:+.3f}[{v-1.96*s:+.3f},{v+1.96*s:+.3f}]"
            for k, v, s in zip(["len"] + MD, bb, ss)))

    # ===== [2] 长度补贴分解 (显式在非other模型上均值中心化, 消除基准漂移) =====
    p("\n[2] 长度补贴分解 (显式均值中心化)")
    b0, _ = fit([])
    th0 = restore_theta(b0, active, models).drop("other", errors="ignore")
    th3 = restore_theta(bc, active, models).drop("other", errors="ignore")
    th0 = th0 - th0.mean()
    th3 = th3 - th3.mean()

    midx_all = {m: i for i, m in enumerate(models)}
    signed = np.r_[zlen, -zlen]
    mids = np.r_[pd.Series(ma).map(midx_all).to_numpy(),
                 pd.Series(mb).map(midx_all).to_numpy()]
    s_m = (np.bincount(mids, weights=signed, minlength=len(models))
           / np.bincount(mids, minlength=len(models)))
    adv = pd.Series(s_m, index=models).drop("other", errors="ignore")
    subsidy = th0 - th3
    slope = (subsidy * adv).sum() / (adv ** 2).sum()
    p(f"  corr(补贴, 平均长度优势) = {np.corrcoef(subsidy, adv)[0,1]:.3f}")
    p(f"  过原点斜率 = {slope:.3f}  beta_len = {bc[-4]:.3f}")
    top = subsidy.abs().nlargest(6).index
    for m in top:
        p(f"    {m[:36]:36s} 补贴{subsidy[m]:+.3f} 长度优势{adv[m]:+.3f} θ3={th3[m]:+.3f}")

    # 导出模型生态汇总表 (供 4.3 节四象限气泡图与山脊图使用)
    all_chars = np.r_[d.a_chars.to_numpy(), d.b_chars.to_numpy()]
    mean_chars = (np.bincount(mids, weights=all_chars, minlength=len(models))
                  / np.bincount(mids, minlength=len(models)))
    eco = pd.DataFrame({
        "model": th0.index,
        "theta0": th0.to_numpy(),
        "theta3": th3.to_numpy(),
        "subsidy": subsidy.to_numpy(),
        "len_adv": adv.loc[th0.index].to_numpy(),
        "mean_chars": pd.Series(mean_chars, index=models).loc[th0.index].to_numpy(),
        "votes": pd.Series(np.bincount(mids, minlength=len(models)), index=models).loc[th0.index].to_numpy(),
    })
    eco.to_csv(f"results/model_ecology_{tag}.csv", index=False, encoding="utf-8-sig")

    # ===== [3] 固定同对模型检验 (Fixed-pair, 脱离任何BT假设) =====
    p("\n[3] 固定同对模型检验 (Fixed-pair, 双向均有观测, 按较小臂加权)")
    du = d[d.a_chars != d.b_chars]
    pair = [tuple(sorted(x)) for x in zip(du.model_a, du.model_b)]
    focal_long = ((du.model_a < du.model_b) == (du.a_chars > du.b_chars)).to_numpy()
    focal_won = ((du.model_a < du.model_b) == (du.winner == "model_a")).to_numpy()
    ptab = pd.DataFrame({"pair": pair, "long": focal_long, "won": focal_won}) \
             .groupby(["pair", "long"]).won.agg(["size", "mean"]).unstack().dropna()
    w_pair = ptab["size"].min(axis=1)
    r_long = float(np.average(ptab["mean"][True], weights=w_pair))
    r_short = float(np.average(ptab["mean"][False], weights=w_pair))
    p(f"  可比模型对={len(ptab):,} (最小臂加权总票={int(w_pair.sum()):,}) | "
      f"同模型写长胜率={r_long*100:.1f}% vs 写短胜率={r_short*100:.1f}% "
      f"(差={(r_long-r_short)*100:+.1f}pp)")

    # ===== [4] 异质性: 长度x任务 (聚类稳健 95% CI + 修正分组重拟合) =====
    p("\n[4] 长度x任务交互 [聚类稳健 95% CI]")
    ints = [zlen * d.tag_math.astype(float).to_numpy(),
            zlen * d.is_code.astype(float).to_numpy(),
            zlen * d.tag_if.astype(float).to_numpy()]
    bi, sei = fit([zlen] + [cnt[k] for k in MD] + ints)
    p(f"  beta_len={fmt_ci(bi[-7], sei[-7])}")
    for lab, idx_c in [("x数学", -3), ("x代码", -2), ("x指令遵循", -1)]:
        p(f"    {lab:6s} = {fmt_ci(bi[idx_c], sei[idx_c])}")

    # 分组重拟合: 补全三类清洗格式控制(与M3对齐), 并按子集自身票数重筛低频模型防完全分离
    resid_mask = (~d.tag_math) & (~d.is_code) & (~d.tag_if)
    for lab, dd in [("余类", d[resid_mask]), ("数学", d[d.tag_math == True])]:
        dd = dd.reset_index(drop=True)
        vc2 = pd.concat([dd.model_a, dd.model_b]).value_counts()
        keep2 = set(vc2[vc2 >= MIN_VOTES_SUB].index)
        ma2 = dd.model_a.where(dd.model_a.isin(keep2), "other").to_numpy()
        mb2 = dd.model_b.where(dd.model_b.isin(keep2), "other").to_numpy()
        mods2 = sorted(set(ma2) | set(mb2))
        ref2 = "other" if "other" in mods2 else mods2[0]
        act2 = [m for m in mods2 if m != ref2]
        Xm2 = build_model_matrix(ma2, mb2, act2)
        both_tok2 = (dd.a_tok_off > 0) & (dd.b_tok_off > 0)
        ta2 = np.where(both_tok2, dd.a_tok_off, dd.a_chars)
        tb2 = np.where(both_tok2, dd.b_tok_off, dd.b_chars)
        z2 = std(base(ta2, tb2))
        fmt2 = [std(base(dd[f"clean_a_{k}"], dd[f"clean_b_{k}"])) for k in MD]
        X2 = sparse.hstack(
            [Xm2] + [sparse.csr_matrix(c.reshape(-1, 1)) for c in [z2] + fmt2]
        ).tocsr()
        y2 = (dd.winner == "model_a").astype(int).to_numpy()
        b2, se2 = fit_logit_cluster(X2, y2, dd.cluster_id.to_numpy())
        p(f"  分组重拟合M3 [{lab}] n={len(dd):,} (保留模型{len(act2)}个) "
          f"beta_len={fmt_ci(b2[-4], se2[-4])}")

    # ===== [5] 非线性: 长度收益是否饱和 (聚类稳健 95% CI) =====
    p("\n[5] 非线性检验 (奇函数项 x|x|, 聚类稳健 95% CI)")
    bq, seq = fit([zlen, zlen * np.abs(zlen)] + [cnt[k] for k in MD])
    p(f"  beta_len  = {fmt_ci(bq[-5], seq[-5])}")
    p(f"  beta_quad = {fmt_ci(bq[-4], seq[-4])}")

Path("results/style_extension.md").write_text("\n".join(OUT), encoding="utf-8")
print("\n→ results/style_extension.md + results/model_ecology_t2/t3.csv")