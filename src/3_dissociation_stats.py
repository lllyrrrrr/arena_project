import pandas as pd
import numpy as np
from pathlib import Path

DF = pd.read_parquet("results/features.parquet")
OUT = []
def p(s=""):
    print(s); OUT.append(str(s))


def cluster_se_mean(vals, clusters):
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
    """胜率% + 聚类 95% CI(±半宽)"""
    v = np.asarray(vals, dtype=float)
    n = len(v)
    if n == 0:
        return "n=0"
    r = v.mean()
    hw = 1.96 * cluster_se_mean(v, clusters)
    return f"{r*100:.1f}%±{hw*100:.1f} (n={n:,})"


W = {"t2": ("t2_100k", "2024-06-20", "2024-07-25"),
     "t3": ("t3_140k", "2025-06-20", "2025-07-25")}

for tag, (src, lo, hi) in W.items():
    # 主样本 ∩ 分胜负 ∩ 不等长
    d = DF[(DF.source == src) & (DF.timestamp >= lo) & (DF.timestamp < hi) &
           (DF.language == "en") & (DF.turns == 1) &
           (DF.q_chars > 0) & (DF.a_chars > 0) & (DF.b_chars > 0) &
           DF.winner.isin(["model_a", "model_b"]) & (DF.a_chars != DF.b_chars)]
    cl = d.cluster_id.to_numpy()
    p(f"\n===== {tag} n={len(d):,} (G={d.cluster_id.nunique():,}, 去代码块Markdown口径) =====")

    # 格式密度(每千字清洗标记数)
    fa = (d.clean_a_bold + d.clean_a_head + d.clean_a_list) / d.a_chars * 1000
    fb = (d.clean_b_bold + d.clean_b_head + d.clean_b_list) / d.b_chars * 1000
    ldiff = (d.a_chars - d.b_chars).abs() / ((d.a_chars + d.b_chars) / 2)
    fdiff = (fa - fb).abs()

    len_big = ldiff > ldiff.median()
    fmt_big = fdiff > fdiff.median()
    p(f"(切点: 长度差中位 {ldiff.median():.2f} | 密度差中位 {fdiff.median():.2f})")

    longer = np.where(d.a_chars > d.b_chars, "a", "b")
    denser = np.where(fa > fb, "a", np.where(fb > fa, "b", "eq"))
    w_long = (d.winner == ("model_" + longer)).astype(int).to_numpy()
    w_dense = ((d.winner == ("model_" + denser)) & (denser != "eq")).astype(int).to_numpy()

    # [1] 四象限: 统一因变量为长侧胜率, 象限③同时报告密侧胜率供对照
    q1 = (~len_big & ~fmt_big).to_numpy()
    q2 = (len_big & ~fmt_big).to_numpy()
    q3 = (~len_big & fmt_big).to_numpy()
    q4 = (len_big & fmt_big).to_numpy()
    p("\n[1] 四象限解耦 (统一长侧胜率, 聚类 95% CI):")
    p(f"  ① 都接近(基准)    长侧 {wr(w_long[q1], cl[q1])}")
    p(f"  ② 长同花(纯长度)  长侧 {wr(w_long[q2], cl[q2])}")
    p(f"  ③ 花同长(纯格式)  长侧 {wr(w_long[q3], cl[q3])} | 密侧 {wr(w_dense[q3], cl[q3])}")
    p(f"  ④ 双大(混合)      长侧 {wr(w_long[q4], cl[q4])}")

    # [2] 固定长度差三档 × 排版稀密方向 (3×3 微距反杀检验)
    rel_dir = np.where(denser == "eq", "两侧同密",
                       np.where(longer == denser, "长侧更密", "长侧更稀"))
    gap_bin = np.where(ldiff <= 0.10, "≤10%", np.where(ldiff <= 0.60, "10%–60%", ">60%"))
    p("\n[2] 固定长度差 × 稀密方向 (全格统一报告长侧胜率, 聚类 95% CI):")
    for row_lab in ["长侧更稀", "两侧同密", "长侧更密"]:
        rm = rel_dir == row_lab
        cells = []
        for col_lab in ["≤10%", "10%–60%", ">60%"]:
            cm = rm & (gap_bin == col_lab)
            cells.append(f"{col_lab}: {wr(w_long[cm], cl[cm])}")
        p(f"  {row_lab:4s} (整行 {wr(w_long[rm], cl[rm])}) | " + " ; ".join(cells))

    # [3] 特征共线性: 两侧非空回答合并后的 Spearman 相关矩阵
    p("\n[3] 回答长度与三类清洗Markdown计数的 Spearman 相关矩阵:")
    ans = pd.DataFrame({
        "chars": np.r_[d.a_chars.to_numpy(), d.b_chars.to_numpy()],
        "head":  np.r_[d.clean_a_head.to_numpy(), d.clean_b_head.to_numpy()],
        "bold":  np.r_[d.clean_a_bold.to_numpy(), d.clean_b_bold.to_numpy()],
        "list":  np.r_[d.clean_a_list.to_numpy(), d.clean_b_list.to_numpy()],
    })
    corr = ans.corr(method="spearman")
    for idx_name, row in corr.iterrows():
        p(f"  {idx_name:5s} " + " ".join(f"{val:6.3f}" for val in row))

Path("results/dissociation_stats.md").write_text("\n".join(OUT), encoding="utf-8")
print("\n→ results/dissociation_stats.md")