"""图 7：2025 分箱散点 + 含二次项逻辑回归拟合曲线（非数学 vs 数学任务边界）"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

IN_FILE = Path("results/features.parquet")
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_MAIN = "#5B3896"     # 非数学：雅致深紫
C_MATH = "#C975A2"     # 数学：柔梅粉紫
C_INK = "#241B35"
C_MUTED = "#6B5E80"
C_GRID = "#EFEBF7"

plt.rcParams.update({
    "font.sans-serif": ["PingFang SC", "Hiragino Sans GB", "Songti SC", "Arial Unicode MS"],
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "text.color": C_INK,
    "axes.labelcolor": C_INK,
    "xtick.color": C_MUTED,
    "ytick.color": C_MUTED,
    "axes.edgecolor": "#CFC6E2",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# ========== 读取 2025 分胜负样本（与 src/6_style_extension.py 完全同口径） ==========
cols = [
    "source", "timestamp", "language", "turns", "winner", "cluster_id",
    "q_chars", "a_chars", "b_chars", "tag_math",
]
df = pd.read_parquet(IN_FILE, columns=cols)
d = df[
    (df.source == "t3_140k") & (df.timestamp >= "2025-06-20") & (df.timestamp < "2025-07-25") &
    (df.language == "en") & (df.turns == 1) &
    (df.q_chars > 0) & (df.a_chars > 0) & (df.b_chars > 0) &
    (df.winner.isin(["model_a", "model_b"]))
].copy()

a = d["a_chars"].to_numpy(dtype=float)
b = d["b_chars"].to_numpy(dtype=float)
raw_len = (a - b) / (a + b)
d["x_len"] = raw_len / np.std(raw_len)
d["win_a"] = (d["winner"] == "model_a").astype(float)
d["is_math"] = d["tag_math"].fillna(False).astype(bool)


def fit_curve_with_ci(sub, x_grid):
    """拟合 logit(P) = b0 + b1*x + b2*(x|x|) + 聚类稳健 Delta Method 95% CI"""
    x = sub["x_len"].to_numpy(dtype=float)
    y = sub["win_a"].to_numpy(dtype=float)
    cl = sub["cluster_id"].to_numpy()

    X = np.column_stack([np.ones_like(x), x, x * np.abs(x)])
    lr = LogisticRegression(fit_intercept=False, C=1e8, tol=1e-7, max_iter=2000).fit(X, y)
    beta = lr.coef_[0]
    p = lr.predict_proba(X)[:, 1]
    w = p * (1.0 - p)
    H = (X.T * w) @ X
    H_inv = np.linalg.inv(H)

    # Liang-Zeger 聚类稳健协方差
    U = X * (y - p)[:, None]
    Ug = pd.DataFrame(U).groupby(cl).sum().to_numpy()
    G = len(Ug)
    V = (G / max(G - 1, 1)) * (H_inv @ (Ug.T @ Ug) @ H_inv)

    Xg = np.column_stack([np.ones_like(x_grid), x_grid, x_grid * np.abs(x_grid)])
    eta = Xg @ beta
    se_eta = np.sqrt(np.einsum("ij,jk,ik->i", Xg, V, Xg))

    sig = lambda z: 1.0 / (1.0 + np.exp(-z)) * 100.0
    return sig(eta), sig(eta - 1.96 * se_eta), sig(eta + 1.96 * se_eta)


def binned_scatter(sub, n_bins):
    q = pd.qcut(sub["x_len"], q=n_bins, duplicates="drop")
    g = sub.groupby(q, observed=True).agg(x=("x_len", "mean"), y=("win_a", "mean"))
    return g["x"].to_numpy(), g["y"].to_numpy() * 100.0


d_non = d[~d["is_math"]]
d_math = d[d["is_math"]]

x_grid = np.linspace(-2.25, 2.25, 200)
p_non, lo_non, hi_non = fit_curve_with_ci(d_non, x_grid)
p_mat, lo_mat, hi_mat = fit_curve_with_ci(d_math, x_grid)

bx_non, by_non = binned_scatter(d_non, 15)
bx_mat, by_mat = binned_scatter(d_math, 9)

# ========== 单图绘制：分箱散点 + 拟合曲线 + 95% CI 带 ==========
fig, ax = plt.subplots(figsize=(6.8, 4.2))
ax.axhline(50.0, color="#D8CEEC", lw=1.0, ls="--", zorder=1)
ax.axvline(0.0, color="#D8CEEC", lw=1.0, ls="--", zorder=1)

# 拟合曲线与置信带
ax.fill_between(x_grid, lo_non, hi_non, color=C_MAIN, alpha=0.14, lw=0, zorder=2)
ax.plot(x_grid, p_non, color=C_MAIN, lw=2.3, ls="-", label="非数学任务", zorder=4)

ax.fill_between(x_grid, lo_mat, hi_mat, color=C_MATH, alpha=0.14, lw=0, zorder=2)
ax.plot(x_grid, p_mat, color=C_MATH, lw=2.1, ls="--", label="数学任务", zorder=4)

# 等频分箱实测散点
ax.scatter(bx_non, by_non, s=34, fc=C_MAIN, ec="white", lw=0.7, alpha=0.92, zorder=5)
ax.scatter(bx_mat, by_mat, s=36, marker="^", fc="white", ec=C_MATH, lw=1.5, alpha=0.95, zorder=5)

ax.set_xlabel(r"$x_{\mathrm{len}}$", fontsize=11.0)
ax.set_ylabel("A 侧胜率（%）", fontsize=9.8)
ax.set_xlim(-2.35, 2.35)
ax.set_ylim(22, 78)
ax.set_yticks([25, 35, 50, 65, 75])
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}%"))
ax.grid(color=C_GRID, lw=0.85)
ax.set_axisbelow(True)
ax.legend(loc="upper left", frameon=False, fontsize=9.2)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"fig7_logit_boundary.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("已生成:", OUT_DIR / "fig7_logit_boundary.pdf", "及 .png")
