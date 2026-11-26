"""图 6：2024 vs 2025 模型平均长度优势 × 风格补贴散点与线性拟合（柔和紫系）"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_T2 = Path("results/model_ecology_t2.csv")
IN_T3 = Path("results/model_ecology_t3.csv")
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_2024 = "#9D8EC7"
C_2024_FILL = "#EBE5F7"
C_2025 = "#5B3896"
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

t2 = pd.read_csv(IN_T2)
t3 = pd.read_csv(IN_T3)

# 计算相关系数与过原点拟合斜率（与 src/6_style_extension.py 完全同口径）
def fit_origin(df):
    x = df["len_adv"].to_numpy(dtype=float)
    y = df["subsidy"].to_numpy(dtype=float)
    r = float(np.corrcoef(x, y)[0, 1])
    k = float(np.dot(x, y) / np.dot(x, x))
    return r, k

r24, k24 = fit_origin(t2)
r25, k25 = fit_origin(t3)

fig, ax = plt.subplots(figsize=(6.6, 4.2))
ax.axhline(0.0, color="#D8CEEC", lw=0.9, ls="--", zorder=1)
ax.axvline(0.0, color="#D8CEEC", lw=0.9, ls="--", zorder=1)

# 拟合直线（分别画在各自样本跨度内，直观凸显 2025 横向跨度拉伸）
x24_grid = np.linspace(t2["len_adv"].min(), t2["len_adv"].max(), 100)
x25_grid = np.linspace(t3["len_adv"].min(), t3["len_adv"].max(), 100)
ax.plot(x24_grid, k24 * x24_grid, color=C_2024, lw=2.0, ls="--", zorder=3)
ax.plot(x25_grid, k25 * x25_grid, color=C_2025, lw=2.2, ls="-", zorder=4)

# 散点
ax.scatter(
    t2["len_adv"], t2["subsidy"], s=44,
    fc=C_2024_FILL, ec=C_2024, lw=1.35, alpha=0.92,
    label=f"2024 (r={r24:.3f}, k={k24:.2f})", zorder=3,
)
ax.scatter(
    t3["len_adv"], t3["subsidy"], s=46, marker="D",
    fc=C_2025, ec="white", lw=0.7, alpha=0.86,
    label=f"2025 (r={r25:.3f}, k={k25:.2f})", zorder=5,
)

ax.set_xlabel(r"$\bar{x}_{\mathrm{len}}$", fontsize=11.0)
ax.set_ylabel(r"$\theta^{(M0)} - \theta^{(M3)}$", fontsize=11.0)
ax.set_xlim(-1.28, 1.02)
ax.set_ylim(-0.68, 0.50)
ax.grid(color=C_GRID, lw=0.85)
ax.set_axisbelow(True)
ax.legend(loc="upper left", frameon=False, fontsize=9.2, handletextpad=0.35)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"fig6_subsidy_fit.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("已生成:", OUT_DIR / "fig6_subsidy_fit.pdf", "及 .png")
