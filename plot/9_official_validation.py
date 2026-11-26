"""图 9：与 LMArena 官方控风格排行榜名次变动交叉核验散点图（2024 & 2025）"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_T2 = Path("results/common_ranks_t2.csv")
IN_T3 = Path("results/common_ranks_t3.csv")
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_2024 = "#9D8EC7"
C_2024_FILL = "#EBE5F7"
C_2025 = "#5B3896"
C_DIAG = "#C975A2"
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

rho24 = spearmanr(t2["d_off"], t2["d_ours"]).statistic
rho25 = spearmanr(t3["d_off"], t3["d_ours"]).statistic

fig, ax = plt.subplots(figsize=(6.4, 4.4))

lim = 19.5
ax.axhline(0, color="#D8CEEC", lw=0.9, zorder=1)
ax.axvline(0, color="#D8CEEC", lw=0.9, zorder=1)
ax.plot([-lim, lim], [-lim, lim], color=C_DIAG, lw=1.3, ls="--", alpha=0.8, zorder=2)

# 一致象限（第一、三象限）极淡底色
ax.fill_between([0, lim], 0, lim, color="#F7F4FC", alpha=0.65, zorder=0)
ax.fill_between([-lim, 0], -lim, 0, color="#F7F4FC", alpha=0.65, zorder=0)

ax.scatter(
    t2["d_off"], t2["d_ours"], s=44,
    fc=C_2024_FILL, ec=C_2024, lw=1.3, alpha=0.92,
    label=f"2024 (n={len(t2)}, ρ={rho24:.3f})", zorder=3,
)
ax.scatter(
    t3["d_off"], t3["d_ours"], s=46, marker="D",
    fc=C_2025, ec="white", lw=0.7, alpha=0.88,
    label=f"2025 (n={len(t3)}, ρ={rho25:.3f})", zorder=4,
)

# 精选 5 个代表性大变动模型标注（避让空白区）
highlights = [
    ("claude-3-7-sonnet-20250219-thinking-32k", "Claude-3.7 (think)", (-10, 10), "right"),
    ("kimi-k2-0711-preview",                    "Kimi-k2",            (-10, 12), "right"),
    ("claude-opus-4-20250514",                  "Claude-Opus-4",      (10, -10), "left"),
    ("grok-3-preview-02-24",                    "Grok-3",             (10, -10), "left"),
    ("gemma-3-27b-it",                          "Gemma-3-27B",        (10, -6),  "left"),
]
for raw, lbl, (dx, dy), ha in highlights:
    r = t3[t3["model"] == raw]
    if len(r) == 1:
        px, py = float(r["d_off"].iloc[0]), float(r["d_ours"].iloc[0])
        ax.annotate(
            lbl, xy=(px, py), xytext=(dx, dy), textcoords="offset points",
            ha=ha, va="center", fontsize=7.8, fontweight="bold", color=C_2025,
            arrowprops=dict(arrowstyle="-", color="#BFB3D6", lw=0.7, shrinkA=0, shrinkB=4),
            zorder=5,
        )

ax.set_xlabel(r"$\Delta_{\mathrm{official}}$", fontsize=11.0)
ax.set_ylabel(r"$\Delta_{\mathrm{ours}}$", fontsize=11.0)
ax.set_xlim(-14.5, 15.5)
ax.set_ylim(-15.5, 19.5)
ax.grid(color=C_GRID, lw=0.85)
ax.set_axisbelow(True)
ax.legend(loc="upper left", frameon=False, fontsize=9.0, handletextpad=0.35)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"fig9_official_validation.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("已生成:", OUT_DIR / "fig9_official_validation.pdf", "及 .png")
