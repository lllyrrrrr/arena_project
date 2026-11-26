"""图 5：2025 控风格实力 θ(M3) × 风格补贴四象限生态气泡图（柔和紫系，颜色映射平均字数）"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

IN_FILE = Path("results/model_ecology_t3.csv")
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

df = pd.read_csv(IN_FILE)
df = df[df["votes"] >= 200].reset_index(drop=True)

# 柔和紫系连续色带（浅薰衣草 → 雅致深紫）
cmap = LinearSegmentedColormap.from_list(
    "soft_purple_len",
    ["#ECE6F8", "#C2B0E4", "#8D6EC6", "#53318E"],
    N=256,
)

fig, ax = plt.subplots(figsize=(6.8, 4.3))
ax.axhline(0.0, color="#BFB2D9", lw=1.0, ls="--", zorder=1)
ax.axvline(0.0, color="#BFB2D9", lw=1.0, ls="--", zorder=1)

sizes = 36.0 + 135.0 * (np.sqrt(df["votes"]) - np.sqrt(df["votes"].min())) / (
    np.sqrt(df["votes"].max()) - np.sqrt(df["votes"].min()) + 1e-9
)

sc = ax.scatter(
    df["theta3"], df["subsidy"],
    s=sizes, c=df["mean_chars"], cmap=cmap, vmin=1400, vmax=6500,
    edgecolors="white", linewidths=0.85, alpha=0.92, zorder=3,
)

# 精选 9 个代表模型标注（全部指向空白区域，零重叠）
highlights = [
    ("o3-2025-04-16",                       "o3",               (10, -14), "left"),
    ("chatgpt-4o-latest-20250326",          "GPT-4o",           (10, 12),  "left"),
    ("gemini-2.5-pro",                      "Gemini-2.5-Pro",   (0, 18),   "center"),
    ("grok-4-0709",                         "Grok-4",           (-12, -16),"right"),
    ("kimi-k2-0711-preview",                "Kimi-k2",          (12, -8),  "left"),
    ("claude-opus-4-20250514",              "Claude-Opus-4",    (14, -18), "left"),
    ("grok-3-preview-02-24",                "Grok-3",           (14, 14),  "left"),
    ("llama-4-maverick-03-26-experimental", "Llama-4-Mav-exp",  (-16, 12), "right"),
    ("claude-3-7-sonnet-20250219",          "Claude-3.7",       (-14, 0),  "right"),
]
for raw, lbl, (dx, dy), ha in highlights:
    r = df[df["model"] == raw]
    if len(r) == 1:
        px, py = float(r["theta3"].iloc[0]), float(r["subsidy"].iloc[0])
        ax.annotate(
            lbl, xy=(px, py), xytext=(dx, dy), textcoords="offset points",
            ha=ha, va="center", fontsize=8.0, fontweight="bold", color=C_INK,
            arrowprops=dict(arrowstyle="-", color="#BFB3D6", lw=0.75, shrinkA=0, shrinkB=4),
            zorder=5,
        )

ax.set_xlabel(r"$\theta^{(M3)}$", fontsize=11.0)
ax.set_ylabel(r"$\theta^{(M0)} - \theta^{(M3)}$", fontsize=11.0)
ax.set_xlim(-0.72, 0.78)
ax.set_ylim(-0.68, 0.50)
ax.grid(color=C_GRID, lw=0.85)
ax.set_axisbelow(True)

cbar = fig.colorbar(sc, ax=ax, fraction=0.042, pad=0.03)
cbar.set_label("平均字符数", fontsize=9.2, color=C_INK)
cbar.formatter = FuncFormatter(lambda v, _: f"{int(v):,}")
cbar.update_ticks()
cbar.outline.set_visible(False)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"fig5_subsidy_ecology.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("已生成:", OUT_DIR / "fig5_subsidy_ecology.pdf", "及 .png")
