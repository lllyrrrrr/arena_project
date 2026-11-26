"""图 10：2024 vs 2025 模型平均回答字数分化与控风格名次变动箱线散点对比图"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ECO_T2 = Path("results/model_ecology_t2.csv")
ECO_T3 = Path("results/model_ecology_t3.csv")
RK_T2 = Path("results/ranks_t2.parquet")
RK_T3 = Path("results/ranks_t3.parquet")
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========== 柔和紫系配色（与全套图表统一） ==========
C_2024 = "#9D8EC7"
C_2024_FILL = "#EBE5F7"
C_2025 = "#5B3896"
C_2025_FILL = "#D6C8EE"
C_INK = "#241B35"
C_MUTED = "#6B5E80"
C_GRID = "#EFEBF7"

plt.rcParams.update({
    "font.sans-serif": ["PingFang SC", "Hiragino Sans GB", "Songti SC", "Arial Unicode MS"],
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "text.color": C_INK,
    "axes.labelcolor": C_INK,
    "xtick.color": C_INK,
    "ytick.color": C_MUTED,
    "axes.edgecolor": "#CFC6E2",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# ========== 读取两期模型级字数与名次变动 ==========
e24 = pd.read_csv(ECO_T2)
e25 = pd.read_csv(ECO_T3)
r24 = pd.read_parquet(RK_T2)
r25 = pd.read_parquet(RK_T3)

len24 = e24["mean_chars"].to_numpy(dtype=float)
len25 = e25["mean_chars"].to_numpy(dtype=float)

d24 = (r24[r24["model"] != "other"]["rank0"] - r24[r24["model"] != "other"]["rank3"]).abs().to_numpy(dtype=float)
d25 = (r25[r25["model"] != "other"]["rank0"] - r25[r25["model"] != "other"]["rank3"]).abs().to_numpy(dtype=float)

sd24, sd25 = float(np.std(len24, ddof=1)), float(np.std(len25, ddof=1))
med24, med25 = float(np.median(d24)), float(np.median(d25))


def draw_box_strip(ax, v24, v25, seed=42):
    rng = np.random.default_rng(seed)
    positions = [0.0, 1.0]

    bp = ax.boxplot(
        [v24, v25], positions=positions, widths=0.38,
        patch_artist=True, showfliers=False,
        medianprops=dict(color=C_INK, lw=1.8),
        whiskerprops=dict(color=C_MUTED, lw=1.1),
        capprops=dict(color=C_MUTED, lw=1.1),
        zorder=2,
    )
    bp["boxes"][0].set(facecolor=C_2024_FILL, edgecolor=C_2024, lw=1.3, alpha=0.85)
    bp["boxes"][1].set(facecolor=C_2025_FILL, edgecolor=C_2025, lw=1.3, alpha=0.80)

    # 叠加全部模型抖动散点
    j24 = rng.uniform(-0.11, 0.11, size=len(v24))
    j25 = rng.uniform(-0.11, 0.11, size=len(v25))
    ax.scatter(positions[0] + j24, v24, s=26, fc="white", ec=C_2024, lw=1.2, alpha=0.92, zorder=3)
    ax.scatter(positions[1] + j25, v25, s=26, marker="D", fc=C_2025, ec="white", lw=0.5, alpha=0.85, zorder=4)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"2024\n(n={len(v24)})", f"2025\n(n={len(v25)})"], fontsize=9.4, color=C_INK)
    ax.set_xlim(-0.48, 1.48)
    ax.grid(axis="y", color=C_GRID, lw=0.9)
    ax.set_axisbelow(True)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 4.1), gridspec_kw={"wspace": 0.30})

# ---- 左：模型平均回答字符数 ----
draw_box_strip(ax1, len24, len25, seed=42)
ax1.set_ylabel("模型平均回答字符数", fontsize=9.8)
ax1.set_ylim(0, max(len25) * 1.18)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
ax1.text(0.0, max(len24) + 260, f"SD = {sd24:.0f}", ha="center", va="bottom",
         fontsize=8.6, color=C_MUTED)
ax1.text(1.0, max(len25) + 240, f"SD = {sd25:,.0f} (×{sd25/sd24:.2f})", ha="center", va="bottom",
         fontsize=8.8, fontweight="bold", color=C_2025)

# ---- 右：控风格前后名次绝对变动 ----
draw_box_strip(ax2, d24, d25, seed=7)
ax2.set_ylabel(r"名次绝对变动 $|\Delta\mathrm{rank}|$", fontsize=9.8)
ax2.set_ylim(-1.0, max(d25) * 1.18)
ax2.text(0.0, max(d24) + 0.9, f"Med = {med24:.1f}", ha="center", va="bottom",
         fontsize=8.6, color=C_MUTED)
ax2.text(1.0, max(d25) + 0.9, f"Med = {med25:.1f} (×{med25/med24:.1f})", ha="center", va="bottom",
         fontsize=8.8, fontweight="bold", color=C_2025)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"fig10_cross_year_divergence.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("已生成:", OUT_DIR / "fig10_cross_year_divergence.pdf", "及 .png")
