"""图 8：2025 长度差 × 用词重合度（Jaccard）平局率 vs 双差率同色阶热力图"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

IN_FILE = Path("results/undecided_grid.csv")
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_INK = "#241B35"
C_MUTED = "#6B5E80"

plt.rcParams.update({
    "font.sans-serif": ["PingFang SC", "Hiragino Sans GB", "Songti SC", "Arial Unicode MS"],
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "text.color": C_INK,
    "axes.labelcolor": C_INK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

df = pd.read_csv(IN_FILE)
df = df[df["period"] == "t3"].copy()

ov_order = [">0.50", "0.30–0.50", "0.15–0.30", "≤0.15"]
gap_order = ["≤10%", "10%–60%", ">60%"]

mat_tie = (
    df.pivot(index="overlap_bin", columns="gap_bin", values="tie_rate")
    .reindex(index=ov_order, columns=gap_order).to_numpy() * 100.0
)
mat_bad = (
    df.pivot(index="overlap_bin", columns="gap_bin", values="both_bad_rate")
    .reindex(index=ov_order, columns=gap_order).to_numpy() * 100.0
)

cmap = LinearSegmentedColormap.from_list(
    "soft_purple_heat",
    ["#F7F4FC", "#DFD3F2", "#A98FD6", "#5B3896"],
    N=256,
)

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(7.4, 3.8), sharey=True,
    gridspec_kw={"wspace": 0.12}
)

vmin, vmax = 6.0, 33.0

for ax, mat, title in [(ax1, mat_tie, "平局 (tie)"), (ax2, mat_bad, "双差 (both_bad)")]:
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for r in range(1, len(ov_order)):
        ax.axhline(r - 0.5, color="white", lw=2.0)
    for c in range(1, len(gap_order)):
        ax.axvline(c - 0.5, color="white", lw=2.0)
    for r in range(len(ov_order)):
        for c in range(len(gap_order)):
            v = mat[r, c]
            txt_col = "white" if v >= 22.0 else C_INK
            ax.text(c, r, f"{v:.1f}%", ha="center", va="center",
                    fontsize=10.2, fontweight="bold", color=txt_col)
    ax.set_title(title, fontsize=10.2, pad=8, color=C_INK)
    ax.set_xticks(range(len(gap_order)))
    ax.set_xticklabels(gap_order, fontsize=9.4, color=C_INK)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)

ax1.set_yticks(range(len(ov_order)))
ax1.set_yticklabels(ov_order, fontsize=9.4, color=C_INK)
ax1.set_ylabel("Jaccard 重合度", fontsize=9.8, labelpad=6)
fig.supxlabel("相对长度差", fontsize=9.8, y=0.01, color=C_INK)

cbar = fig.colorbar(im, ax=[ax1, ax2], fraction=0.032, pad=0.03)
cbar.set_label("占比（%）", fontsize=9.2, color=C_INK)
cbar.set_ticks([10, 15, 20, 25, 30])
cbar.outline.set_visible(False)

fig.subplots_adjust(left=0.13, right=0.87, top=0.88, bottom=0.16)
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"fig8_undecided_heatmap.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("已生成:", OUT_DIR / "fig8_undecided_heatmap.pdf", "及 .png")
