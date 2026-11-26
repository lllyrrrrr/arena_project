"""图 4：2025 代表模型 M0（原始）→ M3（控风格）名次变动双向哑铃图（柔和紫系）"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

IN_FILE = Path("results/ranks_t3.parquet")
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========== 柔和紫系配色（与 fig3 保持通透统一） ==========
C_UP = "#6A4C9C"      # 控风格后上升（克制受益）：雅致紫
C_DN = "#C975A2"      # 控风格后下降（长文回落）：柔梅粉紫
C_M0 = "#A799CC"      # M0 原始名次边框
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
    "ytick.color": C_INK,
    "axes.edgecolor": "#CFC6E2",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

df = pd.read_parquet(IN_FILE)
df = df[df["model"] != "other"].copy()
df["delta"] = df["rank0"] - df["rank3"]  # >0 表示控风格后名次上升

# 精选上升前 7 + 下降前 7 共 14 个变动最显著模型
up7 = df.sort_values("delta", ascending=False).head(7)
dn7 = df.sort_values("delta", ascending=True).head(7)
sel = pd.concat([up7, dn7.iloc[::-1]], ignore_index=True)


def short_model(name):
    s = name
    for suf in (
        "-20250514", "-20250219", "-20241022", "-2025-04-16", "-2025-04-14",
        "-0711-preview", "-03-26-experimental", "-02-24", "-preview-04-17",
        "-preview-05-06", "-preview-06-17", "-05-14",
    ):
        s = s.replace(suf, "")
    return s.replace("-thinking-32k", " (think)").replace("-thinking-16k", " (think)")


sel["label"] = sel["model"].map(short_model)
sel = sel.iloc[::-1].reset_index(drop=True)  # 自上而下：上升组在上，下降组在下

fig, ax = plt.subplots(figsize=(7.0, 4.7))
y = np.arange(len(sel))
ax.axhline(6.5, color="#D8CEEC", lw=1.0, ls="--", zorder=1)

for i, row in sel.iterrows():
    r0, r3, d = float(row["rank0"]), float(row["rank3"]), int(round(row["delta"]))
    col = C_UP if d > 0 else C_DN
    ax.annotate(
        "", xy=(r3, i), xytext=(r0, i),
        arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.28",
                        color=col, lw=2.0, shrinkA=4, shrinkB=4),
        zorder=3,
    )
    if d > 0:
        ax.text(r3 - 1.0, i, f"+{d}", ha="right", va="center",
                fontsize=8.2, fontweight="bold", color=C_UP)
    else:
        ax.text(r3 + 1.0, i, f"{d}", ha="left", va="center",
                fontsize=8.2, fontweight="bold", color="#9E4B78")

ax.scatter(sel["rank0"], y, s=46, fc="white", ec=C_M0, lw=1.7, zorder=4)
ax.scatter(
    sel["rank3"], y, s=48, marker="D",
    fc=[C_UP if d > 0 else C_DN for d in sel["delta"]],
    ec="white", lw=0.7, zorder=5,
)

ax.set_yticks(y)
ax.set_yticklabels(sel["label"], fontsize=8.6, color=C_INK)
ax.set_xlabel("2025 榜单名次（靠左更高）", fontsize=9.8)
ax.set_xlim(-3.5, max(sel[["rank0", "rank3"]].max()) + 4.8)
ax.set_xticks([1, 10, 20, 30, 40, 48])
ax.grid(axis="x", color=C_GRID, lw=0.9)
ax.set_axisbelow(True)

ax.legend(
    handles=[
        Line2D([0], [0], marker="o", lw=0, ms=6.0, mfc="white", mec=C_M0, mew=1.6, label="M0 原始"),
        Line2D([0], [0], marker="D", lw=0, ms=5.8, mfc=C_UP, mec="white", label="M3 上升"),
        Line2D([0], [0], marker="D", lw=0, ms=5.8, mfc=C_DN, mec="white", label="M3 下降"),
    ],
    loc="lower right", frameon=False, fontsize=8.6, handletextpad=0.35,
)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"fig4_rank_dumbbell.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("已生成:", OUT_DIR / "fig4_rank_dumbbell.pdf", "及 .png")
