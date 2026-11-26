"""图 0：按用户提问字符分档，对比 2024 与 2025 模型回答中位字符数"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ========== 路径 ==========
IN_FILE = Path("results/features.parquet")
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========== 紫色系配色 ==========
C_2024 = "#9D8EC7"      # 2024：柔和薰衣草紫
C_2025 = "#4B2882"      # 2025：深皇家紫
C_GAP = "#EDE7FA"       # 膨胀带淡紫底色
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

# ========== 读取主样本 ==========
cols = ["source", "timestamp", "language", "turns", "q_chars", "a_chars", "b_chars"]
df = pd.read_parquet(IN_FILE, columns=cols)

W = {
    "2024": ("t2_100k", "2024-06-20", "2024-07-25"),
    "2025": ("t3_140k", "2025-06-20", "2025-07-25"),
}

bins = [0, 30, 80, 200, 500, np.inf]
labels = ["≤30", "31–80", "81–200", "201–500", ">500"]
x = np.arange(len(labels))

meds = {}
for yr, (src, lo, hi) in W.items():
    sub = df[
        (df.source == src) & (df.timestamp >= lo) & (df.timestamp < hi) &
        (df.language == "en") & (df.turns == 1) &
        (df.q_chars > 0) & (df.a_chars > 0) & (df.b_chars > 0)
    ]
    q = sub.q_chars.to_numpy(dtype=float)
    q_pair = np.concatenate([q, q])
    ans = np.concatenate([sub.a_chars.to_numpy(dtype=float),
                          sub.b_chars.to_numpy(dtype=float)])
    b = pd.cut(q_pair, bins=bins, labels=labels)
    g = pd.DataFrame({"b": b, "ans": ans}).groupby("b", observed=True)["ans"]
    meds[yr] = g.median().reindex(labels).to_numpy()

# ========== 单图绘制 ==========
fig, ax = plt.subplots(figsize=(6.8, 4.1))

ax.fill_between(x, meds["2024"], meds["2025"], color=C_GAP, alpha=0.9, zorder=1)
ax.plot(x, meds["2024"], color=C_2024, lw=2.1, ls="--", marker="o",
        ms=6.6, mfc="white", mew=1.9, label="2024", zorder=3)
ax.plot(x, meds["2025"], color=C_2025, lw=2.5, ls="-", marker="D",
        ms=6.2, mfc=C_2025, mec="white", mew=0.9, label="2025", zorder=4)

for i in range(len(labels)):
    y24, y25 = meds["2024"][i], meds["2025"][i]
    ratio = y25 / y24
    ax.text(i, y25 + 115, f"{y25:,.0f}", ha="center", va="bottom",
            fontsize=8.8, fontweight="bold", color=C_2025)
    ax.text(i, y24 - 135, f"{y24:,.0f}", ha="center", va="top",
            fontsize=8.5, color=C_MUTED)
    ax.text(i, (y24 + y25) / 2, f"×{ratio:.1f}", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="#6B3FA0",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.84))

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9.6, color=C_INK)
ax.set_xlabel("提问字符数", fontsize=10.0)
ax.set_ylabel("回答中位字符数", fontsize=10.0)
ax.set_ylim(0, max(meds["2025"]) * 1.22)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.grid(axis="y", color=C_GRID, lw=0.9)
ax.set_axisbelow(True)
ax.legend(loc="upper left", frameon=False, fontsize=9.5)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"fig0_length_drift.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("已生成:", OUT_DIR / "fig0_length_drift.pdf", "及 .png")
