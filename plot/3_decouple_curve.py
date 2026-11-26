"""图 3：有符号相对长度差 (A−B)/均值 5 档 × 排版稀密分组柱状图（基线 50%，柔和紫系）"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

IN_FILE = Path("results/features.parquet")
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========== 柔和紫系三色（降饱和度、更通透） ==========
C_SPARSE = "#D18BB1"   # A 排版更稀：柔梅粉紫
C_ALL = "#8E73BF"      # 全部对局：柔皇家紫
C_DENSE = "#788BC7"    # A 排版更密：柔靛蓝紫
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

# ========== 读取 2025 分胜负样本（保留有符号长度差 (A−B)/均值） ==========
cols = [
    "source", "timestamp", "language", "turns", "winner", "cluster_id",
    "q_chars", "a_chars", "b_chars",
    "clean_a_bold", "clean_b_bold",
    "clean_a_head", "clean_b_head",
    "clean_a_list", "clean_b_list",
]
df = pd.read_parquet(IN_FILE, columns=cols)
d = df[
    (df.source == "t3_140k") & (df.timestamp >= "2025-06-20") & (df.timestamp < "2025-07-25") &
    (df.language == "en") & (df.turns == 1) &
    (df.q_chars > 0) & (df.a_chars > 0) & (df.b_chars > 0) &
    (df.winner.isin(["model_a", "model_b"]))
].copy()

d["win_a"] = (d.winner == "model_a").astype(float)
d["signed_gap"] = (d.a_chars - d.b_chars) / ((d.a_chars + d.b_chars) / 2.0)

den_a = (d.clean_a_bold + d.clean_a_head + d.clean_a_list) / d.a_chars * 1000.0
den_b = (d.clean_b_bold + d.clean_b_head + d.clean_b_list) / d.b_chars * 1000.0
d["den_diff_a"] = den_a - den_b

bins = [-np.inf, -0.60, -0.10, 0.10, 0.60, np.inf]
labels = ["<−60%", "−60%~−10%", "−10%~+10%", "+10%~+60%", ">+60%"]
x = np.arange(len(labels), dtype=float)


def bin_stats(sub):
    b = pd.cut(sub["signed_gap"], bins=bins, labels=labels, include_lowest=True)
    means, cis = [], []
    for lbl in labels:
        g = sub[b == lbl]
        y = g["win_a"].to_numpy(dtype=float)
        cid = g["cluster_id"].to_numpy()
        m = y.mean()
        s = pd.Series(y - m).groupby(cid).sum().to_numpy()
        G = len(s)
        se = np.sqrt((G / max(G - 1, 1)) * np.sum(s ** 2)) / len(y)
        means.append(m * 100.0)
        cis.append(1.96 * se * 100.0)
    return np.array(means), np.array(cis)


m_spa, c_spa = bin_stats(d[d.den_diff_a < -1e-9])
m_all, c_all = bin_stats(d)
m_den, c_den = bin_stats(d[d.den_diff_a > 1e-9])

# ========== 单图绘制：围绕 50% 基线的对称发散分组柱状图 ==========
fig, ax = plt.subplots(figsize=(7.2, 4.2))
w = 0.25
ax.axhline(50.0, color="#4A3C63", lw=1.0, zorder=3)

groups = [
    (-w, m_spa, c_spa, C_SPARSE, "A 格式密度更低"),
    (0.0, m_all, c_all, C_ALL,   "全部对局"),
    (+w, m_den, c_den, C_DENSE,  "A 格式密度更高"),
]

for dx, m, ci, col, lbl in groups:
    heights = m - 50.0
    ax.bar(
        x + dx, heights, bottom=50.0, width=w * 0.90,
        color=col, edgecolor="white", lw=0.7, alpha=0.92,
        label=lbl, zorder=2,
    )
    ax.errorbar(
        x + dx, m, yerr=ci, fmt="none",
        ecolor="#4A3C63", elinewidth=0.95, capsize=2.1, capthick=0.85, zorder=4,
    )

# 在两端与正中档标出关键数值（清晰呈现对称反转与近距翻边）
for i in (0, 2, 4):
    for dx, m, ci, col, _ in groups:
        if i == 2 and dx == 0.0:
            continue  # 中央档只标更稀与更密，突出对比
        val = m[i]
        txt_col = "#7A3A5E" if col == C_SPARSE else ("#4B2882" if col == C_ALL else "#3B4F9E")
        if val < 50.0:
            ax.text(x[i] + dx, val - ci[i] - 0.8, f"{val:.1f}%",
                    ha="center", va="top", fontsize=7.5, fontweight="bold", color=txt_col)
        else:
            ax.text(x[i] + dx, val + ci[i] + 0.8, f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=txt_col)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9.2, color=C_INK)
ax.set_xlabel("相对长度差 (A−B) / 均值", fontsize=9.8)
ax.set_ylabel("A 侧胜率（基线 50%，聚类 95% CI）", fontsize=9.6)
ax.set_ylim(24, 76)
ax.set_yticks([30, 40, 50, 60, 70])
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}%"))
ax.grid(axis="y", color=C_GRID, lw=0.9)
ax.set_axisbelow(True)
ax.legend(loc="upper left", frameon=False, fontsize=8.8)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"fig3_decouple_curve.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("已生成:", OUT_DIR / "fig3_decouple_curve.pdf", "及 .png")
