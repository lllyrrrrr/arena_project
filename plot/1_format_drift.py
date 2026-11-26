"""图 1：标题 / 加粗 / 列表三类 Markdown 使用占比按提问长度分档的单图对比（2024 vs 2025）"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# ========== 路径 ==========
IN_FILE = Path("results/features.parquet")
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========== 和谐紫系邻近色（靛蓝紫 / 皇家紫 / 梅子紫红） ==========
PALETTE = {
    "list": ("#3B4F9E", "列表"),
    "bold": ("#6D28D9", "加粗"),
    "head": ("#B54684", "标题"),
}
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
cols = [
    "source", "timestamp", "language", "turns",
    "q_chars", "a_chars", "b_chars",
    "clean_a_bold", "clean_b_bold",
    "clean_a_head", "clean_b_head",
    "clean_a_list", "clean_b_list",
]
df = pd.read_parquet(IN_FILE, columns=cols)

W = {
    "2024": ("t2_100k", "2024-06-20", "2024-07-25"),
    "2025": ("t3_140k", "2025-06-20", "2025-07-25"),
}
bins = [0, 30, 80, 200, 500, np.inf]
labels = ["≤30", "31–80", "81–200", "201–500", ">500"]
keys = ["list", "bold", "head"]

curves = {k: {} for k in keys}
overall = {k: {} for k in keys}

for yr, (src, lo, hi) in W.items():
    sub = df[
        (df.source == src) & (df.timestamp >= lo) & (df.timestamp < hi) &
        (df.language == "en") & (df.turns == 1) &
        (df.q_chars > 0) & (df.a_chars > 0) & (df.b_chars > 0)
    ]
    q = sub.q_chars.to_numpy(dtype=float)
    q_pair = np.concatenate([q, q])
    b = pd.cut(q_pair, bins=bins, labels=labels)

    for k in keys:
        used = np.concatenate([
            (sub[f"clean_a_{k}"] > 0).to_numpy(dtype=float),
            (sub[f"clean_b_{k}"] > 0).to_numpy(dtype=float),
        ]) * 100.0
        overall[k][yr] = float(used.mean())
        g = pd.DataFrame({"b": b, "u": used}).groupby("b", observed=True)["u"].mean()
        curves[k][yr] = g.reindex(labels).to_numpy()

# ========== 单图绘制 ==========
fig, ax = plt.subplots(figsize=(6.8, 4.1))
x = np.arange(len(labels))

# 先收集右侧 2025 端点坐标，自动做防重叠微调
end_points = []
for k in keys:
    col, cn = PALETTE[k]
    y24 = curves[k]["2024"]
    y25 = curves[k]["2025"]

    ax.fill_between(x, y24, y25, color=col, alpha=0.09, lw=0, zorder=1)
    ax.plot(x, y24, color=col, lw=1.7, ls="--", marker="o",
            ms=5.4, mfc="white", mew=1.5, alpha=0.78, zorder=3)
    ax.plot(x, y25, color=col, lw=2.3, ls="-", marker="D",
            ms=5.2, mfc=col, mec="white", mew=0.8, zorder=4)

    end_points.append([float(y25[-1]), k, col, cn])

# 右侧标签纵向防重叠（若两标签间距 < 6.5pp 自动错开）
end_points.sort(key=lambda t: t[0])
for i in range(1, len(end_points)):
    if end_points[i][0] - end_points[i - 1][0] < 6.8:
        mid = (end_points[i][0] + end_points[i - 1][0]) / 2.0
        end_points[i - 1][0] = mid - 3.6
        end_points[i][0] = mid + 3.6

for y_lbl, k, col, cn in end_points:
    o24, o25 = overall[k]["2024"], overall[k]["2025"]
    ax.text(
        4.14, y_lbl, f"{cn} {o24:.0f}%→{o25:.0f}%",
        ha="left", va="center", fontsize=8.6, fontweight="bold", color=col,
    )

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9.6, color=C_INK)
ax.set_xlabel("提问字符数", fontsize=10.0)
ax.set_ylabel("使用率（去代码块）", fontsize=10.0)
ax.set_xlim(-0.22, 5.18)
ax.set_ylim(0, 96)
ax.set_yticks([0, 20, 40, 60, 80])
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}%"))
ax.grid(axis="y", color=C_GRID, lw=0.9)
ax.set_axisbelow(True)

# 极简图例：仅说明线型对应的年份（颜色已由右侧直接标注）
ax.legend(
    handles=[
        Line2D([0], [0], color=C_MUTED, lw=1.7, ls="--", marker="o", ms=5.2, mfc="white", mew=1.5, label="2024"),
        Line2D([0], [0], color=C_INK, lw=2.3, ls="-", marker="D", ms=5.0, mfc=C_INK, mec="white", mew=0.7, label="2025"),
    ],
    loc="upper left", frameon=False, fontsize=9.2, ncol=2, columnspacing=1.2,
)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"fig1_format_drift.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("已生成:", OUT_DIR / "fig1_format_drift.pdf", "及 .png")
