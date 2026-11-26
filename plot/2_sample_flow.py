"""图 2：2024 vs 2025 逐级严格闭合样本筛选与主样本三票型分流图（紫色系）"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

IN_FILE = Path("results/features.parquet")
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========== 紫色系配色（与全套图表统一） ==========
C_2024 = "#C9BCE6"      # 2024 筛选阶段条
C_2024_EC = "#8F7DC0"
C_2025 = "#7C5CB5"      # 2025 筛选阶段条
C_DEC = "#4B2882"       # 分胜负票（深皇家紫）
C_TIE = "#A995D4"       # 平局（薰衣草紫）
C_BAD = "#B54684"       # 双差（梅子紫红）
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


def round_sum_100(vals, decimals=1):
    """最大余数法：确保百分比严格相加为 100.0%"""
    scale = 10 ** decimals
    total_int = 100 * scale
    raw = np.asarray(vals, dtype=float)
    raw = raw / raw.sum() * total_int
    floored = np.floor(raw).astype(int)
    rem = total_int - int(floored.sum())
    order = np.argsort(-(raw - floored))
    for idx in order[:rem]:
        floored[idx] += 1
    return floored / scale


# ========== 读取数据并逐级严格闭合统计 ==========
cols = ["source", "timestamp", "language", "turns", "winner", "q_chars", "a_chars", "b_chars"]
df = pd.read_parquet(IN_FILE, columns=cols)

W = {
    "2024": ("t2_100k", "2024-06-20", "2024-07-25"),
    "2025": ("t3_140k", "2025-06-20", "2025-07-25"),
}

counts = {}
splits = {}
for yr, (src, lo, hi) in W.items():
    s0 = df[df.source == src]
    s1 = s0[(s0.timestamp >= lo) & (s0.timestamp < hi)]
    s2 = s1[s1.language == "en"]
    s3 = s2[(s2.turns == 1) & (s2.q_chars > 0) & (s2.a_chars > 0) & (s2.b_chars > 0)]
    dec = s3[s3.winner.isin(["model_a", "model_b"])]
    tie = s3[s3.winner == "tie"]
    bad = s3[s3.winner == "both_bad"]
    s4 = dec[dec.a_chars != dec.b_chars]

    assert len(dec) + len(tie) + len(bad) == len(s3), "主样本三票型未闭合"
    counts[yr] = [len(s0), len(s1), len(s2), len(s3), len(s4)]
    splits[yr] = (len(dec), len(tie), len(bad))

stages = [
    "原始全量",
    "同月窗口\n(6.20–7.24)",
    "限定英文",
    "单轮非空\n(主样本)",
    "分胜负不等长\n(胜率分母)",
]

# ========== 绘制：自上而下 5 级成对水平漏斗，第 4 级内置三票型堆叠 ==========
fig, ax = plt.subplots(figsize=(7.4, 4.4))
y = np.arange(len(stages))[::-1].astype(float)
h = 0.34
gap = 0.03
dy = {"2024": +(h / 2 + gap / 2), "2025": -(h / 2 + gap / 2)}

xmax = max(counts["2024"][0], counts["2025"][0])

for yr in ("2024", "2025"):
    c_list = counts[yr]
    raw_n = c_list[0]
    for i, val in enumerate(c_list):
        yi = y[i] + dy[yr]
        if i == 3:
            # 第 4 级（主样本）：直接按 分胜负 / 平局 / 双差 三段堆叠
            n_dec, n_tie, n_bad = splits[yr]
            pcts = round_sum_100([n_dec, n_tie, n_bad], decimals=1)
            left = 0.0
            for seg_v, seg_c in [(n_dec, C_DEC), (n_tie, C_TIE), (n_bad, C_BAD)]:
                ax.barh(yi, seg_v, left=left, height=h, color=seg_c,
                        edgecolor="white", lw=0.7, zorder=3)
                left += seg_v
            # 右侧紧跟主样本总数 + 精确闭合三占比
            ax.text(
                val + xmax * 0.012, yi,
                f"{yr}: {val:,}  (胜负 {pcts[0]:.1f}% / 平 {pcts[1]:.1f}% / 双差 {pcts[2]:.1f}%)",
                va="center", ha="left", fontsize=7.6,
                fontweight="bold" if yr == "2025" else "normal",
                color=C_DEC if yr == "2025" else C_MUTED,
            )
        else:
            fc = C_2024 if yr == "2024" else C_2025
            ec = C_2024_EC if yr == "2024" else "none"
            ax.barh(yi, val, height=h, color=fc, edgecolor=ec, lw=0.7, zorder=3)
            pct_raw = val / raw_n * 100.0
            lbl = f"{yr}: {val:,}" if i == 0 else f"{yr}: {val:,} ({pct_raw:.1f}%)"
            ax.text(
                val + xmax * 0.012, yi, lbl,
                va="center", ha="left", fontsize=7.8,
                fontweight="bold" if yr == "2025" else "normal",
                color=C_INK if yr == "2025" else C_MUTED,
            )

ax.set_yticks(y)
ax.set_yticklabels(stages, fontsize=9.2, color=C_INK)
ax.set_xlabel("选票数", fontsize=9.8)
ax.set_xlim(0, xmax * 1.24)
ax.set_ylim(-0.6, len(stages) - 0.4)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v/1000)}k" if v > 0 else "0"))
ax.grid(axis="x", color=C_GRID, lw=0.9)
ax.set_axisbelow(True)

ax.legend(
    handles=[
        Patch(facecolor=C_2024, edgecolor=C_2024_EC, lw=0.7, label="2024"),
        Patch(facecolor=C_2025, edgecolor="none", label="2025"),
        Patch(facecolor=C_DEC, edgecolor="white", label="分胜负"),
        Patch(facecolor=C_TIE, edgecolor="white", label="平局"),
        Patch(facecolor=C_BAD, edgecolor="white", label="双差"),
    ],
    loc="lower right", frameon=False, fontsize=8.3, ncol=3,
    handlelength=1.1, handletextpad=0.35, columnspacing=0.9,
)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"fig2_sample_flow.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("已生成:", OUT_DIR / "fig2_sample_flow.pdf", "及 .png")
