import pandas as pd
import numpy as np
from pathlib import Path

RES = Path("results")
OUT = []
def p(s=""):
    print(s); OUT.append(str(s))


REQUIRED_MD = [
    ("4.1 节 · 主描述统计", RES / "main_stats.md"),
    ("4.2 节 · 非参数解耦与共线性", RES / "dissociation_stats.md"),
    ("4.2 & 4.3 节 · BT 消融、聚类推断与整簇留出", RES / "bradley_terry_model_stats.md"),
    ("4.3 节 · 官方榜单外部校验", RES / "leaderboard_validation.md"),
    ("4.3 & 4.4 节 · 模型生态、补贴分解与作用边界", RES / "style_extension.md"),
    ("第 3 章 · 独立 vs 聚类灵敏度对照", RES / "cluster_check.md"),
    ("4.5 节 · 未分胜负票双轨机制", RES / "undecided_votes.md"),
]

missing = [str(path) for _, path in REQUIRED_MD if not path.exists()]
if missing:
    raise FileNotFoundError(f"请先依次跑完脚本 1–8，缺少产物: {missing}")

p("# 全篇数字总账（Numbers Sheet · 单一真源）")
p("")
p("> 口径声明：两期同月窗口 6 月 20 日–7 月 24 日、英文、严格单轮、三段文本非空；")
p("> 全部 Markdown 统计采用去代码块清洗口径；全部区间与检验均采用按评审/会话分簇的聚类稳健标准误。")

# ===== 第 2 章：逐级样本流核验（直接从 features.parquet 独立重算校验） =====
p("\n## 一、第 2 章 · 逐级样本流与算术闭合核验")
df = pd.read_parquet(
    RES / "features.parquet",
    columns=["source", "timestamp", "language", "turns", "winner",
             "q_chars", "a_chars", "b_chars", "cluster_id",
             "clean_a_bold", "clean_a_head", "clean_a_list",
             "clean_b_bold", "clean_b_head", "clean_b_list"])

W = [("t2 (2024)", "t2_100k", "2024-06-20", "2024-07-25"),
     ("t3 (2025)", "t3_140k", "2025-06-20", "2025-07-25")]

p("\n| 筛选层级 | t2 (2024) 票数 | t3 (2025) 票数 |")
p("|---|---:|---:|")
rows = {k: [] for k in [
    "全期原始选票", "对齐窗口 (6.20–7.24)", "限英文 (language=en)",
    "限单轮 (turns=1)", "三段非空 (主样本 main · 四票型)",
    "主样本聚类簇数 G", "分胜负票 (decisive · BT分母)",
    "其中平局 (tie)", "其中双差 (both_bad)",
    "字符不等分胜负票 (unequal · 胜率分母)", "其中分胜负等长剔除",
]}
for _, src, lo, hi in W:
    s0 = df[df.source == src]
    s1 = s0[(s0.timestamp >= lo) & (s0.timestamp < hi)]
    s2 = s1[s1.language == "en"]
    s3 = s2[s2.turns == 1]
    s4 = s3[(s3.q_chars > 0) & (s3.a_chars > 0) & (s3.b_chars > 0)]
    s5 = s4[s4.winner.isin(["model_a", "model_b"])]
    s6 = s5[s5.a_chars != s5.b_chars]
    n_tie = int((s4.winner == "tie").sum())
    n_bad = int((s4.winner == "both_bad").sum())
    n_eq = int((s5.a_chars == s5.b_chars).sum())
    assert len(s4) - n_tie - n_bad == len(s5) and len(s5) - n_eq == len(s6)
    vals = [len(s0), len(s1), len(s2), len(s3), len(s4),
            int(s4.cluster_id.nunique()), len(s5), n_tie, n_bad, len(s6), n_eq]
    for k, v in zip(rows.keys(), vals):
        rows[k].append(v)

for k, (v2, v3) in rows.items():
    p(f"| {k} | {v2:,} | {v3:,} |")

# ===== 4.3 节补充：模型间长度方差撕裂量化（直接从 model_ecology_t2/t3.csv 汇总） =====
p("\n## 二、4.3 节 · 模型间篇幅「方差撕裂」与榜单巨震对比")
p("\n| 指标 (模型层统计，不含 other) | t2 (2024) | t3 (2025) |")
p("|---|---:|---:|")
eco_stats = {}
for tag in ["t2", "t3"]:
    eco = pd.read_csv(RES / f"model_ecology_{tag}.csv")
    rk = pd.read_parquet(RES / f"ranks_{tag}.parquet")
    rk = rk[rk.model != "other"].set_index("model")
    eco = eco.set_index("model").join(rk[["rank0", "rank3"]])
    eco["rank_chg"] = eco.rank0 - eco.rank3
    eco_stats[tag] = {
        "模型数": f"{len(eco)}",
        "各模型平均字符数的均值": f"{eco.mean_chars.mean():.0f}",
        "各模型平均字符数的标准差 (方差撕裂)": f"{eco.mean_chars.std():.0f}",
        "各模型平均字符数 IQR (P75−P25)": f"{eco.mean_chars.quantile(0.75) - eco.mean_chars.quantile(0.25):.0f}",
        "模型平均长度优势标准差": f"{eco.len_adv.std():.3f}",
        "风格补贴标准差 (θ0−θ3)": f"{eco.subsidy.std():.3f}",
        "最大控风格名次升幅": f"+{int(eco.rank_chg.max())} ({eco.rank_chg.idxmax()})",
        "最大控风格名次降幅": f"{int(eco.rank_chg.min())} ({eco.rank_chg.idxmin()})",
        "名次绝对变动中位数 / 最大值": f"{eco.rank_chg.abs().median():.1f} / {int(eco.rank_chg.abs().max())}",
    }

for k in eco_stats["t2"].keys():
    p(f"| {k} | {eco_stats['t2'][k]} | {eco_stats['t3'][k]} |")

# ===== 4.5 节精选：平局 vs 双差核心 OR 对照表 =====
p("\n## 三、4.5 节 · 平局 vs 双差核心调整胜算比对照（聚类 95% CI）")
or_df = pd.read_csv(RES / "undecided_or.csv")
focus_terms = ["jaccard_per_0.1", "format_dist", "specificity", "creativity", "complexity"]
p("\n| 变量 | 期次 | 平局 OR [95% CI] (p) | 双差 OR [95% CI] (p) |")
p("|---|---|---|---|")
for term in focus_terms:
    for tag in ["t2", "t3"]:
        rt = or_df[(or_df.period == tag) & (or_df.outcome == "tie") & (or_df.term == term)].iloc[0]
        rb = or_df[(or_df.period == tag) & (or_df.outcome == "both_bad") & (or_df.term == term)].iloc[0]
        p(f"| `{term}` | {tag} | {rt['or']:.3f} [{rt['or_lo']:.3f}, {rt['or_hi']:.3f}] ({rt['p']:.1e}) "
          f"| {rb['or']:.3f} [{rb['or_lo']:.3f}, {rb['or_hi']:.3f}] ({rb['p']:.1e}) |")

# ===== 附：各脚本完整核验明细原样归档 =====
p("\n## 四、各脚本完整输出归档（按论文章节排序）")
for title, path in REQUIRED_MD:
    p(f"\n### {title} (`{path.as_posix()}`)")
    p("```text")
    p(path.read_text(encoding="utf-8").strip())
    p("```")

out_path = RES / "numbers_sheet.md"
out_path.write_text("\n".join(OUT), encoding="utf-8")
print(f"\n→ {out_path.as_posix()} (汇总校验通过)")
