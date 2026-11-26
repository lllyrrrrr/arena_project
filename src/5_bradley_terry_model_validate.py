import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path

OUT = []
def p(s=""):
    print(s); OUT.append(str(s))

# 人工核验后的别名: 本研究数据中的名字 -> 官方名字(同一模型不同写法)
# 版本不同的同族模型不映射(gemini-2.5-pro-preview != gemini-2.5-pro)
ALIAS = {
    "amazon.nova-pro-v1:0": "amazon-nova-pro-v1.0",
}

PLAN = {
    "t2": ("results/ranks_t2.parquet",
           "data/leaderboard_2026/lb_2024_full_overall.csv",
           "data/leaderboard_2026/lb_2024_style_overall.csv"),
    "t3": ("results/ranks_t3.parquet",
           "data/leaderboard_2026/lb_2025_full_english.csv",
           "data/leaderboard_2026/lb_2025_style_english.csv"),
}

for tag, (rk_f, full_f, style_f) in PLAN.items():
    ours = (pd.read_parquet(rk_f)
              .query("model != 'other'")
              .assign(model=lambda d: d.model.replace(ALIAS))
              [["model", "rank0", "rank3"]])
    assert not ours.model.duplicated().any(), "别名映射产生重名"

    def board(f, col):
        return (pd.read_csv(f)[["model_name", "rank"]]
                  .rename(columns={"model_name": "model", "rank": col})
                  .dropna())

    mf = ours.merge(board(full_f, "off_full"), on="model")
    ms = ours.merge(board(style_f, "off_style"), on="model")
    # 名次变动只在交集内比较: 先在交集中重新排名,再求delta
    # (两边榜的模型全集不同,直接用原名次会把交集外模型的挪动算进来)
    mb = mf.merge(board(style_f, "off_style"), on="model")
    for c in ["rank0", "rank3", "off_full", "off_style"]:
        mb[c] = mb[c].rank()
    mb = mb.assign(d_ours=lambda d: d.rank0 - d.rank3,
                   d_off=lambda d: d.off_full - d.off_style)

    p(f"\n===== {tag} =====")
    p(f"交集: 基础榜对照 n={len(mf)} / 风格榜对照 n={len(ms)} / 变动对照 n={len(mb)}")
    unmatched = sorted(set(ours.model) - set(mf.model))
    p(f"未匹配({len(unmatched)}): {unmatched}")

    for lab, mm, ca, cb in [
        ("本研究基础榜 vs 官方full",   mf, "rank0", "off_full"),
        ("本研究控风格榜 vs 官方style", ms, "rank3", "off_style"),
        ("名次变动向量(delta)",        mb, "d_ours", "d_off"),
    ]:
        rho, pv = spearmanr(mm[ca], mm[cb])
        p(f"  {lab:28s} rho={rho:.3f}  p={pv:.2e}")

    nz = mb[mb.d_ours != 0]
    agree = (np.sign(nz.d_ours) == np.sign(nz.d_off)).mean()
    p(f"  变动方向一致率: {agree*100:.0f}% (n={len(nz)})")

    # 大变动子集: 同时报告本研究单侧 |d_ours|>=3 与对称并集 (|d_ours|>=3 或 |d_off|>=3)
    for sub_lab, sub_df in [
        ("本研究|delta|>=3", mb[mb.d_ours.abs() >= 3]),
        ("对称大变动(|d_ours|>=3或|d_off|>=3)", mb[(mb.d_ours.abs() >= 3) | (mb.d_off.abs() >= 3)]),
    ]:
        if len(sub_df):
            agr_s = (np.sign(sub_df.d_ours) == np.sign(sub_df.d_off)).mean()
            rho_s, pv_s = spearmanr(sub_df.d_ours, sub_df.d_off)
            p(f"  {sub_lab}: 方向一致率 {agr_s*100:.0f}% "
              f"(n={len(sub_df)})  delta rho={rho_s:.3f} p={pv_s:.2e}")

    mb.to_csv(f"results/common_ranks_{tag}.csv", index=False, encoding="utf-8-sig")

    p("  变动最大的共同模型 (交集内重排, +=控风格后上升):")
    top = mb.assign(a=mb.d_ours.abs()).nlargest(12, "a")
    for _, r in top.iterrows():
        flag = "OK" if np.sign(r.d_ours) == np.sign(r.d_off) else "XX"
        p(f"  {flag} {r.model[:36]:36s} 本研究{r.d_ours:+.0f} 官方{r.d_off:+.0f}")

Path("results/leaderboard_validation.md").write_text(
    "\n".join(OUT), encoding="utf-8")
print("\n-> results/leaderboard_validation.md")