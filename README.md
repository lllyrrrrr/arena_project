# LMArena 评测选票中的回答形式偏向与排名失真

上海财经大学「数据分析与可视化」课程期末项目。基于 LMArena（原 Chatbot Arena）两期公开盲测选票，检验回答篇幅与 Markdown 排版对投票结果的影响，以及这种形式偏向如何放大两期公开榜单的名次变动。

## 文件夹结构

```text
arena_project/
├── data/
│   ├── DATASET_CARD.md             ← 140k 官方字段定义
│   └── leaderboard_2026/           ← 已提取好的官方榜单对照 CSV（原始 parquet 见下方链接）
├── src/
│   ├── 0_aggregated_data.py        ← 合并 2024(100k) 与 2025(140k) 两期选票，统一语言码与票型
│   ├── 1_extract_features.py       ← 剥离代码块后提取字数与 Markdown 特征，输出 results/features.parquet
│   ├── 2_main_stats.py             ← 样本筛选、跨期漂移与基础统计
│   ├── 3_dissociation_stats.py     ← 长度差与排版密度的分层分离检验
│   ├── 4_bradley_terry_model_estimate.py  ← 控风格 Bradley-Terry 模型（M0–M3）、聚类稳健 SE 与 Bootstrap
│   ├── 5_bradley_terry_model_validate.py  ← 与 LMArena 官方基础榜/控风格榜交叉核验
│   ├── 6_style_extension.py        ← 计数/密度双口径、数学题交互项与数学子集重拟合
│   ├── 7_cluster_check.py          ← 同人/同会话重复投票聚类效应核查
│   ├── 8_undecided_votes.py        ← 平局（tie）与双差（both_bad）票机制分析
│   └── 9_numbers_sheet.py          ← 汇总各环节关键统计量
├── plot/
│   └── 0_length_drift.py ~ 10_cross_year_divergence.py  ← 11 张分析图脚本，输出至 results/figures/
├── results/                        ← 已跑出的统计汇总表（.md / .csv）与模型排名（ranks_t2/t3.parquet）
└── beamer/
    └── beamer_overleaf.tex         ← 汇报用 Beamer 源码（不含图与演讲稿）
```

> 原始选票数据（合计约 2.2 GB）以及大型中间表（`results/features.parquet` 等）超过 GitHub 体积限制未进仓库，按下方链接下载即可复现。

## 原始数据链接与主样本口径

| 数据 | 来源链接（Hugging Face） | 本地放置路径 |
|---|---|---|
| 2024 期盲测选票（100k，106,134 票） | [lmarena-ai/arena-human-preference-100k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-100k) | `data/arena_explorer_100k.parquet` |
| 2025 期盲测选票（140k，135,634 票） | [lmarena-ai/arena-human-preference-140k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k) | `data/train-0000{0..6}-of-00007.parquet` |
| 官方榜单快照（外部核验用） | [lmarena-ai/leaderboard-dataset](https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset) | 已提取为 CSV 放在 `data/leaderboard_2026/` |

- **主样本筛选**：两期合计 241,768 票，统一取当年 **6 月 20 日–7 月 24 日**（35 天）、英文、单轮对话、提问与双侧回答均非空，得到主样本 2024 年 26,469 票、2025 年 27,380 票。分胜负票用于偏好与排名建模，平局与双差票单独分析，不作噪声丢弃。

## 复现顺序

环境为 Python 3.13，主要依赖：`pandas`、`pyarrow`、`numpy`、`scipy`、`scikit-learn`、`matplotlib`。

原始数据按上表路径放好后，在仓库根目录依次执行：

1. **跑统计与建模**：按编号顺序运行 `src/0_aggregated_data.py` 到 `src/9_numbers_sheet.py`，结果写入 `results/`（汇总统计表与模型排名已直接附在仓库 `results/` 中，可直接查看）。
2. **跑可视化绘图**：运行 `plot/` 下各脚本（`0` 到 `10`），图表会同时保存为 `.pdf` 和 `.png` 到 `results/figures/`。

## Beamer 编译说明

`beamer/beamer_overleaf.tex` 推荐使用 **Overleaf（编译器选 XeLaTeX）** 或本地 TeXLive 编译，字体均为 TeXLive 内置（Fandol / TeX Gyre），无需额外安装字体。

仓库里只保留了 `.tex` 源码，没有上传图片，**直接编译会报缺图**，按下面两步配好图片即可：

1. **放入数据图**：先跑完 `src/` 与 `plot/`，从生成的 `results/figures/` 里挑出对应的 `.pdf` 图放到 `beamer/figures/`（Overleaf 中放在 `figures/` 目录）。其中 `fig0`–`fig6` 文件名完全一致；另有 4 张因调整过 PPT 页序，脚本输出名与 `.tex` 中引用的编号错位，放入前需对应改名：

| `plot/` 脚本输出（`results/figures/`） | `beamer_overleaf.tex` 引用名 |
|---|---|
| `fig10_cross_year_divergence.pdf` | `fig7_cross_year_divergence.pdf` |
| `fig7_logit_boundary.pdf` | `fig8_logit_boundary.pdf` |
| `fig8_undecided_heatmap.pdf` | `fig9_undecided_heatmap.pdf` |
| `fig9_official_validation.pdf` | `fig10_official_validation.pdf` |

2. **处理 3 张外部示意图**：P3 与 P6 引用了 `chatgpt_title_feedback.png`、`arena_profile.png`、`huggingface.png`，这三张未进仓库，自行补图到 `figures/` 或注释掉 `.tex` 中对应的 `\includegraphics` 行即可正常编译。

## 使用注意点

- **独立性**：同一评审（2024）/ 同一 `evaluation_session_id`（2025）可能含多张选票，观测不完全独立；建模已采用聚类稳健标准误与整簇 Bootstrap，样本外划分按人/会话分组。
- **代码块去污染**：提取 Markdown 标题/加粗/列表前已先剥离围栏与行内代码，避免代码注释里的 `#` 被误计为标题。
- **解读边界**：选票反映的是用户主观偏好而非客观正确；回答篇幅未经随机分配，结果代表统计关联，不能直接解读为因果效应。
- **许可提醒**：引用 prompt 示例时注意原数据条款——用户内容为 CC-BY-4.0，模型输出归各厂商条款。
