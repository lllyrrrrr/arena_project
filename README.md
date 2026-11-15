# LMArena 人类偏好投票数据集

**是什么**：LMArena（原Chatbot Arena，UC Berkeley）是全球最大的大模型评测平台：用户提一个问题，两个匿名模型同时回答，用户投票谁更好（A胜/B胜/平局/都差）。本数据集是其开源的**13.5万张真实投票快照**（2025-04-17 ~ 07-24，53个模型，126种语言），每行=一张两两对战投票。来源 `lmarena-ai/arena-human-preference-140k`（HF，CC-BY-4.0），获取记录见 `data/DATA_MANIFEST.md`。

## 文件夹结构

```
arena_project/
├── data/
│   ├── battles_flat.parquet        ← 分析用这个！135634行×48列平表（已清洗展平）
│   ├── train-0000{0..6}.parquet    ← 原始数据7分片（勿动）
│   ├── preview_1000.csv            ← 前1000行预览，Excel可开
│   ├── DATASET_CARD.md             ← 官方字段定义
│   ├── DATA_MANIFEST.md            ← 来源/版本/抓取日期
│   └── download.sh                 ← 数据下载脚本
├── scripts/flatten.py              ← 原始→平表的转换脚本
└── .venv/                          ← Python3.13环境（pandas+pyarrow）
```

## 快速上手

```python
import pandas as pd
df = pd.read_parquet('data/battles_flat.parquet')   # 需要 pip install pandas pyarrow
df.head()
```

注意：跑代码请用项目自带环境 `.venv/bin/python`，全局python没装pandas。

## 48列速览（三组）

| 列组 | 内容 |
|---|---|
| 标识9列 | `id` `model_a` `model_b` `winner`(4类) `evaluation_session_id` `evaluation_order` `language` `is_code` `timestamp` |
| `meta_*` 26列 | 回答风格特征（官方预计算）：`sum_assistant_a/b_tokens`长度、`header_count`标题数、`list_count`列表数、`bold_count`加粗数等 |
| `tag_*` 12列 | 问题类别标注：math/creative_writing/IF指令遵循/7个criteria布尔标签 |
| `prompt` | 用户提示词原文（从嵌套对话中提取） |

## 使用注意点

**结构层（上手优势）**
- 官方预计算好回答风格特征（`meta_*`：回答长度/token数、markdown标题数、列表数、加粗数）和问题类别标签（`tag_*`：math/创意写作/指令遵循等布尔标签）——无需写文本处理代码，直接可分析
- `prompt` 已从嵌套对话提取为单列

**口径层（算之前先想清楚）**
1. 一行=一张票，但同一 `evaluation_session_id` 可能含多票——**观测不完全独立**，统计时按独立单位说明
2. `winner` 四类：model_a/model_b/tie/both_bad，平局+双差占27.5%——**"胜率"先定义分母**（全票？非平局票？）
3. `language` 含 `und`（未识别）约8.7%——业务意义的未知，不是缺失值
4. 顶层字段0缺失 ≠ 数据干净——und语言、空文本、极端长回答都要单独核查

**解读层（结论的边界）**
- 数据测的是**偏好不是正确**：长相特征可量化，内容质量不可直接测量——结论限于"形式特征与投票的关系"
- 时间窗仅为2025年4–7月快照，不代表当前模型版图
- 引用prompt示例注意license：用户内容CC-BY-4.0，模型输出归各厂商条款
