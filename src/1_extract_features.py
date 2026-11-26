import pandas as pd
import numpy as np
import re
from pathlib import Path

IN = Path("results/aggregated_data.parquet")
OUT = Path("results/features.parquet")

BOLD_RE = re.compile(r"\*\*[^\n*]+?\*\*|__[^\n_]+?__")
HEAD_RE = re.compile(r"^\s{0,3}#{1,6}\s", re.M)
LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s", re.M)
FENCE_RE = re.compile(r"(?ms)^\s*(`{3,}|~{3,})[^\n]*\n.*?^\s*\1[^\n]*(?:\n|$)")
INLINE_RE = re.compile(r"(`+)[^\n]*?\1")
WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?|\d+")

CRITERIA = [
    "complexity", "creativity", "domain_knowledge", "problem_solving",
    "real_world", "specificity", "technical_accuracy",
]


def strip_code(text):
    """剥离围栏代码块与行内代码，避免代码注释 '#' 等被误计为 Markdown 排版"""
    return INLINE_RE.sub(" ", FENCE_RE.sub("\n", text))


def jaccard_words(text_a, text_b):
    """双方回答小写英文词元/数字集合的 Jaccard 重合度"""
    if not text_a or not text_b:
        return np.nan
    wa = set(WORD_RE.findall(text_a.lower()))
    wb = set(WORD_RE.findall(text_b.lower()))
    union = len(wa | wb)
    return len(wa & wb) / union if union > 0 else np.nan


def texts_by_role(conv, role):
    """收集某角色所有消息中 type=='text' 的块,拼接为一个字符串"""
    if not isinstance(conv, (list, np.ndarray)) or len(conv) == 0:
        return ""
    parts = []
    for msg in conv:
        if isinstance(msg, dict) and msg.get("role") == role:
            cont = msg.get("content")
            if not isinstance(cont, (list, np.ndarray)):
                continue
            for blk in cont:
                if isinstance(blk, dict) and blk.get("type") == "text":
                    parts.append(blk.get("text") or "")
    return "\n".join(parts)


def nontext_blocks(conv):
    if not isinstance(conv, (list, np.ndarray)):
        return 0
    n = 0
    for msg in conv:
        if not isinstance(msg, dict):
            continue
        cont = msg.get("content")
        if not isinstance(cont, (list, np.ndarray)):
            continue
        n += sum(1 for blk in cont
                 if isinstance(blk, dict) and blk.get("type") != "text")
    return n


def tag_flag(d, block, keys):
    """从 category_tag 里取布尔任务标签;缺块/缺键返回 NaN"""
    if not isinstance(d, dict):
        return np.nan
    v = d.get(block)
    if isinstance(v, dict):
        for k in keys:
            if k in v:
                return bool(v[k])
        return np.nan
    return np.nan if v is None else bool(v)


# ============ 1. 文本提取 ============
df = pd.read_parquet(IN)

pair = df["conversation_a"].map(
    lambda c: (texts_by_role(c, "user"), texts_by_role(c, "assistant")))
df["user_text"], df["resp_a_text"] = zip(*pair)
df["resp_b_text"] = df["conversation_b"].map(
    lambda c: texts_by_role(c, "assistant"))

# ============ 2. 长度特征 ============
df["q_chars"] = df.user_text.str.len()
df["a_chars"] = df.resp_a_text.str.len()
df["b_chars"] = df.resp_b_text.str.len()

# ============ 3. 格式特征 (原始 + 去代码块清洗版) ============
for s, col in [("a", "resp_a_text"), ("b", "resp_b_text")]:
    df[f"{s}_bold"] = df[col].map(lambda t: len(BOLD_RE.findall(t)))
    df[f"{s}_head"] = df[col].map(lambda t: len(HEAD_RE.findall(t)))
    df[f"{s}_list"] = df[col].map(lambda t: len(LIST_RE.findall(t)))
    cleaned = df[col].map(strip_code)
    df[f"clean_{s}_bold"] = cleaned.map(lambda t: len(BOLD_RE.findall(t)))
    df[f"clean_{s}_head"] = cleaned.map(lambda t: len(HEAD_RE.findall(t)))
    df[f"clean_{s}_list"] = cleaned.map(lambda t: len(LIST_RE.findall(t)))

# ============ 3.5 双方回答用词重合度 (Jaccard, 支撑平局/双差机制) ============
df["word_overlap"] = [
    jaccard_words(a, b) for a, b in zip(df.resp_a_text, df.resp_b_text)
]

# ============ 4. 空值/非文本诊断 ============
df["q_empty"] = df.q_chars == 0
df["empty_side"] = np.select(
    [(df.a_chars == 0) & (df.b_chars == 0), df.a_chars == 0, df.b_chars == 0],
    ["both", "a", "b"], default="none")
df["n_nontext"] = (df.conversation_a.map(nontext_blocks)
                   + df.conversation_b.map(nontext_blocks))

# ============ 5. 任务标签 + 七维提问属性摊平 ============
df["tag_math"] = df.category_tag.map(lambda d: tag_flag(d, "math_v0.1", ["math"]))
df["tag_if"] = df.category_tag.map(lambda d: tag_flag(d, "if_v0.1", ["if"]))
df["tag_writing"] = df.category_tag.map(
    lambda d: tag_flag(d, "creative_writing_v0.1",
                       ["creative_writing", "creative_writing_v0.1"]))
crit_cols = []
for c in CRITERIA:
    cname = f"crit_{c}"
    df[cname] = df.category_tag.map(lambda d, k=c: tag_flag(d, "criteria_v0.1", [k]))
    crit_cols.append(cname)
df["crit_count"] = np.where(
    df[crit_cols].notna().all(axis=1),
    df[crit_cols].astype(float).sum(axis=1),
    np.nan)

# ============ 6. 官方token数(交叉校验用) ============
for new, key in [("a_tok_off", "sum_assistant_a_tokens"),
                 ("b_tok_off", "sum_assistant_b_tokens"),
                 ("q_tok_off", "sum_user_tokens")]:
    df[new] = df.conv_metadata.map(
        lambda m: m.get(key) if isinstance(m, dict) else np.nan)

# ============ 6.5 聚类标识: t2用judge_hash, t3用evaluation_session_id ============
has_t2_cl = (df.source == "t2_100k") & df.judge_hash.notna()
has_t3_cl = (df.source == "t3_140k") & df.evaluation_session_id.notna()
df["cluster_id"] = np.select(
    [has_t2_cl, has_t3_cl],
    ["t2_" + df.judge_hash.astype(str), "t3_" + df.evaluation_session_id.astype(str)],
    default="solo_" + df.battle_id.astype(str))
print("聚类标识: t2 簇数", df.loc[df.source == "t2_100k", "cluster_id"].nunique(),
      "| t3 簇数", df.loc[df.source == "t3_140k", "cluster_id"].nunique())

# ============ 7. 检查 ============
print("总行数:", len(df))
print("空文本分布:", df.empty_side.value_counts().to_dict())
print("提问为空:", int(df.q_empty.sum()))
print("含非text块的票:", int((df.n_nontext > 0).sum()),
      "| 其中纯非文本提问:", int(((df.q_chars == 0) & (df.n_nontext > 0)).sum()))
print("标签覆盖率: math %.3f | if %.3f | writing %.3f | crit_all %.3f" % (
    df.tag_math.notna().mean(), df.tag_if.notna().mean(),
    df.tag_writing.notna().mean(), df.crit_count.notna().mean()))
print("用词重合度有效覆盖率: %.3f | 中位数: %.3f" % (
    df.word_overlap.notna().mean(), df.word_overlap.median()))
print("去代码块后标题计数变化(均值): A %.2f→%.2f | B %.2f→%.2f" % (
    df.a_head.mean(), df.clean_a_head.mean(),
    df.b_head.mean(), df.clean_b_head.mean()))
corr = df[["a_chars", "a_tok_off"]].dropna()
print("自建字符 vs 官方token 相关:", round(corr.corr().iloc[0, 1], 3))

keep = ["battle_id", "source", "timestamp", "model_a", "model_b", "winner",
        "turns", "language", "is_code", "evaluation_session_id",
        "evaluation_order", "is_refusal", "anony", "judge_hash", "cluster_id",
        "q_chars", "a_chars", "b_chars",
        "a_bold", "a_head", "a_list", "b_bold", "b_head", "b_list",
        "clean_a_bold", "clean_a_head", "clean_a_list",
        "clean_b_bold", "clean_b_head", "clean_b_list",
        "word_overlap",
        "q_empty", "empty_side", "n_nontext",
        "tag_math", "tag_if", "tag_writing",
        *crit_cols, "crit_count",
        "a_tok_off", "b_tok_off", "q_tok_off",
        "user_text", "resp_a_text", "resp_b_text"]
df[keep].to_parquet(OUT, index=False)
print("\n已写入 results/features.parquet,", len(keep), "列")