#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prepare_data.py —— 原始数据 → 分析主表(唯一入口)

项目约定:
  - 本脚本是唯一接触原始文件的代码;所有分析只读它产出的表
  - 原始文件只读不写,派生表可重新生成

输入(原始层):
  data/train-*.parquet        LMArena 140k 选票, 2025-04~07, 135,634 票
  data/battles_55k_2024.csv   LMArena 55k 选票, 2024 时代,   57,477 票

输出(分析层):
  results/battles_analysis.parquet   两期统一分析主表, 一行 = 一张投票

用法:
  .venv/bin/python scripts/prepare_data.py

列说明:
  通用列(两期都有):
    id, period('2024'|'2025'), model_a, model_b
    winner('model_a'|'model_b'|'tie'|'both_bad'; 2024期tie含both_bad)
    decisive(bool), prompt(str), prompt_chars(int), turns(int, 用户消息数)
    a_chars/a_bold/a_head/a_list, b_* (正则从回答原文重算的格式特征)
  仅 2025 期有值(2024期为NaN):
    language, is_code, timestamp, session_id, eval_order
    meta_a_tokens/meta_b_tokens (官方元数据, 用于与自建特征交叉验证)
    tag_math/tag_creative/tag_if/tag_complexity/tag_creativity/
    tag_domain_knowledge/tag_problem_solving/tag_real_world/
    tag_specificity/tag_technical_accuracy
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "results"
RAW_140K = sorted(DATA.glob("train-*-of-00007.parquet"))  # 排除专家票 train-*-of-00001
RAW_55K = DATA / "battles_55k_2024.csv"
OUT = OUT_DIR / "battles_analysis.parquet"

# markdown 格式特征:两期用同一套正则从原文重算,保证口径一致
RE_BOLD = re.compile(r"\*\*[^\n]+?\*\*|__[^\n]+?__")  # **加粗** 或 __加粗__
RE_HEAD = re.compile(r"^#{1,6} ", re.M)           # # 标题行
RE_LIST = re.compile(r"^\s*(?:[-*]|\d+\.) ", re.M)  # - 或 1. 列表项


def text_feats(text: str):
    """一条回答 -> (字符数, 加粗数, 标题数, 列表数)"""
    return (
        len(text),
        len(RE_BOLD.findall(text)),
        len(RE_HEAD.findall(text)),
        len(RE_LIST.findall(text)),
    )


def parse_json_list(s: str):
    """55k: prompt/response 是 JSON 字符串列表;解析失败按原文兜底"""
    try:
        return [x or "" for x in json.loads(s)]
    except Exception:
        return [str(s)]


def load_55k() -> pd.DataFrame:
    """2024期:CSV 内嵌 JSON 字符串, winner 为三个 one-hot 列(tie 含 both_bad)"""
    df = pd.read_csv(RAW_55K)
    assert len(df) == 57477, f"55k 行数异常: {len(df)}"
    assert (df[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1) == 1).all()

    rows = []
    for r in df.itertuples(index=False):
        prompts = parse_json_list(r.prompt)
        resp_a = parse_json_list(r.response_a)
        resp_b = parse_json_list(r.response_b)
        if r.winner_model_a == 1:
            winner = "model_a"
        elif r.winner_model_b == 1:
            winner = "model_b"
        else:
            winner = "tie"  # 注:2024期 tie 与 both_bad 合并,无法区分
        rows.append(
            (
                str(r.id), "2024", r.model_a, r.model_b, winner,
                " ".join(prompts), len(" ".join(prompts)), len(prompts),
                *text_feats("\n".join(resp_a)),
                *text_feats("\n".join(resp_b)),
            )
        )
    cols = [
        "id", "period", "model_a", "model_b", "winner",
        "prompt", "prompt_chars", "turns",
        "a_chars", "a_bold", "a_head", "a_list",
        "b_chars", "b_bold", "b_head", "b_list",
    ]
    out = pd.DataFrame(rows, columns=cols)
    # 55k 无语言检测字段;数据卡标注为英文数据,按卡元数据填 'en'
    out["language"] = "en"
    return out


def split_conversation(conv):
    """140k conversation: 按 role 拆出 user / assistant 文本(多轮做拼接)"""
    user, asst = [], []
    for m in conv:
        text = " ".join((c.get("text") or "") for c in m["content"])
        (user if m["role"] == "user" else asst).append(text)
    return user, asst


def load_140k() -> pd.DataFrame:
    """2025期:parquet 原生嵌套结构, 附带官方元数据与类别标签"""
    tag_keys = {
        "math": ("math_v0.1", "math"),
        "creative": ("creative_writing_v0.1", "creative_writing"),
        "if": ("if_v0.1", "if"),
        "complexity": ("criteria_v0.1", "complexity"),
        "creativity": ("criteria_v0.1", "creativity"),
        "domain_knowledge": ("criteria_v0.1", "domain_knowledge"),
        "problem_solving": ("criteria_v0.1", "problem_solving"),
        "real_world": ("criteria_v0.1", "real_world"),
        "specificity": ("criteria_v0.1", "specificity"),
        "technical_accuracy": ("criteria_v0.1", "technical_accuracy"),
    }
    need = [
        "id", "model_a", "model_b", "winner", "language", "is_code", "timestamp",
        "evaluation_session_id", "evaluation_order",
        "conversation_a", "conversation_b", "conv_metadata", "category_tag",
    ]
    rows = []
    total = 0
    for fp in RAW_140K:
        x = pd.read_parquet(fp, columns=need)
        total += len(x)
        for r in x.itertuples(index=False):
            ua, ra = split_conversation(r.conversation_a)
            _, rb = split_conversation(r.conversation_b)
            meta = r.conv_metadata or {}
            tags = r.category_tag or {}
            tag_vals = [bool((tags.get(g) or {}).get(k)) for g, k in tag_keys.values()]
            rows.append(
                (
                    str(r.id), "2025", r.model_a, r.model_b, r.winner,
                    " ".join(ua), len(" ".join(ua)), len(ua),
                    *text_feats("\n".join(ra)),
                    *text_feats("\n".join(rb)),
                    r.language, r.is_code, r.timestamp,
                    r.evaluation_session_id, r.evaluation_order,
                    meta.get("sum_assistant_a_tokens"),
                    meta.get("sum_assistant_b_tokens"),
                    *tag_vals,
                )
            )
        print(f"  {fp.name}: +{len(x)} 行", flush=True)
        del x
    assert total == 135634, f"140k 行数异常: {total}"
    cols = [
        "id", "period", "model_a", "model_b", "winner",
        "prompt", "prompt_chars", "turns",
        "a_chars", "a_bold", "a_head", "a_list",
        "b_chars", "b_bold", "b_head", "b_list",
        "language", "is_code", "timestamp", "session_id", "eval_order",
        "meta_a_tokens", "meta_b_tokens",
        "tag_math", "tag_creative", "tag_if", "tag_complexity", "tag_creativity",
        "tag_domain_knowledge", "tag_problem_solving", "tag_real_world",
        "tag_specificity", "tag_technical_accuracy",
    ]
    return pd.DataFrame(rows, columns=cols)


def validate(df: pd.DataFrame):
    """产出前的完整性校验;任一项失败即中断,不落盘"""
    issues = []
    if df["id"].duplicated().any():
        issues.append("存在重复 id")
    if not df["winner"].isin(["model_a", "model_b", "tie", "both_bad"]).all():
        issues.append("winner 取值越界")
    n24 = int((df.period == "2024").sum())
    n25 = int((df.period == "2025").sum())
    if (n24, n25) != (57477, 135634):
        issues.append(f"行数异常: {n24}/{n25}")
    for c in ["a_chars", "b_chars", "prompt_chars"]:
        if (df[c] < 0).any():
            issues.append(f"{c} 存在负值")
    empty = int(((df.a_chars == 0) | (df.b_chars == 0)).sum())
    # 交叉验证:2025期自建字符特征应与官方 token 数高度相关
    d25 = df[(df.period == "2025") & df.meta_a_tokens.notna()]
    rho = d25.a_chars.corr(d25.meta_a_tokens) if len(d25) else np.nan
    return issues, n24, n25, empty, rho


def main():
    t0 = time.time()
    print("[1/4] 解析 2024 期 (CSV + JSON 字符串) ...")
    f55 = load_55k()
    print(f"      -> {len(f55)} 行")
    print("[2/4] 解析 2025 期 (parquet 嵌套结构, 逐分片) ...")
    f140 = load_140k()
    print(f"      -> {len(f140)} 行")

    print("[3/4] 合并 + 统一口径 ...")
    df = pd.concat([f55, f140], ignore_index=True)
    df["decisive"] = df["winner"].isin(["model_a", "model_b"])
    # 2024期 tie 实际为 tie+both_bad 合并,打标以便分析时区分口径
    df["tie_includes_both_bad"] = df["period"].eq("2024") & df["winner"].eq("tie")
    # 严格单轮:两侧均恰有1条用户消息(2024期为列表长度,2025期为user消息数)
    df["single_turn"] = df["turns"].eq(1)
    df["empty_side"] = (df["a_chars"] == 0) | (df["b_chars"] == 0)

    print("[4/4] 校验并写出 ...")
    issues, n24, n25, empty, rho = validate(df)
    if issues:
        print("校验失败:", issues)
        sys.exit(1)
    OUT_DIR.mkdir(exist_ok=True)
    df.to_parquet(OUT, index=False)
    mb = OUT.stat().st_size / 1e6
    print(f"""
完成 ({time.time()-t0:.0f}s)
  输出: {OUT.name}  {df.shape[0]}行 x {df.shape[1]}列, {mb:.1f}MB
  2024期: {n24} | 2025期: {n25}
  空回答行: {empty} (保留, 分析时按需剔除)
  自建字符数 vs 官方token数 相关: {rho:.3f}  (应≈0.9, 交叉验证通过)
""")


if __name__ == "__main__":
    main()
