import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("data")
OUT = Path("results"); OUT.mkdir(exist_ok=True)

# ============ 1. 100k (t2: 2024.6–8) ============
d2 = pd.read_parquet(DATA / "arena_explorer_100k.parquet")
d2 = d2.rename(columns={"question_id": "battle_id", "turn": "turns"})
d2["source"] = "t2_100k"
d2["timestamp"] = pd.to_datetime(d2["tstamp"], unit="s")
d2["winner"] = d2["winner"].replace({"tie (bothbad)": "both_bad"})

# 语言全名 → ISO 码（与140k数据保持一致）
LANG_MAP = {
    "English": "en", "Russian": "ru", "Chinese": "zh", "Japanese": "ja",
    "Vietnamese": "vi", "German": "de", "Korean": "ko", "French": "fr",
    "Spanish": "es", "Portuguese": "pt", "Polish": "pl", "Italian": "it",
    "Turkish": "tr", "Ukrainian": "uk", "Arabic": "ar", "Hindi": "hi",
    "Thai": "th", "Dutch": "nl", "Czech": "cs", "Indonesian": "id",
    "Swedish": "sv", "Romanian": "ro", "Hungarian": "hu", "Greek": "el",
    "Welsh": "cy", "Catalan": "ca", "Interlingua": "ia", "Slovenian": "sl",
    "Maltese": "mt", "Croatian": "hr", "Swahili": "sw", "Malay": "ms",
    "Quechua": "qu", "Interlingue": "ie", "Bangla": "bn",
    "Kinyarwanda": "rw", "Slovak": "sk", "Wolof": "wo", "Yoruba": "yo",
    "Uzbek": "uz", "Manx": "gv", "Faroese": "fo", "Bulgarian": "bg",
    "Occitan": "oc", "Icelandic": "is", "Tigrinya": "ti",
    "Western Frisian": "fy", "Latvian": "lv", "Somali": "so",
    "Cebuano": "ceb", "Bislama": "bi", "Tsonga": "ts", "Aymara": "ay",
    "Afar": "aa", "Kalaallisut": "kl", "Galician": "gl", "Sanskrit": "sa",
    "Fijian": "fj", "Corsican": "co", "Hebrew": "he", "Tswana": "tn",
    "Scots": "sco", "Danish": "da", "Irish": "ga", "Esperanto": "eo",
    "Lithuanian": "lt", "Klingon": "tlh", "Southern Sotho": "st",
    "Latin": "la", "Ganda": "lg", "Xhosa": "xh", "Oromo": "om",
    "Hawaiian": "haw", "Sundanese": "su", "Volapük": "vo",
    "Estonian": "et", "Tatar": "tt", "Afrikaans": "af",
    "Macedonian": "mk", "Norwegian": "no", "Norwegian Nynorsk": "nn",
    "Basque": "eu", "Serbian": "sr", "Guarani": "gn",
    "Haitian Creole": "ht", "Luxembourgish": "lb", "Finnish": "fi",
    "Persian": "fa", "Malagasy": "mg",
    "unknown": "und",
}
unmapped = set(d2.language.dropna().unique()) - set(LANG_MAP)
if unmapped:
    print("⚠️ 还有未映射的语言名，请补进 LANG_MAP:", unmapped)
d2["language"] = d2["language"].map(lambda x: LANG_MAP.get(x, "und"))

# ============ 2. 140k (t3: 2025.4–7) ============
files = sorted(DATA.glob("train-0000*-of-00007.parquet"))
d3 = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
d3 = d3.rename(columns={"id": "battle_id"})
d3["source"] = "t3_140k"
d3["turns"] = d3["conv_metadata"].map(lambda m: m["turns"])

# ============ 3. 按统一列集合拼装 ============
def norm_conv(c):
    """统一对话结构：外层转list；content统一为内容块列表
    100k: content 是纯字符串 → 包成 [{'type':'text','text':...}]
    140k: content 是 ndarray/list 的内容块 → 转 list"""
    if not isinstance(c, (list, np.ndarray)):
        return None
    msgs = []
    for m in c:
        m = dict(m)
        cont = m.get("content")
        if isinstance(cont, np.ndarray):
            cont = list(cont)
        if isinstance(cont, str):
            cont = [{"type": "text", "text": cont}]
        m["content"] = cont
        msgs.append(m)
    return msgs

for col in ["conversation_a", "conversation_b"]:
    d2[col] = d2[col].map(norm_conv)
    d3[col] = d3[col].map(norm_conv)

common = ["battle_id", "source", "timestamp", "model_a", "model_b",
          "winner", "turns", "conversation_a", "conversation_b",
          "language", "is_code", "conv_metadata", "category_tag"]

t2 = d2.reindex(columns=common + ["anony", "is_refusal", "dedup_tag"])
t2["evaluation_session_id"] = pd.NA
t2["evaluation_order"] = pd.NA

t3 = d3.reindex(columns=common + ["evaluation_session_id", "evaluation_order"])
t3["anony"] = pd.NA
t3["is_refusal"] = pd.NA
t3["dedup_tag"] = pd.NA

unified = pd.concat([t2, t3], ignore_index=True)
unified["battle_id"] = unified["source"].str[:2] + "_" + unified["battle_id"].astype(str)

# ============ 4. 检查 ============
print("总行数:", len(unified))
print(unified.groupby("source").agg(
    票数=("battle_id", "size"),
    起始=("timestamp", "min"), 截止=("timestamp", "max")))
print("\nwinner 分布(按期):"); print(unified.groupby("source").winner.value_counts())
print("\n部分列缺失率(应为 1.0 的一侧):")
print(unified.groupby("source")[["evaluation_session_id", "is_refusal"]]
      .apply(lambda g: g.isna().mean()))

unified.to_parquet(OUT / "aggregated_data.parquet", index=False)
print("\n已写入 results/aggregated_data.parquet")