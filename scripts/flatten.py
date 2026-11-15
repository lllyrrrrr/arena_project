"""把原始arena数据拍平成分析宽表 -> data/battles_flat.parquet"""
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
df = pd.concat([pd.read_parquet(f) for f in sorted(DATA.glob('train-*.parquet'))],
               ignore_index=True)

# 1) 拍平 conv_metadata -> meta_* 列；category_tag -> tag_* 列
meta = pd.json_normalize(df['conv_metadata']).add_prefix('meta_')
tags = pd.json_normalize(df['category_tag']).add_prefix('tag_')

# 2) 从嵌套对话里只取"用户最后一句提示词"——其余原样丢弃
def last_user_text(convo):
    for msg in reversed(convo):
        if msg['role'] == 'user':
            return ''.join(c.get('text') or '' for c in msg['content'])
    return None

flat = pd.concat([
    df[['id','model_a','model_b','winner','evaluation_session_id',
        'evaluation_order','language','is_code','timestamp']],
    meta, tags,
    df['conversation_a'].apply(last_user_text).rename('prompt'),
], axis=1)

flat.to_parquet(DATA / 'battles_flat.parquet')
print("saved:", DATA / 'battles_flat.parquet')
print("shape:", flat.shape)
print("new cols:", [c for c in flat.columns if c.startswith(('meta_','tag_'))])
