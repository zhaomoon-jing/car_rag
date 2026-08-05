from FlagEmbedding import FlagReranker
from retriever.dense_retriever import dense_search
from retriever.bm25_retriever import bm25_search

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import RERANK_MODEL_NAME, TOP_K_RERANK

cache_dir = "/root/autodl-tmp/models" 
modelscope_model_id = RERANK_MODEL_NAME

#本地路径格式化
name_lst = RERANK_MODEL_NAME.split("/")
model_format_name= name_lst[0] +"--"+ name_lst[1]
local_dir = Path("/root/autodl-tmp/models/models"+ model_format_name)


# 判断本地是否存在完整模型（核心：检测config.json）
config_paths = list(local_dir.rglob("config.json"))

if local_dir.exists() and config_paths[0].exists():
    print(f"✅ 检测到本地模型，离线加载：{local_dir}")
    LOCAL_RERANK_MODEL_NAME = RERANK_MODEL_NAME
    reranker = FlagReranker(LOCAL_RERANK_MODEL_NAME, use_fp16=True)
else:
    from modelscope import snapshot_download
    model_dir = snapshot_download(RERANK_MODEL_NAME,cache_dir=cache_dir)
    reranker = FlagReranker(model_dir, use_fp16=True)


def rerank_filter(query, candidate_chunks):
    pairs = [(query, item["text"]) for item in candidate_chunks]
    scores = reranker.compute_score(pairs)
    # 绑定分数+原文，排序取top
    scored = list(zip(scores, candidate_chunks))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [x[1] for x in scored[:TOP_K_RERANK]]
    return top

def hybrid_retrieve(query):
    # 混合检索
    dense_res = dense_search(query)
    bm25_res = bm25_search(query)
    # 去重
    all_candidates = []
    seen_id = set()
    for item in dense_res + bm25_res:
        if item["chunk_id"] not in seen_id:
            seen_id.add(item["chunk_id"])
            all_candidates.append(item)
    # 重排过滤
    final_ctx = rerank_filter(query, all_candidates)
    # 拼接参考文本
    ctx_text = ""
    for idx, c in enumerate(final_ctx):
        ctx_text += f"【参考{idx+1} 来源：{c['source_file']}】\n{c['text']}\n\n"
    return ctx_text, final_ctx