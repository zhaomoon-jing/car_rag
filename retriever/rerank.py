'''
import sys
from pathlib import Path
# 获取当前脚本所在目录的上级 = 项目根目录
#sys.path.append(str(Path(__file__).parent.parent))

root_path = str(Path(__file__).parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from FlagEmbedding import FlagReranker
from retriever.dense_retriever import dense_search
from retriever.bm25_retriever import bm25_search


from config import RERANK_MODEL_NAME, TOP_K_RERANK

cache_dir = "/root/autodl-tmp/models" 
modelscope_model_id = RERANK_MODEL_NAME

#本地路径格式化
name_lst = RERANK_MODEL_NAME.split("/")
model_format_name= name_lst[0] +"--"+ name_lst[1]
#local_dir = Path("/root/autodl-tmp/models/models"+ model_format_name)

CACHE_ROOT = Path("/root/autodl-tmp/models")
MODEL_CACHE_FOLDER = CACHE_ROOT / model_format_name




# 判断本地是否存在完整模型（核心：检测config.json）
config_paths = list(MODEL_CACHE_FOLDER.rglob("config.json"))

if MODEL_CACHE_FOLDER.exists() and config_paths[0].exists():
    real_model_path = list((MODEL_CACHE_FOLDER / "snapshots").iterdir())[0]
    print(f"✅ 检测到本地模型，离线加载：{real_model_path}")
    LOCAL_RERANK_MODEL_NAME = RERANK_MODEL_NAME
    reranker = FlagReranker(real_model_path, use_fp16=True)
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


'''


import sys
from pathlib import Path
# 获取当前脚本所在目录的上级 = 项目根目录
#sys.path.append(str(Path(__file__).parent.parent))
root_path = str(Path(__file__).parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)
from FlagEmbedding import FlagReranker
#from retriever.dense_retriever import dense_search
#from retriever.bm25_retriever import bm25_search
from config import RERANK_MODEL_NAME, TOP_K_RERANK

# ===================== 新增配置与RRF工具 =====================
# RRF超参
RRF_K = 60
# RRF粗筛截断数量，送入重排前最多保留这么多条
TOP_K_RRF = 20

def reciprocal_rank_fusion(vec_list: list[dict], bm25_list: list[dict], rrf_k: int = 60, top_rrf: int = 40) -> list[dict]:
    """
    RRF融合两路召回结果，使用chunk_id作为唯一标识
    :param vec_list: 向量召回结果（已按相似度降序）
    :param bm25_list: BM25召回结果（已按分数降序）
    :param rrf_k: RRF公式平滑系数
    :param top_rrf: 融合后截断保留条数
    :return: RRF排序后的候选列表（去重）
    """
    # 构建 chunk_id -> rank 映射，rank从1开始
    vec_rank_map = {}
    for rank, item in enumerate(vec_list, start=1):
        cid = item["chunk_id"]
        vec_rank_map[cid] = rank

    bm25_rank_map = {}
    for rank, item in enumerate(bm25_list, start=1):
        cid = item["chunk_id"]
        bm25_rank_map[cid] = rank

    # 合并所有chunk，chunk_id去重
    all_chunk_dict = {}
    for item in vec_list + bm25_list:
        cid = item["chunk_id"]
        if cid not in all_chunk_dict:
            all_chunk_dict[cid] = item

    # 计算每条文档RRF总分
    rrf_result = []
    for cid, chunk_item in all_chunk_dict.items():
        score = 0.0
        if cid in vec_rank_map:
            score += 1.0 / (rrf_k + vec_rank_map[cid])
        if cid in bm25_rank_map:
            score += 1.0 / (rrf_k + bm25_rank_map[cid])
        rrf_result.append((score, chunk_item))

    # RRF分数降序排序，截断
    rrf_result.sort(key=lambda x: x[0], reverse=True)
    top_fused = [item for (s, item) in rrf_result[:top_rrf]]
    return top_fused
# ==========================================================

cache_dir = "/root/autodl-tmp/models" 
modelscope_model_id = RERANK_MODEL_NAME
#本地路径格式化
name_lst = RERANK_MODEL_NAME.split("/")
model_format_name= name_lst[0] +"--"+ name_lst[1]
#local_dir = Path("/root/autodl-tmp/models/models"+ model_format_name)

CACHE_ROOT = Path("/root/autodl-tmp/models")
MODEL_CACHE_FOLDER = CACHE_ROOT / model_format_name


# 判断本地是否存在完整模型（核心：检测config.json）
config_paths = list(MODEL_CACHE_FOLDER.rglob("config.json"))
if MODEL_CACHE_FOLDER.exists() and config_paths[0].exists():
    real_model_path = list((MODEL_CACHE_FOLDER / "snapshots").iterdir())[0]
    print(f"✅ 检测到本地模型，离线加载：{real_model_path}")
    LOCAL_RERANK_MODEL_NAME = RERANK_MODEL_NAME
    reranker = FlagReranker(real_model_path, use_fp16=True)
else:
    from modelscope import snapshot_download
    model_dir = snapshot_download(RERANK_MODEL_NAME,cache_dir=cache_dir)
    reranker = FlagReranker(model_dir, use_fp16=True)

def rerank_filter(query, candidate_chunks):
    pairs = [(query, item["text"]) for item in candidate_chunks]
    scores = reranker.compute_score(pairs, normalize=True)
    # 绑定分数+原文，排序取top
    scored = list(zip(scores, candidate_chunks))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [x[1] for x in scored[:TOP_K_RERANK]]
    return top

def hybrid_retrieve(query):
    from retriever.dense_retriever import dense_search
    from retriever.bm25_retriever import bm25_search
    # 混合检索
    dense_res = dense_search(query)
    bm25_res = bm25_search(query)

    # ========== 改造：替换原有直接合并去重，改用RRF融合粗筛 ==========
    all_candidates = reciprocal_rank_fusion(
        vec_list=dense_res,
        bm25_list=bm25_res,
        rrf_k=RRF_K,
        top_rrf=TOP_K_RRF
    )
    # ==============================================================

    # 重排过滤（逻辑完全不变）
    final_ctx = rerank_filter(query, all_candidates)
    # 拼接参考文本（输出格式完全不变，下游无感知）
    ctx_text = ""
    for idx, c in enumerate(final_ctx):
        ctx_text += f"【参考{idx+1} 来源：{c['source_file']}】\n{c['text']}\n\n"
    return ctx_text, final_ctx


if __name__ == "__main__":
    query = "空调怎么开"
    ctx_text, final_ctx = hybrid_retrieve(query)
    print("=== 检索结果 ===")
    print(ctx_text)