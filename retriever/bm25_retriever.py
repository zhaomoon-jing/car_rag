import jsonlines
from rank_bm25 import BM25Okapi
from config import CHUNK_DATA_PATH, TOP_K_BM25

# 全局初始化
corpus = []
meta_list = []
with jsonlines.open(CHUNK_DATA_PATH, "r") as r:
    for obj in r:
        txt = obj["text"]
        corpus.append(txt.split())
        meta_list.append(obj)
bm25 = BM25Okapi(corpus)

def bm25_search(query):
    tokenized_q = query.split()
    scores = bm25.get_scores(tokenized_q)
    top_idx = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)[:TOP_K_BM25]
    return [meta_list[i] for i in top_idx]