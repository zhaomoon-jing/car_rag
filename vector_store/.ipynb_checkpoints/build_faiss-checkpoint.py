import faiss
import jsonlines
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import EMBED_MODEL_NAME, CHUNK_DATA_PATH, FAISS_SAVE_DIR


def build_faiss_vector_db():
    # 加载嵌入模型
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    print("加载模型成功")
    # 读取分块
    chunks = []
    meta_list = []
    with jsonlines.open(CHUNK_DATA_PATH, "r") as reader:
        for obj in reader:
            chunks.append(obj["text"])
            meta_list.append(obj)
    # 批量向量化
    print("开始向量化文档片段...")
    embeddings = embed_model.encode(chunks, batch_size=32, convert_to_numpy=True)
    dim = embeddings.shape[1]
    # 构建FAISS
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    # 持久化
    os.makedirs(FAISS_SAVE_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(FAISS_SAVE_DIR, "car_index.faiss"))
    # 保存元数据
    with open(os.path.join(FAISS_SAVE_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)
    print("FAISS向量库构建完成")

def load_faiss():
    index = faiss.read_index(os.path.join(FAISS_SAVE_DIR, "car_index.faiss"))
    with open(os.path.join(FAISS_SAVE_DIR, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return index, meta, embed_model

if __name__ == "__main__":
    build_faiss_vector_db()