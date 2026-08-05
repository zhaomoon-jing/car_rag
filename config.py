import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data/raw")
CHUNK_DATA_PATH = os.path.join(BASE_DIR, "data/chunks/chunks.jsonl")
QA_TRAIN_PATH = os.path.join(BASE_DIR, "data/qa_train/train.jsonl")
FAISS_SAVE_DIR = os.path.join(BASE_DIR, "vector_store/faiss_index")

# 分块参数
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# 向量模型
EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

#意图分类
#INTENT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INTENT_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# 检索参数
TOP_K_DENSE = 4
TOP_K_BM25 = 4
TOP_K_RERANK = 2

# LLM配置（轻量化Qwen0.5B）
#LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LOAD_4BIT = False
LORA_WEIGHT_DIR = os.path.join(BASE_DIR, "llm_infer/lora_car")

# ASR
ASR_MODEL_SIZE = "tiny"

# 意图分类标签
INTENT_LABELS = [
    "control",    # 座舱控制：空调、车窗、座椅
    "qa_know",    # 车辆知识问答（走RAG）
    "nav",        # 导航
    "chat_none"   # 无关闲聊
]