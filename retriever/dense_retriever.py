from vector_store.build_faiss import load_faiss
from config import TOP_K_DENSE

index, meta_data, embed_model = load_faiss()

def dense_search(query):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    distance, idx_arr = index.search(q_emb, TOP_K_DENSE)
    res = []
    for i in idx_arr[0]:
        res.append(meta_data[i])
    return res
