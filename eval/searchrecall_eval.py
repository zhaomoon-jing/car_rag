def recall_at_k(ground_truth_doc:str, retrieved_contexts:list, k:int):
    """
    ground_truth_doc：包含正确答案的那份参考文档
    retrieved_contexts：检索返回全部文档列表
    k：取前k个
    return 1：前k条存在正确文档；0：不存在
    """
    topk = retrieved_contexts[:k]
    # 判断：正确文档是否在topk检索结果中
    for doc in topk:
        if ground_truth_doc in doc:
            return 1
    return 0

# 批量计算整体Recall@k
def calc_total_recall(samples, k=3):
    hit = 0
    total = len(samples)
    for s in samples:
        hit += recall_at_k(s["correct_doc"], s["retrieved_contexts"], k)
    return hit / total

# 示例调用
sample = {
    "question": "车机卡死该怎么办？",
    "correct_doc": "长按OK按键可以重启车机",
    "retrieved_contexts": ["长按OK按键可以重启车机","空调调节范围16‑30摄氏度"]
}
print(calc_total_recall([sample],k=3))