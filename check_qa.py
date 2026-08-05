import json

with open("data/qa_train/train.jsonl", "r", encoding="utf-8") as f:
    data = json.load(f)

# 校验格式
err_count = 0
for idx, item in enumerate(data):
    if "question" not in item or "answer" not in item:
        print(f"第{idx}条字段缺失：{item}")
        err_count += 1
    if len(item["question"]) < 2 or len(item["answer"]) < 5:
        print(f"第{idx}条内容过短：{item}")
        err_count += 1

print(f"总问答条数：{len(data)}，错误条数：{err_count}")