import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import INTENT_LABELS
import os


'''
#模型下载
cache_dir = "/root/autodl-tmp/models" 
modelscope_model_id = "sentence-transformers/all-MiniLM-L6-v2"
from modelscope import snapshot_download
model_dir = snapshot_download(
        modelscope_model_id,
        cache_dir=cache_dir  # 指定下载文件夹
    )
'''
#model_dir = snapshot_download('sentence-transformers/all-MiniLM-L6-v2')
# 路径配置
CSV_PATH = "./intent_cls/train.csv"
SAVE_MODEL_DIR = "./intent_cls/model_finetune"
MODEL_NAME =  "/root/autodl-tmp/models/all-MiniLM-L6-v2"

# 1. 加载标签映射
label2id = {label: idx for idx, label in enumerate(INTENT_LABELS)}
id2label = {idx: label for idx, label in enumerate(INTENT_LABELS)}

# 2. 加载数据集
df = pd.read_csv(CSV_PATH, encoding="utf-8")
ds = Dataset.from_pandas(df)

# 3. tokenizer初始化
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_func(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

# 4. 标签编码
def encode_label(examples):
    examples["label_id"] = label2id[examples["label"]]
    return examples

ds = ds.map(encode_label)
token_ds = ds.map(tokenize_func, batched=True)
token_ds = token_ds.rename_column("label_id", "labels")
token_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 划分训练验证集
split_ds = token_ds.train_test_split(test_size=0.1)
train_ds = split_ds["train"]
val_ds = split_ds["test"]

# 5. 加载分类模型
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(INTENT_LABELS),
    id2label=id2label,
    label2id=label2id
)

# 6. 训练参数
train_args = TrainingArguments(
    output_dir=SAVE_MODEL_DIR,
    num_train_epochs=6,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir="./intent_cls/logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2,
    no_cuda=not torch.cuda.is_available()
)

# 7. 训练器
trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_ds,
    eval_dataset=val_ds
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model(SAVE_MODEL_DIR)
    tokenizer.save_pretrained(SAVE_MODEL_DIR)
    print(f"意图分类模型训练完成，保存路径：{SAVE_MODEL_DIR}")