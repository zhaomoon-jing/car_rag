import pandas as pd
import torch
import os
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import evaluate

# ====================== 配置区（统一管理，参考文件1规范） ======================
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import INTENT_LABELS
# 4分类意图标签
#INTENT_LABELS = [
#   "control",    # 座舱控制：空调、车窗、座椅
#    "qa_know",    # 车辆知识问答（走RAG）
#    "nav",        # 导航
#    "chat_none"   # 闲聊无关
#]


num_labels = len(INTENT_LABELS)
label2id = {label: idx for idx, label in enumerate(INTENT_LABELS)}
id2label = {idx: label for idx, label in enumerate(INTENT_LABELS)}

# 模型本地路径（解决snapshots嵌套找不到config.json问题）
CACHE_ROOT = Path("/root/autodl-tmp/models")
MODEL_CACHE_FOLDER = CACHE_ROOT / "models--BAAI--bge-small-zh-v1.5"
#MODEL_CACHE_FOLDER = CACHE_ROOT / "models--sentence-transformers--all-MiniLM-L6-v2"
# 自动获取快照内真实模型目录
real_model_path = list((MODEL_CACHE_FOLDER / "snapshots").iterdir())[0]

if (MODEL_CACHE_FOLDER / "snapshots").exists():
    print(f"模型已存在于 {MODEL_CACHE_FOLDER}，无需重新下载。")
else:
    from modelscope import snapshot_download
    model_dir = snapshot_download(
        INTENT_MODEL_NAME,
        cache_dir=cache_dir  # 指定下载文件夹
    )


# 数据&输出路径
CSV_PATH = Path("./intent_cls/train.csv")
SAVE_MODEL_DIR = Path("./intent_cls/model_finetune")
SAVE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 训练超参（对标文件1优质配置）
MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 6
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
# ==============================================================================

def load_raw_dataset(csv_path: Path):
    """加载csv数据集，参考文件1增加异常捕获"""
    if not csv_path.exists():
        raise FileNotFoundError(f"训练数据集不存在：{csv_path}，请补充标注数据！")
    df = pd.read_csv(csv_path, encoding="utf-8")
    # 校验标签合法性
    unknown_labels = set(df["label"].unique()) - set(INTENT_LABELS)
    if unknown_labels:
        raise ValueError(f"数据集存在未定义意图标签：{unknown_labels}")
    ds = Dataset.from_pandas(df)
    print(f"原始数据集加载完成，总样本量：{len(ds)}")
    # 统计每类样本数量，打印分布，方便排查数据不均衡
    print("===== 各类样本分布 =====")
    for lab in INTENT_LABELS:
        cnt = len(df[df["label"] == lab])
        print(f"{lab}: {cnt} 条")
    return ds

def preprocess_dataset(dataset, tokenizer):
    def tokenize_func(examples):
        # 删掉 return_tensors="pt"！！
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=MAX_LEN
        )

    def encode_label(examples):
        examples["labels"] = [label2id[l] for l in examples["label"]]
        return examples

    # 1. 先转数字标签
    ds = dataset.map(encode_label, batched=True)
    # 2. 分词
    token_ds = ds.map(tokenize_func, batched=True)
    # 3. 关键：移除原始字符串列 text / label，只保留模型需要的数字字段
    token_ds = token_ds.remove_columns(["text", "label"])
    return token_ds

def compute_metrics(eval_pred):
    """评估指标：准确率（参考文件1，以分类精度监控训练）"""
    metric = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = torch.tensor(predictions).argmax(dim=-1).numpy()
    return metric.compute(predictions=predictions, references=labels)

def main():
    print("=" * 60)
    print("4分类车辆意图分类训练工具（优化版）")
    print("=" * 60)
    print(f"分类类别：{INTENT_LABELS}")
    print(f"预训练模型本地路径：{real_model_path}")

    # 1. 加载分词器（强制本地离线，杜绝联网401/权重缺失）
    tokenizer = AutoTokenizer.from_pretrained(
        str(real_model_path),
        local_files_only=True
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 2. 加载并预处理数据
    raw_ds = load_raw_dataset(CSV_PATH)
    token_ds = preprocess_dataset(raw_ds, tokenizer)

    # 划分训练/验证集 9:1
    split_ds = token_ds.train_test_split(test_size=0.1, seed=42)
    train_ds = split_ds["train"]
    val_ds = split_ds["test"]
    print(f"训练集：{len(train_ds)} 条，验证集：{len(val_ds)} 条")

    # 3. 加载分类模型
    model = AutoModelForSequenceClassification.from_pretrained(
        str(real_model_path),
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        local_files_only=True
    )

    # 4. 训练参数（优化后）
    training_args = TrainingArguments(
        output_dir=str(SAVE_MODEL_DIR),
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_dir=str(SAVE_MODEL_DIR / "logs"),
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",  # 核心：按准确率保存最优模型
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        no_cuda=not torch.cuda.is_available()
    )

    # 5. 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # 开始训练
    print("\n🚀 开始训练...")
    trainer.train()

    # 验证集最终评估
    print("\n📊 最优模型验证集评估结果：")
    eval_res = trainer.evaluate()
    for k, v in eval_res.items():
        print(f"{k}: {v:.4f}")

    # 保存模型&分词器
    trainer.save_model(SAVE_MODEL_DIR)
    tokenizer.save_pretrained(SAVE_MODEL_DIR)
    print(f"\n✅ 训练完成，模型保存至：{SAVE_MODEL_DIR}")

if __name__ == "__main__":
    main()