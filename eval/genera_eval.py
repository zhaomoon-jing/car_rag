from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 加载本地评判模型，使用Qwen2.5‑1.8B‑Instruct作为judge
model_name = "Qwen/Qwen2.5‑1.8B‑Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype="bfloat16"
)

pipe = pipeline(
    "text‑generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1
)
langchain_llm = HuggingFacePipeline(pipeline=pipe)
ragas_llm = LangchainLLMWrapper(langchain_llm)

# 设置RAGAS使用本地模型做评判
faithfulness.llm = ragas_llm
answer_relevancy.llm = ragas_llm

# 构造测试数据集
eval_data = [
    {
        "question": "车机卡死该怎么办？",
        "answer": "车机卡死可以长按OK按键重启。",
        "contexts": [
            "长按OK按键可以重启车机",
            "空调调节范围16‑30摄氏度"
        ],
        "ground_truth": "长按OK按键可以重启车机"
    }
]

ds = Dataset.from_list(eval_data)

# 执行评测
result = evaluate(
    ds,
    metrics=[faithfulness, answer_relevancy]
)

print(result)
print(f"忠实度 faithfulness: {result['faithfulness']:.3f}")
print(f"回答相关性 answer_relevancy: {result['answer_relevancy']:.3f}")