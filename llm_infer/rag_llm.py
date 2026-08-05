from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import PeftModel

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import LLM_MODEL_NAME, LOAD_4BIT, LORA_WEIGHT_DIR

# 4bit量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
) if LOAD_4BIT else None

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
# 加载LoRA微调权重（无则注释）
try:
    llm_model = PeftModel.from_pretrained(base_model, LORA_WEIGHT_DIR)
except:
    llm_model = base_model

# 车载RAG固定Prompt
SYSTEM_PROMPT = """
你是车载座舱专属语音问答助手，严格遵守规则：
1. 只能使用下方【车辆参考资料】回答，资料无对应内容必须回复：“暂无相关车辆资料，请查阅车辆用户手册”；
2. 严禁编造故障、功能、参数，禁止幻觉；
3. 回答简短口语化，适合车载播报，不超过3句话；
4. 不输出无关内容、不拓展闲聊。
"""

def generate_rag_answer(query, context):
    prompt = f"""<|im_start|>system
{SYSTEM_PROMPT}
【车辆参考资料】
{context}
<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.1,
        top_p=0.3,
        repetition_penalty=1.05
    )
    ans = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return ans