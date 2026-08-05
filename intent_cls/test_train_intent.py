from transformers import AutoTokenizer, AutoModelForSequenceClassification
save_path = "./intent_cls/model_finetune"
tokenizer = AutoTokenizer.from_pretrained(save_path)
model = AutoModelForSequenceClassification.from_pretrained(save_path)
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import INTENT_LABELS

def predict_intent(text):
    inputs = tokenizer(text, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    pred_idx = out.logits.argmax().item()
    score = torch.softmax(out.logits, dim=-1)[0][pred_idx].item()
    return INTENT_LABELS[pred_idx], round(score, 3)

print(predict_intent("帮我打开副驾座椅加热"))