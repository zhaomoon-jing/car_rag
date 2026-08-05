from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import INTENT_LABELS

model_name = "/root/autodl-tmp/models/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
cls_model = AutoModelForSequenceClassification.from_pretrained("./intent_cls/model_finetune")
cls_model.eval()

def get_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        out = cls_model(**inputs)
    pred_idx = torch.argmax(out.logits, dim=-1).item()
    label = INTENT_LABELS[pred_idx]
    return label