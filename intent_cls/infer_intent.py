'''
import torch
from preload_all import intent_tokenizer,intent_model
from config import INTENT_LABELS

def get_intent(text:str)->str:
    inputs = intent_tokenizer(
        text,return_tensors="pt",truncation=True,padding=True,max_length=128
    )
    with torch.no_grad():
        out = intent_model(**inputs)
    pred_idx = torch.argmax(out.logits,dim=-1).item()
    return INTENT_LABELS[pred_idx]
'''

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import INTENT_LABELS


CACHE_ROOT = Path("/root/autodl-tmp/models")
MODEL_CACHE_FOLDER = CACHE_ROOT / "models--BAAI--bge-small-zh-v1.5"
real_model_path = list((MODEL_CACHE_FOLDER / "snapshots").iterdir())[0]

tokenizer = AutoTokenizer.from_pretrained(real_model_path)
cls_model = AutoModelForSequenceClassification.from_pretrained("./intent_cls/model_finetune")
cls_model.eval()

def get_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        out = cls_model(**inputs)
    pred_idx = torch.argmax(out.logits, dim=-1).item()
    label = INTENT_LABELS[pred_idx]
    return label