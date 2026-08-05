import pdfplumber
import os
import json

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import RAW_DATA_DIR
def parse_single_pdf(pdf_path):
    all_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text.append(text)
    full_text = "\n".join(all_text)
    return {
        "source": os.path.basename(pdf_path),
        "text": full_text
    }

def batch_parse_all_pdf():
    res = []
    for file in os.listdir(RAW_DATA_DIR):
        if file.endswith(".pdf"):
            p = os.path.join(RAW_DATA_DIR, file)
            data = parse_single_pdf(p)
            res.append(data)
    return res

if __name__ == "__main__":
    datas = batch_parse_all_pdf()
    with open("data/raw/all_raw_text.json", "w", encoding="utf-8") as f:
        json.dump(datas, f, ensure_ascii=False, indent=2)
    print("PDF解析完成，保存至 data/raw/all_raw_text.json")