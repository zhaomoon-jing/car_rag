import json
import jsonlines

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_DATA_PATH

def split_text(text):
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def build_chunks():
    with open("data/raw/all_raw_text.json", "r", encoding="utf-8") as f:
        raw_datas = json.load(f)
    out = []
    for item in raw_datas:
        source = item["source"]
        full_text = item["text"]
        chunk_list = split_text(full_text)
        for idx, ck in enumerate(chunk_list):
            if len(ck) < 20:
                continue
            out.append({
                "chunk_id": f"{source}_{idx}",
                "source_file": source,
                "text": ck
            })
    with jsonlines.open(CHUNK_DATA_PATH, "w") as writer:
        writer.write_all(out)
    print(f"分块完成，共 {len(out)} 条片段，保存 {CHUNK_DATA_PATH}")

if __name__ == "__main__":
    build_chunks()