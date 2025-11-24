from datasets import load_dataset
import json, os
from tqdm import tqdm

OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

dataset = load_dataset("Tevatron/msmarco-passage", split="train[:1%]")

out_path = os.path.join(OUT_DIR, "corpus_en.jsonl")
with open(out_path, "w", encoding="utf-8") as f:
    for i, item in enumerate(tqdm(dataset)):
        doc_id = str(item.get("pid") or item.get("passage_id") or i)
        text = item.get("passage") or item.get("text")
        if text is None:
            positives = item.get("positive_passages")
            if positives and isinstance(positives, list) and len(positives) > 0:
                text = positives[0].get("text")
        if not text:
            continue
        f.write(json.dumps({
            "doc_id": doc_id,
            "lang": "en",
            "text": text
        }) + "\n")

print(f"✅ Saved subset corpus to {out_path}")