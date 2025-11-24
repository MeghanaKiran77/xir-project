#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def ensure_corpus_converted(corpus_in: Path, jsonl_out: Path) -> None:
    jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    if jsonl_out.exists():
        return
    if not corpus_in.exists():
        print(f"Missing input: {corpus_in}", file=sys.stderr)
        sys.exit(1)
    written = 0
    with corpus_in.open("r", encoding="utf-8") as fin, jsonl_out.open("w", encoding="utf-8") as fout:
        for line in fin:
            d = json.loads(line)
            doc_id = d.get("doc_id") or d.get("id")
            contents = d.get("text") or d.get("contents")
            if not doc_id or not contents:
                continue
            fout.write(json.dumps({"id": str(doc_id), "contents": contents}, ensure_ascii=False) + "\n")
            written += 1
    print(f"Wrote {written} docs to {jsonl_out}")

def run_index(jsonl_dir: Path, index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", str(jsonl_dir),
        "--index", str(index_dir),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "4",
        "--storePositions", "--storeDocvectors", "--storeRaw",
    ]
    print("Running:", " ".join(cmd))
    env = os.environ.copy()
    subprocess.check_call(cmd, env=env)
    print(f"✅ Built BM25 index at {index_dir}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Build BM25 index from corpus")
    parser.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "data/processed/corpus_en.jsonl",
                       help="Path to input corpus JSONL file")
    parser.add_argument("--index-name", type=str, default=None,
                       help="Index name (e.g., 'msmarco_en'). If not provided, inferred from corpus filename")
    parser.add_argument("--index-dir", type=Path, default=None,
                       help="Output index directory (default: index_sparse/{index_name})")
    args = parser.parse_args()
    
    corpus_in = (args.corpus if args.corpus.is_absolute() else (PROJECT_ROOT / args.corpus)).resolve()
    
    # Infer index name from corpus filename if not provided
    if args.index_name is None:
        # Extract language from filename (e.g., corpus_en.jsonl -> en)
        stem = corpus_in.stem  # e.g., "corpus_en"
        if "_" in stem:
            lang = stem.split("_", 1)[1]  # e.g., "en"
            index_name = f"msmarco_{lang}"
        else:
            index_name = "msmarco_en"
    else:
        index_name = args.index_name
    
    # Set up paths
    jsonl_dir = PROJECT_ROOT / "data/processed/jsonl"
    jsonl_out = jsonl_dir / f"{corpus_in.stem}_id_contents.jsonl"
    index_dir = args.index_dir if args.index_dir else (PROJECT_ROOT / "index_sparse" / index_name)
    
    ensure_corpus_converted(corpus_in, jsonl_out)
    run_index(jsonl_dir, index_dir)

if __name__ == "__main__":
    main()