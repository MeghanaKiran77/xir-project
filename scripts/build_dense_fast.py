#!/usr/bin/env python3
"""
Fast dense index builder for CLIR:

- No memmap
- No huge temporary arrays
- Streams docs in batches
- Uses FAISS IndexHNSWFlat (ANN)
- Writes SQLite docstore + meta.json compatible with DenseSearcher
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Make things stable on macOS
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = PROJECT_ROOT / "data/processed/corpus_en_filtered.jsonl"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "index_dense"
DEFAULT_DOCSTORE_DIR = PROJECT_ROOT / "docstore"
DEFAULT_INDEX_NAME = "msmarco_en"
DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast HNSW-based dense index builder.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help="Path to filtered *.jsonl corpus (en/hi/sv).",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Directory for FAISS index + meta.",
    )
    parser.add_argument(
        "--docstore-dir",
        type=Path,
        default=DEFAULT_DOCSTORE_DIR,
        help="Directory for SQLite docstore.",
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default=DEFAULT_INDEX_NAME,
        help="Logical name for output assets (e.g., msmarco_en).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="SentenceTransformer model id.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of passages to encode per batch.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cpu / cuda / cuda:0 ...). Defaults to auto.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing index + docstore + meta if present.",
    )
    return parser.parse_args()


def resolve_device(requested: str | None) -> str:
    try:
        import torch  # type: ignore
    except ImportError:
        return "cpu"
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def prepare_sqlite(db_path: Path, overwrite: bool) -> sqlite3.Connection:
    if overwrite and db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            row_id INTEGER PRIMARY KEY,
            doc_id TEXT UNIQUE,
            lang TEXT,
            text TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    conn.commit()
    return conn


def iter_corpus(corpus_path: Path) -> Iterable[Dict[str, Any]]:
    with corpus_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield obj


def truncate_text(text: str, max_tokens: int = 128) -> str:
    # Simple whitespace tokenization; good enough for our purposes
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens])


def main() -> None:
    args = parse_args()

    corpus_path = (
        args.corpus if args.corpus.is_absolute() else (PROJECT_ROOT / args.corpus)
    ).resolve()
    index_dir = (
        args.index_dir if args.index_dir.is_absolute() else (PROJECT_ROOT / args.index_dir)
    ).resolve()
    docstore_dir = (
        args.docstore_dir if args.docstore_dir.is_absolute() else (PROJECT_ROOT / args.docstore_dir)
    ).resolve()

    index_dir.mkdir(parents=True, exist_ok=True)
    docstore_dir.mkdir(parents=True, exist_ok=True)

    index_prefix = args.index_name
    faiss_path = index_dir / f"{index_prefix}.faiss"
    meta_path = index_dir / f"{index_prefix}.meta.json"
    docstore_path = docstore_dir / f"{index_prefix}.sqlite"

    if not args.overwrite:
        for path in (faiss_path, meta_path, docstore_path):
            if path.exists():
                raise FileExistsError(
                    f"{path} already exists. Use --overwrite to rebuild."
                )
    else:
        for path in (faiss_path, meta_path):
            if path.exists():
                path.unlink()
        # docstore will be recreated by prepare_sqlite

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    device = resolve_device(args.device)
    print(f"[fast_dense] Loading encoder '{args.model_name}' on {device}.")
    model = SentenceTransformer(args.model_name, device=device)
    embedding_dim = model.get_sentence_embedding_dimension()

    print(f"[fast_dense] Preparing SQLite docstore at {docstore_path}.")
    conn = prepare_sqlite(docstore_path, overwrite=args.overwrite)

    # Build HNSW index (approximate, but standard and fast)
    print(f"[fast_dense] Creating HNSW index (dim={embedding_dim}).")
    hnsw_m = 32  # graph degree
    index = faiss.IndexHNSWFlat(embedding_dim, hnsw_m)
    index.hnsw.efConstruction = 128  # good quality / speed tradeoff

    row_id = 0
    batch_texts: List[str] = []
    batch_rows: List[Tuple[int, str, str, str]] = []

    print(f"[fast_dense] Reading corpus from {corpus_path} ...")
    progress = tqdm(desc="Encoding corpus", unit="doc")

    def flush_batch() -> None:
        nonlocal row_id
        if not batch_texts:
            return
        emb = model.encode(
            batch_texts,
            batch_size=min(args.batch_size, len(batch_texts)),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        index.add(emb)
        conn.executemany(
            "INSERT OR REPLACE INTO documents(row_id, doc_id, lang, text) VALUES (?, ?, ?, ?)",
            batch_rows,
        )
        conn.commit()
        progress.update(len(batch_texts))
        batch_texts.clear()
        batch_rows.clear()

    for obj in iter_corpus(corpus_path):
        doc_id = str(obj.get("doc_id"))
        text = obj.get("text", "")
        if not doc_id or not text:
            continue
        lang = obj.get("lang", "")

        truncated = truncate_text(text, max_tokens=128)

        batch_texts.append(truncated)
        batch_rows.append((row_id, doc_id, lang, text))
        row_id += 1

        if len(batch_texts) >= args.batch_size:
            flush_batch()

    flush_batch()
    progress.close()

    doc_count = row_id
    print(f"[fast_dense] Finished encoding {doc_count:,} documents.")

    print(f"[fast_dense] Writing FAISS index to {faiss_path}.")
    faiss.write_index(index, str(faiss_path))

    # Write meta JSON compatible with DenseSearcher
    built_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "model_name": args.model_name,
        "dimension": embedding_dim,
        "doc_count": doc_count,
        "built_at": built_at,
        "corpus_path": str(corpus_path),
        "faiss_index_path": str(faiss_path),
        "docstore_path": str(docstore_path),
        "normalize_embeddings": True,
    }

    with meta_path.open("w", encoding="utf-8") as mf:
        json.dump(metadata, mf, indent=2)
        mf.write("\n")

    # Also store metadata in SQLite for completeness
    conn.execute("DELETE FROM metadata")
    conn.executemany(
        "INSERT INTO metadata(key, value) VALUES (?, ?)",
        [
            (
                k,
                json.dumps(v) if isinstance(v, (dict, list)) else str(v),
            )
            for k, v in metadata.items()
        ],
    )
    conn.commit()
    conn.close()

    print(
        f"[fast_dense] Done. FAISS index: {faiss_path}, docstore: {docstore_path}, meta: {meta_path}"
    )


if __name__ == "__main__":
    main()