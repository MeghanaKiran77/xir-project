#!/usr/bin/env python3
"""
Build a dense (FAISS) index and SQLite docstore from `data/processed/*.jsonl`.

The pipeline mirrors `scripts/build_bm25.py` but uses SentenceTransformer
embeddings so we can support hybrid / multilingual retrieval in later phases.
"""

from __future__ import annotations

import argparse
import os

# Disable multiprocessing to avoid segfaults on macOS
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    import torch
except ImportError:  # pragma: no cover - torch is a dependency of sentence-transformers
    torch = None

# export PYSERINI_JAVA_HOME="$JAVA_HOME"   
# export PATH="$JAVA_HOME/bin:$PATH"  

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = PROJECT_ROOT / "data/processed/corpus_en.jsonl"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "index_dense"
DEFAULT_DOCSTORE_DIR = PROJECT_ROOT / "docstore"
DEFAULT_INDEX_NAME = "msmarco_en"
DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a dense FAISS index + SQLite docstore.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="Path to *.jsonl corpus.")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR, help="Directory for FAISS/index assets.")
    parser.add_argument("--docstore-dir", type=Path, default=DEFAULT_DOCSTORE_DIR, help="Directory for SQLite docstore.")
    parser.add_argument("--index-name", type=str, default=DEFAULT_INDEX_NAME, help="Logical name for output assets.")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="SentenceTransformer model id.")
    parser.add_argument("--batch-size", type=int, default=256, help="Number of passages to encode per batch.")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu / cuda / cuda:0 ...).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing assets if present.")
    parser.add_argument("--keep-memmap", action="store_true", help="Keep intermediate memmap embeddings on disk.")
    return parser.parse_args()


def count_documents(corpus_path: Path) -> int:
    with corpus_path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def ensure_dirs(index_dir: Path, docstore_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    docstore_dir.mkdir(parents=True, exist_ok=True)


def resolve_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch is not None and torch.cuda.is_available():
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
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_doc_id ON documents(doc_id)")
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


def iter_corpus(corpus_path: Path) -> Iterable[dict]:
    with corpus_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def encode_corpus(
    model: SentenceTransformer,
    corpus_path: Path,
    memmap: np.memmap,
    conn: sqlite3.Connection,
    batch_size: int,
) -> int:
    total_docs = memmap.shape[0]
    progress = tqdm(total=total_docs, desc="Encoding corpus", unit="doc")
    processed = 0
    batch_texts: List[str] = []
    batch_rows: List[Tuple[int, str, str, str]] = []

    def flush_batch() -> None:
        nonlocal processed
        if not batch_texts:
            return
        # Encode batch (single-threaded to avoid segfaults on macOS)
        embeddings = model.encode(
            batch_texts,
            batch_size=min(batch_size, len(batch_texts)),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        start = processed
        end = processed + len(batch_texts)
        memmap[start:end] = embeddings
        conn.executemany(
            "INSERT OR REPLACE INTO documents(row_id, doc_id, lang, text) VALUES (?, ?, ?, ?)",
            batch_rows,
        )
        conn.commit()
        progress.update(len(batch_texts))
        processed = end
        batch_texts.clear()
        batch_rows.clear()

    for entry in iter_corpus(corpus_path):
        doc_id = str(entry.get("doc_id"))
        text = entry.get("text", "")
        if not doc_id or not text:
            continue
        lang = entry.get("lang") or ""
        row_id = processed + len(batch_texts)
        batch_texts.append(text)
        batch_rows.append((row_id, doc_id, lang, text))
        if len(batch_texts) >= batch_size:
            flush_batch()

    flush_batch()
    progress.close()
    if processed != total_docs:
        raise RuntimeError(
            f"Expected {total_docs} docs but encoded {processed}. "
            "Did the corpus change between counting and encoding?"
        )
    return processed


def main() -> None:
    args = parse_args()
    corpus_path = (args.corpus if args.corpus.is_absolute() else (PROJECT_ROOT / args.corpus)).resolve()
    index_dir = (args.index_dir if args.index_dir.is_absolute() else (PROJECT_ROOT / args.index_dir)).resolve()
    docstore_dir = (args.docstore_dir if args.docstore_dir.is_absolute() else (PROJECT_ROOT / args.docstore_dir)).resolve()

    ensure_dirs(index_dir, docstore_dir)

    index_prefix = args.index_name
    faiss_path = index_dir / f"{index_prefix}.faiss"
    meta_path = index_dir / f"{index_prefix}.meta.json"
    memmap_path = index_dir / f"{index_prefix}.fp32.memmap"
    docstore_path = docstore_dir / f"{index_prefix}.sqlite"

    if not args.overwrite:
        for path in (faiss_path, meta_path, docstore_path, memmap_path):
            if path.exists():
                raise FileExistsError(f"{path} already exists. Use --overwrite to rebuild.")
    else:
        for path in (faiss_path, meta_path, memmap_path):
            if path.exists():
                path.unlink()

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    print(f"[build_dense] Counting documents in {corpus_path} ...")
    doc_count = count_documents(corpus_path)
    if doc_count == 0:
        raise RuntimeError("Corpus is empty; nothing to encode.")
    print(f"[build_dense] Found {doc_count:,} documents.")

    device = resolve_device(args.device)
    print(f"[build_dense] Loading encoder '{args.model_name}' on {device}.")
    model = SentenceTransformer(args.model_name, device=device)
    embedding_dim = model.get_sentence_embedding_dimension()

    print(f"[build_dense] Allocating memmap at {memmap_path} ({doc_count} x {embedding_dim}).")
    memmap = np.memmap(
        memmap_path,
        mode="w+",
        dtype="float32",
        shape=(doc_count, embedding_dim),
    )

    conn = prepare_sqlite(docstore_path, overwrite=args.overwrite)

    print("[build_dense] Encoding corpus ...")
    encode_corpus(model, corpus_path, memmap, conn, batch_size=args.batch_size)
    memmap.flush()

    print(f"[build_dense] Building FAISS IndexFlatIP at {faiss_path}.")
    embeddings = np.memmap(
        memmap_path,
        mode="r",
        dtype="float32",
        shape=(doc_count, embedding_dim),
    )
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)
    faiss.write_index(index, str(faiss_path))
    del embeddings

    if not args.keep_memmap and memmap_path.exists():
        memmap_path.unlink()

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

    conn.execute("DELETE FROM metadata")
    conn.executemany(
        "INSERT INTO metadata(key, value) VALUES (?, ?)",
        [(k, json.dumps(v) if isinstance(v, (dict, list)) else str(v)) for k, v in metadata.items()],
    )
    conn.commit()
    conn.close()

    print(f"[build_dense] Finished. FAISS index: {faiss_path}, docstore: {docstore_path}, meta: {meta_path}")


if __name__ == "__main__":
    main()

