from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import torch
except ImportError:  # pragma: no cover - torch is a dependency of sentence-transformers
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX = PROJECT_ROOT / "index_dense/msmarco_en.faiss"
DEFAULT_DOCSTORE = PROJECT_ROOT / "docstore/msmarco_en.sqlite"
DEFAULT_META = PROJECT_ROOT / "index_dense/msmarco_en.meta.json"


class DenseSearcher:
    """
    Thin wrapper around a FAISS index + SQLite docstore for dense retrieval.

    The builder (`scripts/build_dense.py`) stores row-aligned embeddings and metadata,
    so we can recover doc_id / language / raw text after FAISS search without opening
    the original JSONL corpus.
    """

    def __init__(
        self,
        index_path: Optional[str | Path] = None,
        docstore_path: Optional[str | Path] = None,
        meta_path: Optional[str | Path] = DEFAULT_META,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        normalize: bool = True,
    ) -> None:
        meta = self._load_meta(meta_path)
        self.index_path = Path(index_path or meta.get("faiss_index_path") or DEFAULT_INDEX).resolve()
        self.docstore_path = Path(docstore_path or meta.get("docstore_path") or DEFAULT_DOCSTORE).resolve()
        self.model_name = model_name or meta.get("model_name")
        if self.model_name is None:
            raise ValueError("model_name must be specified explicitly or via the meta file.")

        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not self.docstore_path.exists():
            raise FileNotFoundError(f"Docstore not found: {self.docstore_path}")

        self.index = faiss.read_index(str(self.index_path))
        self.dimension = self.index.d
        self.normalize = normalize
        self.device = device or self._default_device()
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.conn = sqlite3.connect(str(self.docstore_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def search(self, query_text: str, k: int = 10, return_text: bool = False) -> List[Dict[str, Any]]:
        """
        Encode a query, run FAISS search, and hydrate doc metadata from SQLite.
        """
        if not query_text:
            return []
        query_vec = self._encode_query(query_text)
        scores, indices = self.index.search(query_vec[None, :], k)
        hits: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = self._fetch_document(int(idx))
            if not doc:
                continue
            hit = {
                "doc_id": doc["doc_id"],
                "dense_score": float(score),
                "lang": doc.get("lang", ""),
            }
            if return_text:
                hit["text"] = doc.get("text", "")
            hits.append(hit)
        return hits

    def close(self) -> None:
        self.conn.close()

    def _encode_query(self, query_text: str) -> np.ndarray:
        vector = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )[0].astype("float32")
        if self.normalize:
            faiss.normalize_L2(vector.reshape(1, -1))
        return vector

    def _fetch_document(self, row_id: int) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT doc_id, lang, text FROM documents WHERE row_id = ?",
            (row_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    @staticmethod
    def _default_device() -> str:
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @staticmethod
    def _load_meta(meta_path: Optional[str | Path]) -> Dict[str, Any]:
        if not meta_path:
            return {}
        meta_file = Path(meta_path)
        if not meta_file.exists():
            return {}
        with meta_file.open("r", encoding="utf-8") as mf:
            try:
                data = json.load(mf)
            except json.JSONDecodeError:
                return {}
        return data

