from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from retrieval.bm25_search import BM25Searcher
from retrieval.dense_search import DenseSearcher
from retrieval.fusion import fuse_scores

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class MultilingualRetriever:
    """
    Hybrid retriever combining BM25 (sparse) and Dense (semantic) search.

    Supports cross-lingual retrieval: queries in any language can search documents
    in any target language (once corresponding indexes are built).

    Dense retrieval uses a multilingual encoder, so it fully supports cross-lingual.
    BM25 uses a language-specific Lucene index per target language.
    """

    def __init__(
        self,
        supported_languages: Optional[List[str]] = None,
        index_base_dir: str | Path = "index_sparse",
        dense_index_base_dir: str | Path = "index_dense",
        docstore_base_dir: str | Path = "docstore",
        fusion_alpha: float = 1.0,  # we tuned α≈1.0 for CLIRMatrix
    ):
        """
        Initialize MultilingualRetriever with support for multiple languages.

        Args:
            supported_languages: List of language codes to support (e.g., ['en', 'hi', 'sv']).
                                 If None, auto-detects from available indexes.
            index_base_dir: Base directory for BM25 indexes (default: "index_sparse")
            dense_index_base_dir: Base directory for dense indexes (default: "index_dense")
            docstore_base_dir: Base directory for docstores (default: "docstore")
            fusion_alpha: Default weight for BM25 in fusion (0-1). 1.0 = BM25-only,
                          0.0 = Dense-only, 0.5 = equal weight.
        """
        self.index_base_dir = Path(index_base_dir) if isinstance(index_base_dir, str) else index_base_dir
        self.dense_index_base_dir = (
            Path(dense_index_base_dir) if isinstance(dense_index_base_dir, str) else dense_index_base_dir
        )
        self.docstore_base_dir = Path(docstore_base_dir) if isinstance(docstore_base_dir, str) else docstore_base_dir
        self.fusion_alpha = fusion_alpha

        # Auto-detect supported languages from available indexes
        if supported_languages is None:
            supported_languages = self._detect_supported_languages()

        self.supported_languages = supported_languages
        self.bm25_searchers: Dict[str, BM25Searcher] = {}
        self.dense_searchers: Dict[str, DenseSearcher] = {}

        # Initialize searchers for each supported language
        for lang in supported_languages:
            self._load_searchers_for_language(lang)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _detect_supported_languages(self) -> List[str]:
        """Auto-detect supported languages from available indexes."""
        languages = set()

        # Check BM25 indexes
        if self.index_base_dir.exists():
            for index_dir in self.index_base_dir.iterdir():
                if index_dir.is_dir() and index_dir.name.startswith("msmarco_"):
                    lang = index_dir.name.replace("msmarco_", "")
                    languages.add(lang)

        # Check dense indexes
        if self.dense_index_base_dir.exists():
            for meta_file in self.dense_index_base_dir.glob("*.meta.json"):
                # e.g. msmarco_en.meta.json -> "msmarco_en.meta" (stem), then -> "en"
                stem = meta_file.stem  # "msmarco_en.meta"
                lang = stem.replace("msmarco_", "").replace(".meta", "")
                languages.add(lang)

        return sorted(list(languages)) if languages else ["en"]  # Default to English

    def _load_searchers_for_language(self, lang: str) -> None:
        """Load BM25 and Dense searchers for a specific language."""
        # BM25 searcher
        bm25_index_dir = self.index_base_dir / f"msmarco_{lang}"
        if bm25_index_dir.exists():
            self.bm25_searchers[lang] = BM25Searcher(
                index_dir=str(bm25_index_dir),
                language=lang,
            )

        # Dense searcher
        dense_meta_path = self.dense_index_base_dir / f"msmarco_{lang}.meta.json"
        if dense_meta_path.exists():
            self.dense_searchers[lang] = DenseSearcher(meta_path=str(dense_meta_path))

    # -------------------------------------------------------------------------
    # General entrypoint for code (eval, UI, etc.)
    # -------------------------------------------------------------------------

    def search(
        self,
        query_text: str,
        query_lang: str,
        target_lang: str,
        k: int = 10,
        mode: str = "hybrid",  # "bm25", "dense", "hybrid"
        fusion_alpha: float | None = None,
        return_text: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        General retrieval entrypoint.

        Args:
            query_text: Search query.
            query_lang: Language code of the query (for logging/analysis only for now).
            target_lang: Target document language code (e.g., 'en', 'hi', 'sv').
            k: Number of results to return.
            mode: 'bm25', 'dense', or 'hybrid'.
            fusion_alpha: Optional override of fusion weight for hybrid mode.
            return_text: If True, include 'text' in dense/hybrid hits (needed for reranking).

        Returns:
            List of result dicts. Keys:
              - BM25:  {'doc_id', 'bm25_score'}
              - Dense: {'doc_id', 'dense_score', 'lang', 'text'?}
              - Hybrid: {'doc_id', 'bm25_score', 'dense_score', 'fused_score', 'text'?}
        """
        if target_lang not in self.supported_languages:
            raise ValueError(
                f"Target language '{target_lang}' not supported. "
                f"Available languages: {self.supported_languages}. "
                f"Build indexes for '{target_lang}' first."
            )

        bm25_searcher = self.bm25_searchers.get(target_lang)
        dense_searcher = self.dense_searchers.get(target_lang)

        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError(f"Unknown mode '{mode}'. Use 'bm25', 'dense', or 'hybrid'.")

        # ---------------- BM25-only ----------------
        if mode == "bm25":
            if bm25_searcher is None:
                raise ValueError(f"No BM25 index for target language '{target_lang}'.")
            return bm25_searcher.search(query_text, k=k)

        # ---------------- Dense-only ----------------
        if mode == "dense":
            if dense_searcher is None:
                raise ValueError(f"No dense index for target language '{target_lang}'.")
            return dense_searcher.search(query_text, k=k, return_text=return_text)

        # ---------------- Hybrid (BM25 + Dense) ----------------
        if bm25_searcher is None and dense_searcher is None:
            raise ValueError(
                f"No indexes found for target language '{target_lang}'. "
                f"Build BM25 and/or dense indexes first."
            )

        bm25_hits: List[Dict[str, Any]] = []
        dense_hits: List[Dict[str, Any]] = []

        if bm25_searcher is not None:
            bm25_hits = bm25_searcher.search(query_text, k=k * 2)

        if dense_searcher is not None:
            dense_hits = dense_searcher.search(query_text, k=k * 2, return_text=return_text)

        # If only one system is available, behave gracefully
        if not bm25_hits and dense_hits:
            # Dense-only path, but may still be called under 'hybrid' mode
            return dense_hits[:k]
        if bm25_hits and not dense_hits:
            # BM25-only path under 'hybrid' mode
            for hit in bm25_hits:
                hit.setdefault("dense_score", 0.0)
                hit.setdefault("fused_score", hit["bm25_score"])
            return bm25_hits[:k]

        alpha = fusion_alpha if fusion_alpha is not None else self.fusion_alpha
        fused_results = fuse_scores(
            bm25_hits=bm25_hits,
            dense_hits=dense_hits,
            alpha=alpha,
            k=k,
        )

        # Attach text if requested (for CE reranking)
        if return_text and dense_hits:
            text_lookup = {h["doc_id"]: h.get("text", "") for h in dense_hits}
            for item in fused_results:
                item["text"] = text_lookup.get(item["doc_id"], "")

        return fused_results

    # -------------------------------------------------------------------------
    # Backward-compat API for old CLI (`cross_lingual_search`)
    # -------------------------------------------------------------------------

    def cross_lingual_search(
        self,
        query_text: str,
        query_lang: str,
        target_lang: str,
        k: int = 10,
        fusion_alpha: float | None = None,
        return_text: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Convenience helper: default hybrid retrieval with optional text.

        This is what:
          - demo_search_cli.py
          - eval_utils.py (for hybrid / hybrid_ce)
        currently call.
        """
        return self.search(
            query_text=query_text,
            query_lang=query_lang,
            target_lang=target_lang,
            k=k,
            mode="hybrid",
            fusion_alpha=fusion_alpha,
            return_text=return_text,
        )

    def close(self) -> None:
        """Close connections to all dense indices."""
        for searcher in self.dense_searchers.values():
            searcher.close()
