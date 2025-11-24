# retrieval/eval_utils.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any

import json

from retrieval.metrics import Qrels, Run, compute_all_metrics
from retrieval.multilingual_search import MultilingualRetriever


def _resolve_path(path_str: str) -> Path:
    """
    Resolve a path string relative to project root if not absolute.
    """
    p = Path(path_str)
    if p.is_absolute():
        return p
    # project root = retrieval/.. (one level up)
    return Path(__file__).resolve().parent.parent / p


def load_qrels_tsv(path: str) -> Qrels:
    """
    Load qrels from a simple TSV/whitespace file:

        qid<TAB>doc_id<TAB>relevance_int
    """
    qrels: Qrels = {}
    fpath = _resolve_path(path)

    with fpath.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Be tolerant: split on any whitespace (tabs or spaces)
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Invalid qrels line: {line}")

            qid, doc_id, rel_str = parts[0], parts[1], parts[2]
            try:
                rel = int(rel_str)
            except ValueError:
                raise ValueError(f"Invalid relevance value in qrels line: {line}")

            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][doc_id] = rel

    return qrels


def load_topics_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Load topics from jsonl.

    Each line:
    {
      "qid": "1",
      "query": "...",
      "query_lang": "hi",
      "target_lang": "en"
    }
    """
    topics: List[Dict[str, Any]] = []
    fpath = _resolve_path(path)

    with fpath.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            topics.append(obj)
    return topics


def run_retrieval_eval(
    topics_path: str,
    qrels_path: str,
    mode: str = "hybrid",          # "bm25", "dense", "hybrid", "hybrid_ce"
    alpha: float = 0.5,
    k: int = 100,
    ndcg_k: int = 10,
    mrr_k: int = 10,
    recall_k: int = 100,
) -> Dict[str, float]:
    """
    High-level evaluation pipeline.

    1. Load topics + qrels
    2. For each query, run retrieval via MultilingualRetriever
    3. Optionally apply cross-encoder reranking (mode == "hybrid_ce")
    4. Compute nDCG@k, MRR@k, Recall@k over all topics.
    """
    topics = load_topics_jsonl(topics_path)
    qrels = load_qrels_tsv(qrels_path)

    retriever = MultilingualRetriever(fusion_alpha=alpha)

    # Lazy import to avoid loading cross-encoder when not needed
    ce_reranker = None
    if mode == "hybrid_ce":
        from retrieval.reranker import CrossEncoderReranker
        ce_reranker = CrossEncoderReranker()

    run: Run = {}

    for t in topics:
        qid = str(t["qid"])
        query = t["query"]
        query_lang = t.get("query_lang", "en")
        target_lang = t.get("target_lang", "en")

        # --- Retrieval phase ---
        if mode == "hybrid_ce":
            # For reranking, retrieve a deeper candidate pool (e.g., 100)
            base_k = max(k, 100)
            hits = retriever.cross_lingual_search(
                query_text=query,
                query_lang=query_lang,
                target_lang=target_lang,
                k=base_k,
                fusion_alpha=alpha,
                return_text=True,   # we need doc text for cross-encoder
            )
            # --- Reranking phase ---
            assert ce_reranker is not None
            hits = ce_reranker.rerank(
                query_text=query,
                hits=hits,
                top_k=k,  # keep top-k after reranking
            )
        else:
            # No reranking: standard retrieval
            hits = retriever.cross_lingual_search(
                query_text=query,
                query_lang=query_lang,
                target_lang=target_lang,
                k=k,
                fusion_alpha=alpha,
                return_text=False,
            )

        # [Optional debug] show which docs we retrieved per query
        # print(f"[DEBUG] qid={qid} retrieved doc_ids: {[h['doc_id'] for h in hits]}")

        # --- Build Run structure ---
        scored_hits: List[tuple[str, float]] = []
        for h in hits:
            doc_id = str(h["doc_id"])
            if mode == "bm25":
                score = float(h.get("bm25_score", 0.0))
            elif mode == "dense":
                score = float(h.get("dense_score", 0.0))
            elif mode == "hybrid":
                score = float(h.get("fused_score", h.get("dense_score", 0.0)))
            elif mode == "hybrid_ce":
                score = float(h.get("ce_score", h.get("fused_score", 0.0)))
            else:
                raise ValueError(f"Unknown mode: {mode}")

            scored_hits.append((doc_id, score))

        run[qid] = scored_hits

    retriever.close()

    # --- Metric computation ---
    metrics = compute_all_metrics(
        run=run,
        qrels=qrels,
        ndcg_k=ndcg_k,
        mrr_k=mrr_k,
        recall_k=recall_k,
    )
    return metrics
