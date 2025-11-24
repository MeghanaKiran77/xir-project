# retrieval/metrics.py

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import math

# Types
# qid -> {doc_id: relevance_int}
Qrels = Dict[str, Dict[str, int]]
# qid -> [(doc_id, score), ...] sorted by score (desc)
Run = Dict[str, List[Tuple[str, float]]]


def _dcg_at_k(relevances: Sequence[float], k: int) -> float:
    """Discounted Cumulative Gain at rank k."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        rank = i + 1
        dcg += (2.0**rel - 1.0) / math.log2(rank + 1.0)
    return dcg


def ndcg_at_k(run: Run, qrels: Qrels, k: int = 10) -> float:
    """
    Compute nDCG@k averaged over queries.

    run:  qid -> list[(doc_id, score)] sorted by score desc
    qrels: qid -> {doc_id: relevance_int}
    """
    scores: List[float] = []

    for qid, hits in run.items():
        if qid not in qrels:
            continue
        rel_dict = qrels[qid]

        rels_run = [float(rel_dict.get(doc_id, 0)) for doc_id, _ in hits[:k]]
        dcg = _dcg_at_k(rels_run, k)

        ideal_rels = sorted((float(r) for r in rel_dict.values()), reverse=True)
        idcg = _dcg_at_k(ideal_rels, k)

        if idcg <= 0.0:
            continue

        scores.append(dcg / idcg)

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def mrr_at_k(run: Run, qrels: Qrels, k: int = 10) -> float:
    """
    Compute MRR@k (binary relevance: rel > 0).
    """
    mrrs: List[float] = []

    for qid, hits in run.items():
        if qid not in qrels:
            continue
        rel_dict = qrels[qid]

        rr = 0.0
        for i, (doc_id, _) in enumerate(hits[:k]):
            if rel_dict.get(doc_id, 0) > 0:
                rr = 1.0 / float(i + 1)
                break

        mrrs.append(rr)

    if not mrrs:
        return 0.0
    return sum(mrrs) / len(mrrs)


def recall_at_k(run: Run, qrels: Qrels, k: int = 100) -> float:
    """
    Compute Recall@k (binary relevance: rel > 0).

    #rel docs retrieved in top-k / #rel docs.
    """
    recalls: List[float] = []

    for qid, rel_dict in qrels.items():
        relevant_docs = {d for d, rel in rel_dict.items() if rel > 0}
        if not relevant_docs:
            continue

        retrieved_docs = set()
        for doc_id, _ in run.get(qid, [])[:k]:
            if doc_id in relevant_docs:
                retrieved_docs.add(doc_id)

        recalls.append(len(retrieved_docs) / float(len(relevant_docs)))

    if not recalls:
        return 0.0
    return sum(recalls) / len(recalls)


def compute_all_metrics(
    run: Run,
    qrels: Qrels,
    ndcg_k: int = 10,
    mrr_k: int = 10,
    recall_k: int = 100,
) -> Dict[str, float]:
    """
    Convenience wrapper: returns {metric_name: value}.
    """
    return {
        f"nDCG@{ndcg_k}": ndcg_at_k(run, qrels, ndcg_k),
        f"MRR@{mrr_k}": mrr_at_k(run, qrels, mrr_k),
        f"Recall@{recall_k}": recall_at_k(run, qrels, recall_k),
    }
