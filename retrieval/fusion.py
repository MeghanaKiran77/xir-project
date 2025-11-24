from __future__ import annotations

from typing import Any, Dict, List


def fuse_scores(
    bm25_hits: List[Dict[str, Any]],
    dense_hits: List[Dict[str, Any]],
    alpha: float = 0.5,
    k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Fuse BM25 and Dense retrieval results using a weighted linear combination.

    Args:
        bm25_hits: List of BM25 results, each with 'doc_id' and 'bm25_score'
        dense_hits: List of Dense results, each with 'doc_id' and 'dense_score'
        alpha: Weight for BM25 (0-1). fused_score = alpha * bm25 + (1-alpha) * dense
        k: Number of top results to return

    Returns:
        List of fused results, each with 'doc_id', 'bm25_score', 'dense_score', 'fused_score'
    """
    # Normalize scores to [0, 1] range using min-max normalization
    # This ensures both BM25 and Dense scores are on the same scale
    bm25_scores = {hit["doc_id"]: hit["bm25_score"] for hit in bm25_hits}
    dense_scores = {hit["doc_id"]: hit["dense_score"] for hit in dense_hits}

    # Get all unique doc_ids
    all_doc_ids = set(bm25_scores.keys()) | set(dense_scores.keys())

    if not all_doc_ids:
        return []

    # Normalize BM25 scores to [0, 1]
    bm25_values = [v for v in bm25_scores.values() if v > 0]  # Only consider positive scores
    if bm25_values:
        bm25_min = min(bm25_values)
        bm25_max = max(bm25_values)
        bm25_range = bm25_max - bm25_min if bm25_max > bm25_min else 1.0
    else:
        # No positive BM25 scores - all will be normalized to 0
        bm25_min, bm25_max, bm25_range = 0.0, 1.0, 1.0

    # Normalize Dense scores to [0, 1]
    # Dense scores from normalized embeddings are typically in [0, 1] already, but we normalize anyway
    dense_values = list(dense_scores.values())
    if dense_values:
        dense_min = min(dense_values)
        dense_max = max(dense_values)
        dense_range = dense_max - dense_min if dense_max > dense_min else 1.0
    else:
        dense_min, dense_max, dense_range = 0.0, 1.0, 1.0

    # Compute fused scores for all documents
    fused_results: List[Dict[str, Any]] = []
    for doc_id in all_doc_ids:
        bm25_score = bm25_scores.get(doc_id, 0.0)
        dense_score = dense_scores.get(doc_id, 0.0)

        # Normalize scores to [0, 1]
        if bm25_range > 0 and bm25_score > 0:
            bm25_norm = (bm25_score - bm25_min) / bm25_range
        else:
            bm25_norm = 0.0

        if dense_range > 0:
            dense_norm = (dense_score - dense_min) / dense_range
        else:
            dense_norm = 0.0

        # Clamp to [0, 1] to handle any floating point issues
        bm25_norm = max(0.0, min(1.0, bm25_norm))
        dense_norm = max(0.0, min(1.0, dense_norm))

        # Fuse: alpha * bm25 + (1 - alpha) * dense
        fused_score = alpha * bm25_norm + (1.0 - alpha) * dense_norm

        fused_results.append({
            "doc_id": doc_id,
            "bm25_score": bm25_score,
            "dense_score": dense_score,
            "fused_score": fused_score,
        })

    # Sort by fused_score descending and return top-k
    fused_results.sort(key=lambda x: x["fused_score"], reverse=True)
    return fused_results[:k]

