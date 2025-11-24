from __future__ import annotations
from typing import List, Dict
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query_text: str, hits: List[Dict], top_k: int = 50) -> List[Dict]:
        """
        Rerank top-K hits using a multilingual MS MARCO cross-encoder.

        Args:
            query_text: the query string (any language)
            hits: list of dicts with fields: doc_id, text, bm25_score, dense_score, fused_score
            top_k: how many candidates to rerank

        Returns:
            Reranked list of hits with new field "ce_score".
        """

        # Select top_k from hybrid
        candidates = hits[:top_k]

        # Build inference pairs
        pairs = [[query_text, doc["text"]] for doc in candidates]

        # Predict cross-encoder scores
        scores = self.model.predict(pairs)

        # Attach scores
        for h, s in zip(candidates, scores):
            h["ce_score"] = float(s)

        # Sort by CE score (descending)
        candidates.sort(key=lambda x: x["ce_score"], reverse=True)
        return candidates