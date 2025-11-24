from typing import List, Dict
from pyserini.search.lucene import LuceneSearcher

class BM25Searcher:
    """
    Thin wrapper around Pyserini's LuceneSearcher for BM25 retrieval.
    """

    def __init__(self, index_dir: str, language: str = "en"):
        self.language = language
        # This path must match the index you just built.
        self.searcher = LuceneSearcher(index_dir)

        # You can play with BM25 hyperparams later.
        # MS MARCO-ish defaults around k1=0.9, b=0.4 are common,
        # but leave defaults first just to make sure it works.
        # If LuceneSearcher complains about set_bm25 missing in this version,
        # just comment it out.
        try:
            self.searcher.set_bm25(k1=0.9, b=0.4)
        except Exception:
            pass

    def search(self, query_text: str, k: int = 10) -> List[Dict]:
        hits = self.searcher.search(query_text, k=k)
        out = []
        for h in hits:
            out.append({
                "doc_id": h.docid,
                "bm25_score": float(h.score),
            })
        return out
