import argparse
import sys
from pathlib import Path

# Ensure project root is importable so `retrieval` package resolves when run as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from retrieval.multilingual_search import MultilingualRetriever
try:
    from eval.metrics_utils import time_call, dummy_metrics  # type: ignore
except Exception:
    # Fallbacks if eval/metrics.utils.py is not importable or is empty
    import time
    from typing import Any, Callable, Dict, List, Tuple

    def time_call(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[Any, float]:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return result, (end - start)

    def dummy_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"num_results": len(results)}

def main():
    print("[debug] starting demo_search_cli.main()")

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--query_lang", type=str, default="en")
    parser.add_argument("--target_lang", type=str, default="en")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    print("[debug] parsed args:", args)

    retriever = MultilingualRetriever()
    print("[debug] created MultilingualRetriever")

    results, latency = time_call(
        retriever.cross_lingual_search,
        args.query,
        args.query_lang,
        args.target_lang,
        args.k
    )

    print("[debug] got results back from cross_lingual_search")

    print(f"Query: {args.query} ({args.query_lang} → {args.target_lang})")
    print(f"Latency: {latency*1000:.2f} ms\n")

    for rank, hit in enumerate(results, start=1):
        print(f"{rank}. doc_id={hit['doc_id']}")
        print(f"   bm25={hit['bm25_score']:.3f} dense={hit['dense_score']:.3f} fused={hit['fused_score']:.3f}")
        print()

    metrics = dummy_metrics(results)
    print("Dummy Metrics:", metrics)

    retriever.close()

if __name__ == "__main__":
    print("[debug] __main__ entry reached")
    main()