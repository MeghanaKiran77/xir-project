# scripts/eval_retrieval.py

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is importable like in demo_search_cli.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.eval_utils import run_retrieval_eval  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate BM25 / Dense / Hybrid / Hybrid+CE CLIR retrieval."
    )
    parser.add_argument(
        "--topics",
        type=str,
        required=True,
        help="Path to topics jsonl (e.g., eval/topics/en_dev.jsonl).",
    )
    parser.add_argument(
        "--qrels",
        type=str,
        required=True,
        help="Path to qrels tsv (e.g., eval/qrels/en_dev.tsv).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["bm25", "dense", "hybrid", "hybrid_ce"],
        help="Retrieval mode.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Fusion weight α for hybrid mode (0–1).",
    )
    parser.add_argument(
        "-k",
        "--top_k",
        type=int,
        default=100,
        help="Top-k documents to retrieve.",
    )
    parser.add_argument(
        "--ndcg_k",
        type=int,
        default=10,
        help="nDCG cutoff.",
    )
    parser.add_argument(
        "--mrr_k",
        type=int,
        default=10,
        help="MRR cutoff.",
    )
    parser.add_argument(
        "--recall_k",
        type=int,
        default=100,
        help="Recall cutoff.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print metrics as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    metrics = run_retrieval_eval(
        topics_path=args.topics,
        qrels_path=args.qrels,
        mode=args.mode,
        alpha=args.alpha,
        k=args.top_k,
        ndcg_k=args.ndcg_k,
        mrr_k=args.mrr_k,
        recall_k=args.recall_k,
    )

    if args.pretty:
        print(json.dumps(metrics, indent=2))
    else:
        # Simple TSV-style output
        for name, value in metrics.items():
            print(f"{name}\t{value:.4f}")


if __name__ == "__main__":
    main()
