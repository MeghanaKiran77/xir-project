#!/usr/bin/env python3
"""
Download and convert CLIRMatrix (BI-139 base) subsets for Hindi, Swedish, and English.

We do three things:

1. Build monolingual corpora for each language:
   - data/processed/corpus_en.jsonl
   - data/processed/corpus_hi.jsonl
   - data/processed/corpus_sv.jsonl

   Each line:
     {"doc_id": "...", "lang": "en|hi|sv", "text": "..."}

2. Build topics + qrels for 4 cross-lingual directions (dev split only):
   - hi -> en  : clirmatrix/en/bi139-base/hi/dev
   - en -> hi  : clirmatrix/hi/bi139-base/en/dev
   - sv -> en  : clirmatrix/en/bi139-base/sv/dev
   - en -> sv  : clirmatrix/sv/bi139-base/en/dev

   Topics:
     eval/topics/clirmatrix_hi_en_dev.jsonl
     eval/topics/clirmatrix_en_hi_dev.jsonl
     eval/topics/clirmatrix_sv_en_dev.jsonl
     eval/topics/clirmatrix_en_sv_dev.jsonl

   Qrels:
     eval/qrels/clirmatrix_hi_en_dev.tsv
     eval/qrels/clirmatrix_en_hi_dev.tsv
     eval/qrels/clirmatrix_sv_en_dev.tsv
     eval/qrels/clirmatrix_en_sv_dev.tsv

3. These corpora + eval files are compatible with your existing:
   - scripts/build_bm25.py
   - scripts/build_dense.py
   - scripts/eval_retrieval.py
   - retrieval/MultilingualRetriever
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple
import ir_datasets

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EVAL_TOPICS_DIR = PROJECT_ROOT / "eval" / "topics"
EVAL_QRELS_DIR = PROJECT_ROOT / "eval" / "qrels"


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "jsonl").mkdir(parents=True, exist_ok=True)
    EVAL_TOPICS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_QRELS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------- CORPUS BUILDING -------------------------


def build_corpus_for_lang(lang: str, overwrite: bool = False) -> Path:
    """
    Build a monolingual corpus JSONL from CLIRMatrix docs:

      dataset: clirmatrix/{lang}
      output : data/processed/corpus_{lang}.jsonl

    We concatenate title + text when available.
    """
    ds_id = f"clirmatrix/{lang}"
    ds = ir_datasets.load(ds_id)

    out_path = PROCESSED_DIR / f"corpus_{lang}.jsonl"
    if out_path.exists() and not overwrite:
        print(f"[clirmatrix] Corpus already exists, skipping: {out_path}")
        return out_path

    print(f"[clirmatrix] Building corpus for lang='{lang}' from dataset '{ds_id}'")
    count = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for doc in ds.docs_iter():
            # GenericDoc typically has: doc_id, title, text
            doc_id = str(doc.doc_id)
            title = getattr(doc, "title", None) or ""
            text = getattr(doc, "text", None) or ""

            if not text and not title:
                continue

            if title and text:
                full_text = f"{title}. {text}"
            else:
                full_text = title or text

            record = {
                "doc_id": doc_id,
                "lang": lang,
                "text": full_text,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"[clirmatrix] Wrote {count:,} docs to {out_path}")
    return out_path


# ------------------------- TOPICS + QRELS -------------------------


def build_topics_and_qrels_for_pair(
    query_lang: str,
    doc_lang: str,
    split: str = "dev",
    overwrite: bool = False,
) -> Tuple[Path, Path]:
    """
    Build topics (queries) and qrels for a given CLIRMatrix pair.

    CLIRMatrix ID pattern (BI-139 base):
      clirmatrix/{doc_lang}/bi139-base/{query_lang}/{split}

    Example:
      hi -> en (queries in hi, docs in en):
        doc_lang   = "en"
        query_lang = "hi"
        dataset id = "clirmatrix/en/bi139-base/hi/dev"
    """
    ds_id = f"clirmatrix/{doc_lang}/bi139-base/{query_lang}/{split}"
    ds = ir_datasets.load(ds_id)

    topics_out = EVAL_TOPICS_DIR / f"clirmatrix_{query_lang}_{doc_lang}_{split}.jsonl"
    qrels_out = EVAL_QRELS_DIR / f"clirmatrix_{query_lang}_{doc_lang}_{split}.tsv"

    if topics_out.exists() and qrels_out.exists() and not overwrite:
        print(f"[clirmatrix] Topics + qrels already exist for {query_lang}->{doc_lang}, skipping.")
        return topics_out, qrels_out

    print(f"[clirmatrix] Building topics + qrels for {query_lang} -> {doc_lang} from '{ds_id}'")

    # --- Topics ---
    # ir_datasets gives GenericQuery: query_id, text
    with topics_out.open("w", encoding="utf-8") as ft:
        for q in ds.queries_iter():
            qid = str(q.query_id)
            text = getattr(q, "text", None) or ""
            if not text:
                continue
            rec = {
                "qid": qid,
                "query": text,
                "query_lang": query_lang,
                "target_lang": doc_lang,
            }
            ft.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # --- Qrels ---
    # TrecQrel: query_id, doc_id, relevance
    with qrels_out.open("w", encoding="utf-8") as fq:
        for qr in ds.qrels_iter():
            qid = str(qr.query_id)
            doc_id = str(qr.doc_id)
            rel = int(qr.relevance)
            # Your loader uses line.split(), so any whitespace is fine.
            fq.write(f"{qid}\t{doc_id}\t{rel}\n")

    print(f"[clirmatrix] Wrote topics to {topics_out}")
    print(f"[clirmatrix] Wrote qrels to  {qrels_out}")
    return topics_out, qrels_out


# ------------------------- MAIN -------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download & convert CLIRMatrix (BI-139 base) for hi/en/sv."
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=["en", "hi", "sv"],
        help="Languages to build corpora for (default: en hi sv).",
    )
    parser.add_argument(
        "--build-pairs",
        action="store_true",
        help="Also build topics + qrels for 4 cross-lingual directions "
             "(hi<->en, sv<->en) on 'dev' split.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing corpora/topics/qrels if present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    # 1) Build corpora for requested languages
    for lang in args.langs:
        build_corpus_for_lang(lang, overwrite=args.overwrite)

    # 2) Build topics + qrels for 4 directions (dev split)
    if args.build_pairs:
        # 4 core directions: hi<->en, sv<->en
        pairs = [
            ("hi", "en"),  # hi -> en
            ("en", "hi"),  # en -> hi
            ("sv", "en"),  # sv -> en
            ("en", "sv"),  # en -> sv
        ]
        for query_lang, doc_lang in pairs:
            build_topics_and_qrels_for_pair(
                query_lang=query_lang,
                doc_lang=doc_lang,
                split="dev",
                overwrite=args.overwrite,
            )

    print("\n[clirmatrix] Done.")


if __name__ == "__main__":
    main()
