#!/usr/bin/env python3
"""
Filter CLIRMatrix corpora down to only the documents that appear in dev qrels.

Input:
  data/processed/corpus_en.jsonl
  data/processed/corpus_hi.jsonl
  data/processed/corpus_sv.jsonl

  eval/qrels/clirmatrix_hi_en_dev.tsv
  eval/qrels/clirmatrix_sv_en_dev.tsv
  eval/qrels/clirmatrix_en_hi_dev.tsv
  eval/qrels/clirmatrix_en_sv_dev.tsv

Output:
  data/processed/corpus_en_filtered.jsonl
  data/processed/corpus_hi_filtered.jsonl
  data/processed/corpus_sv_filtered.jsonl

Each filtered corpus keeps only docs whose doc_id appears in the corresponding qrels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Set


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EVAL_QRELS_DIR = PROJECT_ROOT / "eval" / "qrels"


def load_doc_ids_from_qrels() -> Dict[str, Set[str]]:
    """
    Read the 4 CLIRMatrix dev qrels files and collect doc_ids per language.

    Mapping:
      hi -> en : clirmatrix_hi_en_dev.tsv  (English docs)
      sv -> en : clirmatrix_sv_en_dev.tsv  (English docs)
      en -> hi : clirmatrix_en_hi_dev.tsv  (Hindi docs)
      en -> sv : clirmatrix_en_sv_dev.tsv  (Swedish docs)
    """
    doc_ids_by_lang: Dict[str, Set[str]] = {
        "en": set(),
        "hi": set(),
        "sv": set(),
    }

    qrels_specs = [
        ("clirmatrix_hi_en_dev.tsv", "en"),
        ("clirmatrix_sv_en_dev.tsv", "en"),
        ("clirmatrix_en_hi_dev.tsv", "hi"),
        ("clirmatrix_en_sv_dev.tsv", "sv"),
    ]

    for filename, lang in qrels_specs:
        path = EVAL_QRELS_DIR / filename
        if not path.exists():
            print(f"[filter] WARNING: missing qrels file {path}, skipping.")
            continue
        print(f"[filter] Reading qrels from {path} for lang='{lang}'")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                _, doc_id, _ = parts[0], parts[1], parts[2]
                doc_ids_by_lang[lang].add(doc_id)

    for lang, ids in doc_ids_by_lang.items():
        print(f"[filter] Collected {len(ids):,} doc_ids for lang='{lang}' from qrels.")
    return doc_ids_by_lang


def filter_corpus_for_lang(lang: str, doc_ids: Set[str]) -> None:
    """
    Filter data/processed/corpus_{lang}.jsonl down to only the given doc_ids.

    Writes data/processed/corpus_{lang}_filtered.jsonl.
    """
    in_path = PROCESSED_DIR / f"corpus_{lang}.jsonl"
    out_path = PROCESSED_DIR / f"corpus_{lang}_filtered.jsonl"

    if not in_path.exists():
        print(f"[filter] WARNING: input corpus not found: {in_path}, skipping.")
        return

    if not doc_ids:
        print(f"[filter] No doc_ids for lang='{lang}', skipping filtering.")
        return

    kept = 0
    total = 0
    print(f"[filter] Filtering {in_path} -> {out_path}")
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_id = str(obj.get("doc_id", ""))
            if doc_id in doc_ids:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1

    print(f"[filter] Done lang='{lang}': scanned {total:,} lines, kept {kept:,} docs -> {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter CLIRMatrix corpora to docs appearing in dev qrels.")
    return parser.parse_args()


def main() -> None:
    _ = parse_args()
    doc_ids_by_lang = load_doc_ids_from_qrels()
    for lang in ["en", "hi", "sv"]:
        filter_corpus_for_lang(lang, doc_ids_by_lang[lang])
    print("[filter] All done.")


if __name__ == "__main__":
    main()
