#!/usr/bin/env python3
import subprocess

directions = [
    ("clirmatrix_hi_en_dev", "hi", "en"),
    ("clirmatrix_sv_en_dev", "sv", "en"),
    ("clirmatrix_en_hi_dev", "en", "hi"),
    ("clirmatrix_en_sv_dev", "en", "sv"),
]

alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

for name, qlang, tlang in directions:
    print(f"\n=== Sweeping α for {qlang} → {tlang} ({name}) ===")

    topics = f"eval/topics/{name}.jsonl"
    qrels  = f"eval/qrels/{name}.tsv"

    for a in alphas:
        print(f"\n--> α = {a}")
        cmd = [
            "python", "scripts/eval_retrieval.py",
            "--topics", topics,
            "--qrels", qrels,
            "--mode", "hybrid",
            "--alpha", str(a)
        ]
        subprocess.run(cmd)