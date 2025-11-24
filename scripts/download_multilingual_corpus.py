#!/usr/bin/env python3
"""
Download and prepare Hindi and Swedish document corpora for cross-lingual retrieval.

Uses CLIRMatrix or similar multilingual datasets to get document collections.
For now, we'll create a small synthetic dataset for demonstration, but this can be
extended to use real CLIRMatrix data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data/processed"


def download_hindi_corpus(output_path: Path, num_docs: int = 2000) -> None:
    """
    Download/prepare Hindi corpus.
    
    For demonstration, we'll use a small subset. In production, you'd use CLIRMatrix
    or other Hindi document collections.
    """
    print(f"[download_multilingual] Preparing Hindi corpus ({num_docs} docs)...")
    
    # Try to use CLIRMatrix or fallback to synthetic data
    try:
        # Attempt to load CLIRMatrix Hindi-English data
        # CLIRMatrix has document collections, but structure varies
        # For now, create a simple synthetic corpus
        print("[download_multilingual] Creating synthetic Hindi corpus (replace with CLIRMatrix in production)")
        
        # Sample Hindi texts (in a real scenario, these would come from CLIRMatrix)
        hindi_samples = [
            "मशीन लर्निंग कृत्रिम बुद्धिमत्ता का एक उपक्षेत्र है जो कंप्यूटर को डेटा से सीखने की क्षमता प्रदान करता है।",
            "प्राकृतिक भाषा प्रसंस्करण कंप्यूटर और मानव भाषा के बीच बातचीत का अध्ययन है।",
            "गहन शिक्षण तंत्रिका नेटवर्क का उपयोग करके जटिल पैटर्न सीखने की एक विधि है।",
            "सूचना पुनर्प्राप्ति प्रणाली उपयोगकर्ता की जानकारी आवश्यकताओं को पूरा करने के लिए दस्तावेज़ खोजने में मदद करती है।",
            "क्रॉस-भाषाई सूचना पुनर्प्राप्ति एक भाषा में क्वेरी का उपयोग करके दूसरी भाषा में दस्तावेज़ खोजने की प्रक्रिया है।",
        ]
        
        with output_path.open("w", encoding="utf-8") as f:
            for i in range(num_docs):
                # Cycle through samples and add variations
                text = hindi_samples[i % len(hindi_samples)]
                # Add some variation
                if i > len(hindi_samples):
                    text = f"{text} यह दस्तावेज़ संख्या {i} है।"
                
                f.write(json.dumps({
                    "doc_id": f"hi_{i}",
                    "lang": "hi",
                    "text": text
                }, ensure_ascii=False) + "\n")
        
        print(f"✅ Saved Hindi corpus to {output_path}")
        
    except Exception as e:
        print(f"⚠️  Error downloading Hindi corpus: {e}")
        print("Creating minimal synthetic corpus...")
        # Fallback: create minimal corpus
        with output_path.open("w", encoding="utf-8") as f:
            for i in range(min(100, num_docs)):
                f.write(json.dumps({
                    "doc_id": f"hi_{i}",
                    "lang": "hi",
                    "text": f"हिंदी दस्तावेज़ {i}: यह एक नमूना पाठ है।"
                }, ensure_ascii=False) + "\n")
        print(f"✅ Created minimal Hindi corpus at {output_path}")


def download_swedish_corpus(output_path: Path, num_docs: int = 2000) -> None:
    """
    Download/prepare Swedish corpus.
    
    For demonstration, we'll use a small subset. In production, you'd use CLIRMatrix
    or other Swedish document collections.
    """
    print(f"[download_multilingual] Preparing Swedish corpus ({num_docs} docs)...")
    
    try:
        print("[download_multilingual] Creating synthetic Swedish corpus (replace with CLIRMatrix in production)")
        
        # Sample Swedish texts
        swedish_samples = [
            "Maskininlärning är en delmängd av artificiell intelligens som ger datorer förmågan att lära sig från data.",
            "Naturlig språkbehandling är studiet av interaktionen mellan datorer och mänskligt språk.",
            "Djupinlärning är en metod för att lära sig komplexa mönster med hjälp av neurala nätverk.",
            "Informationsåtervinning är processen att hitta dokument som uppfyller en användares informationsbehov.",
            "Tvärspråklig informationsåtervinning är processen att söka efter dokument på ett språk med hjälp av frågor på ett annat språk.",
        ]
        
        with output_path.open("w", encoding="utf-8") as f:
            for i in range(num_docs):
                text = swedish_samples[i % len(swedish_samples)]
                if i > len(swedish_samples):
                    text = f"{text} Detta är dokument nummer {i}."
                
                f.write(json.dumps({
                    "doc_id": f"sv_{i}",
                    "lang": "sv",
                    "text": text
                }, ensure_ascii=False) + "\n")
        
        print(f"✅ Saved Swedish corpus to {output_path}")
        
    except Exception as e:
        print(f"⚠️  Error downloading Swedish corpus: {e}")
        print("Creating minimal synthetic corpus...")
        with output_path.open("w", encoding="utf-8") as f:
            for i in range(min(100, num_docs)):
                f.write(json.dumps({
                    "doc_id": f"sv_{i}",
                    "lang": "sv",
                    "text": f"Svenskt dokument {i}: Detta är en exempeltext."
                }, ensure_ascii=False) + "\n")
        print(f"✅ Created minimal Swedish corpus at {output_path}")


def main() -> None:
    import argparse
    
    parser = argparse.ArgumentParser(description="Download/prepare Hindi and Swedish corpora")
    parser.add_argument("--lang", type=str, choices=["hi", "sv", "all"], default="all",
                       help="Language to download (hi, sv, or all)")
    parser.add_argument("--num-docs", type=int, default=2000,
                       help="Number of documents per language (default: 2000)")
    args = parser.parse_args()
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.lang in ("hi", "all"):
        hindi_path = OUT_DIR / "corpus_hi.jsonl"
        download_hindi_corpus(hindi_path, num_docs=args.num_docs)
    
    if args.lang in ("sv", "all"):
        swedish_path = OUT_DIR / "corpus_sv.jsonl"
        download_swedish_corpus(swedish_path, num_docs=args.num_docs)
    
    print("\n✅ Multilingual corpus preparation complete!")


if __name__ == "__main__":
    main()

