# XIR — Cross-Lingual Hybrid Retrieval
# Hindi ↔ Swedish ↔ English Semantic Search Engine

## Hybrid BM25 + Dense FAISS + Fusion + CLIRMatrix Evaluation

## 1. Project Overview

XIR is a full cross-lingual information retrieval (CLIR) system that supports:
Hindi, Swedish, and English.

It is designed to retrieve documents across languages (e.g., Hindi query → Swedish documents) using a hybrid architecture:

- Sparse retrieval (BM25 / Lucene / Pyserini)
- Dense retrieval (multilingual-e5 embeddings + FAISS)
- Hybrid score fusion (α-weighted interpolation)
- Cross-lingual routing
- Evaluation using CLIRMatrix Qrels

The system is structured with clear modularity and reproducibility.

## 2. System Architecture
               ┌────────────────────────────┐
               │    User Query (hi/en/sv)   │
               └───────────────┬────────────┘
                               ▼
                 ┌──────────────────────────┐
                 │  MultilingualRetriever   │
                 └───────────────┬──────────┘
     ┌───────────────────────────┼──────────────────────────┐
     ▼                           ▼                          ▼
┌────────────┐            ┌─────────────┐           ┌────────────────┐
│  BM25      │            │ Dense (E5)  │           │  Cross-lingual │
│ (Pyserini) │            │ (FAISS-HNSW)│           │   Routing      │
└────────────┘            └─────────────┘           └────────────────┘
                 ┌──────────────────────────┐
                 │      Fusion (α-mix)      │
                 └──────────────────────────┘
                               ▼
                   Top-k Ranked Documents

## 3. Datasets Used
- MS MARCO (English subset):
Used to bootstrap the monolingual baseline (Phase 1).

- Synthetic Hindi & Swedish corpora"
Used to expand multilingual testing (Phase 2).

- CLIRMatrix (High-quality multilingual IR benchmark)

Used for official cross-lingual evaluation (Phase 3):
hi → en
en → hi
sv → en
en → sv

These files have been created
eval/topics/clirmatrix_*_dev.jsonl  
eval/qrels/clirmatrix_*_dev.tsv

## Dataset Statistics

| Target Language  | Raw Docs   | Filtered Docs (CLIRMatrix-only)  | Final Indexed Docs  | Query Directions   | Dense Encoding Time  |
|------------------|------------|----------------------------------|---------------------|--------------------|----------------------|
| **English (en)** | ~5,984,197 | **177,020**                      | 177,020             | hi → en, sv → en   | ~11 hours            |
| **Hindi (hi)**   | ~118,413   | **61,210**                       | 61,210              | en → hi            | ~3 hours             |
| **Swedish (sv)** | ~3,728,177 | **98,061**                       | 98,061              | en → sv            | ~7 hours             |


## 4. Key Components
4.1. Sparse Retrieval (BM25)
Powered by Pyserini + Lucene
Separate indexes for en, hi, sv
Fast lexical matching

4.2. Dense Retrieval (Semantic)
Encoder: intfloat/multilingual-e5-base
FAISS index built using HNSW (for fast ANN)
SQLite docstores for metadata

4.3 Hybrid Retrieval (Fusion)
Score =
α * normalized_bm25 + (1 - α) * dense_score
Defaults:
α = 0.5
k = 100

4.4 Cross-lingual Routing
Automatic mapping of:
query_lang → target_lang for 9 retrieval directions (hi→en, sv→hi, en→sv, etc.)

## 5. Evaluation Framework (CLIRMatrix)
These were built:
scripts/eval_retrieval.py
retrieval/metrics.py
retrieval/eval_utils.py

### Metrics:
nDCG@10
MRR@10
Recall@100

Topics & qrels, that are provided in the eval/ directory, will automatically load and evaluate all queries.

Example command:

python scripts/eval_retrieval.py \
  --topics eval/topics/clirmatrix_hi_en_dev.jsonl \
  --qrels eval/qrels/clirmatrix_hi_en_dev.tsv \
  --mode hybrid --pretty

## 6. Demo CLI

Example:
python scripts/demo_search_cli.py \
  --query "मशीन लर्निंग क्या है" \
  --query_lang hi \
  --target_lang sv \
  --k 5

## 7. Repository Structure
retrieval/
    bm25_search.py
    dense_search.py
    fusion.py
    multilingual_search.py
    metrics.py
    eval_utils.py
index_dense/
index_sparse/
data/processed/
eval/topics/
eval/qrels/
scripts/
ui/app.py

## 8. Workflow
### ✔ Phase 1 — English Monolingual Baseline
MS MARCO subset downloaded
BM25 + Dense implemented
Hybrid fusion functional
CLI demo works

### ✔ Phase 2 — Multilingual Retrieval
Synthetic hi/sv corpora
Built BM25 + Dense indexes for all languages
Implemented full cross-lingual routing
9-direction retrieval functional

### ✔ Phase 3 — CLIRMatrix
Full CLIRMatrix download & conversion
Topics and qrels generated
Corpus filtered to relevant doc_ids
Dense FAISS indexes built using HNSW
Cross-lingual queries producing valid results

## 9. Future Work (Next Phases)
### Phase 3.5 — Cross-Encoder Reranking
MiniLM-based re-ranking for top-50 candidates

### Phase 4 — Streamlit UI
Interactive multilingual search demo

## 10. License
MIT License