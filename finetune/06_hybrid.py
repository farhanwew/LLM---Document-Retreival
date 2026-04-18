"""
Stage 6: BM25 + Dense hybrid retrieval via Reciprocal Rank Fusion (RRF).

Improves recall for queries containing specific Swiss law citations (e.g. "Art. 42 OR",
"BGE 145 II") where exact string matching outperforms dense embeddings.

Usage:
    pip install rank_bm25

    # Sweep thresholds on val:
    python finetune/06_hybrid.py --sweep-val

    # Generate hybrid submission (BM25 + dense → RRF → rerank):
    python finetune/06_hybrid.py \
        --bi-encoder-model scenario1-both \
        --reranker-path finetune/models/bge-reranker-v2-m3 \
        --output submission_hybrid.csv

Notes:
- BM25 index over laws_de (~175k docs) builds in ~90s on CPU. court_considerations
  (2.4M docs) takes ~25min — default is laws_de only (--bm25-laws-only).
- RRF k=60 is the standard constant; rarely needs tuning.
- Install rank_bm25: pip install rank_bm25 (pure Python, offline-safe)
"""

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm

from finetune.config import cfg, get_model_config
from finetune.logger import setup_logger

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("ERROR: rank_bm25 not installed. Run: pip install rank_bm25")
    sys.exit(1)


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

def tokenize_multilingual(text: str) -> list[str]:
    """Simple whitespace+punctuation tokenizer; safe for DE/FR/IT legal text."""
    return re.findall(r'\w+', text.lower())


def build_bm25_index(corpus_texts: list[str]) -> BM25Okapi:
    print(f"  Building BM25 index over {len(corpus_texts):,} docs...")
    tokenized = [tokenize_multilingual(t) for t in tqdm(corpus_texts, desc="Tokenizing")]
    return BM25Okapi(tokenized)


def bm25_retrieve(
    index: BM25Okapi,
    queries: list[str],
    citations: list[str],
    top_n: int,
) -> list[list[str]]:
    results = []
    for q in tqdm(queries, desc="BM25 retrieval"):
        tokens = tokenize_multilingual(q)
        scores = index.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_n]
        results.append([citations[i] for i in top_idx])
    return results


# ---------------------------------------------------------------------------
# Dense retrieval
# ---------------------------------------------------------------------------

def encode_corpus_chunked(
    model: SentenceTransformer,
    texts: list[str],
    passage_prefix: str,
    passage_prompt_name: str = "",
    batch_size: int = 256,
    chunk_size: int = 50_000,
) -> np.ndarray:
    all_embs = []
    for start in tqdm(range(0, len(texts), chunk_size), desc="Corpus chunks"):
        chunk = texts[start:start + chunk_size]
        if passage_prompt_name:
            embs = model.encode(chunk, prompt_name=passage_prompt_name,
                                batch_size=batch_size, normalize_embeddings=True,
                                show_progress_bar=False)
        else:
            embs = model.encode([passage_prefix + t for t in chunk],
                                batch_size=batch_size, normalize_embeddings=True,
                                show_progress_bar=False)
        all_embs.append(embs)
    return np.vstack(all_embs)


def dense_retrieve(
    q_embs: np.ndarray,
    c_embs: np.ndarray,
    citations: list[str],
    top_n: int,
) -> list[list[str]]:
    results = []
    batch = 64
    for i in range(0, len(q_embs), batch):
        scores = q_embs[i:i+batch] @ c_embs.T
        top_idx = np.argsort(scores, axis=1)[:, ::-1][:, :top_n]
        for row in top_idx:
            results.append([citations[j] for j in row])
    return results


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[str]:
    """Merge multiple ranked lists via RRF. k=60 is the standard constant."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


def rrf_merge_all(
    bm25_results: list[list[str]],
    dense_results: list[list[str]],
    top_n: int,
    k: int = 60,
) -> list[list[str]]:
    merged = []
    for bm25_row, dense_row in zip(bm25_results, dense_results):
        fused = reciprocal_rank_fusion([bm25_row, dense_row], k=k)
        merged.append(fused[:top_n])
    return merged


# ---------------------------------------------------------------------------
# Re-ranking (optional, same as 05_rerank.py)
# ---------------------------------------------------------------------------

def rerank_predictions(
    reranker: CrossEncoder,
    queries: list[str],
    candidates: list[list[str]],
    corpus_dict: dict[str, str],
    batch_size: int,
    score_threshold: float,
    min_k: int,
    max_k: int,
) -> list[list[str]]:
    results = []
    for query, cands in tqdm(zip(queries, candidates), total=len(queries),
                              desc="Re-ranking"):
        if not cands:
            results.append([])
            continue
        pairs = [(query, corpus_dict.get(cit, "")) for cit in cands]
        logits = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        sorted_pairs = sorted(zip(logits, cands), key=lambda x: x[0], reverse=True)
        forced = [cit for _, cit in sorted_pairs[:min_k]]
        additional = [cit for score, cit in sorted_pairs[min_k:max_k]
                      if score >= score_threshold]
        results.append(forced + additional)
    return results


# ---------------------------------------------------------------------------
# Val sweep helpers
# ---------------------------------------------------------------------------

def compute_macro_f1(all_pred: list[list[str]], all_gold: list[list[str]]) -> float:
    def f1(pred, gold):
        ps, gs = set(pred), set(gold)
        if not ps and not gs:
            return 1.0
        if not ps or not gs:
            return 0.0
        tp = len(ps & gs)
        p, r = tp / len(ps), tp / len(gs)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return float(np.mean([f1(p, g) for p, g in zip(all_pred, all_gold)]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bi-encoder-model", type=str, default="scenario1-original")
    parser.add_argument("--reranker-path", type=str,
                        default=cfg.rerank.reranker_model_path)
    parser.add_argument("--no-rerank", action="store_true",
                        help="Skip cross-encoder re-ranking (faster, lower precision)")
    parser.add_argument("--rerank-top-n", type=int, default=cfg.rerank.rerank_top_n,
                        help="Candidates fed to cross-encoder after RRF merge")
    parser.add_argument("--bm25-top-n", type=int, default=50,
                        help="BM25 candidates before RRF merge")
    parser.add_argument("--dense-top-n", type=int, default=50,
                        help="Dense candidates before RRF merge")
    parser.add_argument("--rrf-k", type=int, default=60,
                        help="RRF constant k (standard=60, rarely needs tuning)")
    parser.add_argument("--score-threshold", type=float,
                        default=cfg.rerank.reranker_score_threshold,
                        help="Reranker logit threshold (calibrate via --sweep-val)")
    parser.add_argument("--min-k", type=int, default=cfg.rerank.min_k)
    parser.add_argument("--max-k", type=int, default=cfg.rerank.max_k)
    parser.add_argument("--bm25-laws-only", action="store_true", default=True,
                        help="Build BM25 index over laws_de only (default). "
                             "Court 2.4M docs takes ~25min to index.")
    parser.add_argument("--no-court", action="store_true",
                        help="Skip court_considerations from dense corpus too")
    parser.add_argument("--model-type", choices=["e5-large", "gemma"], default="e5-large")
    parser.add_argument("--output", type=str, default="submission_hybrid.csv")
    parser.add_argument("--sweep-val", action="store_true",
                        help="Evaluate RRF vs dense-only on val.csv and exit")
    parser.add_argument("--log-file", type=str, default=None)
    args = parser.parse_args()

    mcfg = get_model_config(args.model_type)
    log_file = args.log_file or "finetune/logs/hybrid.txt"
    setup_logger(log_file)

    # Resolve bi-encoder path
    bi_path = args.bi_encoder_model
    if not Path(bi_path).exists():
        candidate = Path(cfg.train.models_root) / args.bi_encoder_model / "final"
        if candidate.exists():
            bi_path = str(candidate)
        else:
            print(f"ERROR: Bi-encoder not found at {bi_path} or {candidate}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Hybrid retrieval (BM25 + Dense → RRF → rerank)")
    print(f"  Bi-encoder:  {bi_path}")
    print(f"  Reranker:    {'skipped' if args.no_rerank else args.reranker_path}")
    print(f"  RRF k:       {args.rrf_k}")
    print(f"  Output:      {args.output}")
    print(f"{'='*60}\n")

    # --- Load corpus ---
    print("[1/6] Loading corpus...")
    all_parts = []
    bm25_parts = []

    if os.path.exists(cfg.data.laws_csv):
        df = pd.read_csv(cfg.data.laws_csv)[["citation", "text"]].dropna()
        all_parts.append(df)
        bm25_parts.append(df)
        print(f"  laws_de: {len(df):,} docs")

    if not args.no_court and os.path.exists(cfg.data.court_csv):
        df = pd.read_csv(cfg.data.court_csv)[["citation", "text"]].dropna()
        all_parts.append(df)
        if not args.bm25_laws_only:
            bm25_parts.append(df)
            print(f"  court (BM25 + dense): {len(df):,} docs")
        else:
            print(f"  court (dense only): {len(df):,} docs")

    corpus_df = pd.concat(all_parts, ignore_index=True)
    bm25_df = pd.concat(bm25_parts, ignore_index=True)
    corpus_citations = corpus_df["citation"].tolist()
    corpus_texts = corpus_df["text"].tolist()
    bm25_citations = bm25_df["citation"].tolist()
    bm25_texts = bm25_df["text"].tolist()
    corpus_dict = dict(zip(corpus_citations, corpus_texts))
    print(f"  Dense corpus: {len(corpus_citations):,}  BM25 corpus: {len(bm25_citations):,}")

    # --- Build BM25 index ---
    print("\n[2/6] Building BM25 index...")
    bm25_index = build_bm25_index(bm25_texts)

    # --- Load bi-encoder ---
    print(f"\n[3/6] Loading bi-encoder: {bi_path}")
    bi_model = SentenceTransformer(bi_path)
    bi_model.max_seq_length = 128

    # --- Load queries ---
    print("\n[4/6] Loading test.csv...")
    test_df = pd.read_csv("data/test.csv")
    test_queries = test_df["query"].tolist()
    test_ids = test_df["query_id"].tolist()
    print(f"  Test queries: {len(test_queries)}")

    # --- Encode ---
    if mcfg.query_prompt_name:
        q_embs = bi_model.encode(test_queries, prompt_name=mcfg.query_prompt_name,
                                  batch_size=256, normalize_embeddings=True,
                                  show_progress_bar=False)
    else:
        q_embs = bi_model.encode([mcfg.query_prefix + q for q in test_queries],
                                  batch_size=256, normalize_embeddings=True,
                                  show_progress_bar=False)

    print(f"\n[5/6] Encoding dense corpus ({len(corpus_texts):,} docs)...")
    c_embs = encode_corpus_chunked(
        model=bi_model,
        texts=corpus_texts,
        passage_prefix=mcfg.passage_prefix,
        passage_prompt_name=mcfg.passage_prompt_name,
        batch_size=256,
    )

    # --- Optional val sweep ---
    if args.sweep_val and os.path.exists(cfg.data.val_csv):
        val_df = pd.read_csv(cfg.data.val_csv)
        val_queries_sw = val_df["query"].tolist()
        gold_sw = []
        for _, row in val_df.iterrows():
            if pd.isna(row.get("gold_citations", None)):
                gold_sw.append([])
            else:
                gold_sw.append([c.strip() for c in str(row["gold_citations"]).split(";")])

        if mcfg.query_prompt_name:
            q_embs_val = bi_model.encode(val_queries_sw, prompt_name=mcfg.query_prompt_name,
                                          batch_size=256, normalize_embeddings=True,
                                          show_progress_bar=False)
        else:
            q_embs_val = bi_model.encode([mcfg.query_prefix + q for q in val_queries_sw],
                                          batch_size=256, normalize_embeddings=True,
                                          show_progress_bar=False)

        # Dense-only top-10
        dense_val = dense_retrieve(q_embs_val, c_embs, corpus_citations, top_n=10)
        f1_dense = compute_macro_f1(dense_val, gold_sw)

        # BM25-only top-10
        bm25_val = bm25_retrieve(bm25_index, val_queries_sw, bm25_citations, top_n=50)

        # RRF merged
        dense_val_50 = dense_retrieve(q_embs_val, c_embs, corpus_citations, top_n=50)
        rrf_val = rrf_merge_all(bm25_val, dense_val_50, top_n=10, k=args.rrf_k)
        f1_rrf = compute_macro_f1(rrf_val, gold_sw)

        print(f"\nVal comparison:")
        print(f"  Dense-only F1@10:  {f1_dense:.4f}")
        print(f"  RRF hybrid F1@10:  {f1_rrf:.4f}  (delta: {f1_rrf - f1_dense:+.4f})")

        if args.output == "submission_hybrid.csv":
            print("\nSweep complete. Re-run without --sweep-val to write submission.")
            return

    # --- Hybrid retrieve ---
    print("\n[6/6] Retrieving (BM25 + Dense → RRF)...")
    bm25_results = bm25_retrieve(bm25_index, test_queries, bm25_citations, top_n=args.bm25_top_n)
    dense_results = dense_retrieve(q_embs, c_embs, corpus_citations, top_n=args.dense_top_n)
    merged = rrf_merge_all(bm25_results, dense_results, top_n=args.rerank_top_n, k=args.rrf_k)

    # --- Optional re-rank ---
    if not args.no_rerank and os.path.exists(args.reranker_path):
        print(f"\n  Re-ranking top-{args.rerank_top_n} with cross-encoder...")
        reranker = CrossEncoder(args.reranker_path, max_length=512)
        predictions = rerank_predictions(
            reranker=reranker,
            queries=test_queries,
            candidates=merged,
            corpus_dict=corpus_dict,
            batch_size=cfg.rerank.reranker_batch_size,
            score_threshold=args.score_threshold,
            min_k=args.min_k,
            max_k=args.max_k,
        )
    else:
        if not args.no_rerank:
            print(f"  WARNING: Reranker not found at {args.reranker_path}, skipping.")
        predictions = [row[:args.max_k] for row in merged]

    # --- Write submission ---
    rows = [
        {"query_id": qid, "predicted_citations": ";".join(cits)}
        for qid, cits in zip(test_ids, predictions)
    ]
    sub_df = pd.DataFrame(rows)
    sub_df.to_csv(args.output, index=False)
    print(f"\nSubmission saved to: {args.output}")
    print(f"  Rows: {len(sub_df)}")
    print(sub_df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
