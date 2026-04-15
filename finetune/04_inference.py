"""
Stage 4: Generate submission.csv for test.csv

Usage:
    python finetune/04_inference.py --model scenario1-original
    python finetune/04_inference.py --model scenario1-original --top-k 10
    python finetune/04_inference.py --model scenario1-original --top-k 10 --no-court

Output: submission.csv (query_id, predicted_citations)

Notes:
- Corpus is encoded in chunks to manage memory (laws_de ~175k + court ~2.4M)
- Top-K citations per query are predicted
- For offline Kaggle: upload fine-tuned model as Kaggle dataset first
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from finetune.config import cfg


def encode_corpus_chunked(
    model: SentenceTransformer,
    citations: list[str],
    texts: list[str],
    passage_prefix: str,
    batch_size: int = 256,
    chunk_size: int = 50_000,
) -> np.ndarray:
    """Encode corpus in chunks to manage memory. Returns (n_docs, dim) array."""
    all_embs = []
    for start in tqdm(range(0, len(texts), chunk_size), desc="Corpus chunks"):
        chunk_texts = [passage_prefix + t for t in texts[start:start + chunk_size]]
        embs = model.encode(
            chunk_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        all_embs.append(embs)
    return np.vstack(all_embs)


def retrieve_top_k(
    q_embs: np.ndarray,
    c_embs: np.ndarray,
    citations: list[str],
    top_k: int,
) -> list[list[str]]:
    """Dot product similarity, return top-K citation lists per query."""
    # Process in batches to avoid memory spike on large corpus
    results = []
    batch = 64
    for i in range(0, len(q_embs), batch):
        scores = q_embs[i:i+batch] @ c_embs.T  # (batch, n_corpus)
        top_idx = np.argsort(scores, axis=1)[:, ::-1][:, :top_k]
        for row in top_idx:
            results.append([citations[j] for j in row])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="scenario1-original",
                        help="Model run name under finetune/models/ (or full path)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of citations to predict per query")
    parser.add_argument("--no-court", action="store_true",
                        help="Skip court_considerations.csv (faster but lower recall)")
    parser.add_argument("--output", type=str, default="submission.csv",
                        help="Output path for submission.csv")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Encoding batch size")
    args = parser.parse_args()

    # Resolve model path
    model_path = args.model
    if not Path(model_path).exists():
        candidate = Path(cfg.train.models_root) / args.model / "final"
        if candidate.exists():
            model_path = str(candidate)
        else:
            print(f"ERROR: Model not found at {model_path} or {candidate}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Inference for submission")
    print(f"  Model:  {model_path}")
    print(f"  Top-K:  {args.top_k}")
    print(f"  Court:  {'no' if args.no_court else 'yes'}")
    print(f"  Output: {args.output}")
    print(f"{'='*60}\n")

    # --- Load test queries ---
    print("[1/4] Loading test.csv...")
    test_df = pd.read_csv("data/test.csv")
    test_queries = test_df["query"].tolist()
    test_ids = test_df["query_id"].tolist()
    print(f"  Test queries: {len(test_queries)}")

    # --- Load corpus ---
    print("\n[2/4] Loading corpus...")
    parts = []
    if os.path.exists(cfg.data.laws_csv):
        df = pd.read_csv(cfg.data.laws_csv)[["citation", "text"]].dropna()
        parts.append(df)
        print(f"  laws_de: {len(df):,} docs")
    if not args.no_court and os.path.exists(cfg.data.court_csv):
        df = pd.read_csv(cfg.data.court_csv)[["citation", "text"]].dropna()
        parts.append(df)
        print(f"  court_considerations: {len(df):,} docs")
    elif args.no_court:
        print("  court_considerations: skipped (--no-court)")

    corpus_df = pd.concat(parts, ignore_index=True)
    corpus_citations = corpus_df["citation"].tolist()
    corpus_texts = corpus_df["text"].tolist()
    print(f"  Total corpus: {len(corpus_citations):,} docs")

    # --- Load model ---
    print(f"\n[3/4] Loading model: {model_path}")
    model = SentenceTransformer(model_path)
    model.max_seq_length = 128
    print(f"  max_seq_length: {model.max_seq_length}")

    # --- Encode ---
    print(f"\n[4/4] Encoding...")
    print(f"  Encoding {len(test_queries)} queries...")
    prefixed_queries = [cfg.model.query_prefix + q for q in test_queries]
    q_embs = model.encode(
        prefixed_queries,
        batch_size=args.batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    print(f"  Encoding corpus ({len(corpus_texts):,} docs)...")
    c_embs = encode_corpus_chunked(
        model=model,
        citations=corpus_citations,
        texts=corpus_texts,
        passage_prefix=cfg.model.passage_prefix,
        batch_size=args.batch_size,
        chunk_size=50_000,
    )

    # --- Retrieve ---
    print(f"\n  Retrieving top-{args.top_k} per query...")
    predictions = retrieve_top_k(q_embs, c_embs, corpus_citations, args.top_k)

    # --- Write submission ---
    rows = []
    for qid, cits in zip(test_ids, predictions):
        rows.append({
            "query_id": qid,
            "predicted_citations": ";".join(cits),
        })

    sub_df = pd.DataFrame(rows)
    sub_df.to_csv(args.output, index=False)
    print(f"\nSubmission saved to: {args.output}")
    print(f"  Rows: {len(sub_df)}")
    print(f"\nSample:")
    print(sub_df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
