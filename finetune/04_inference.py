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

from finetune.config import cfg, get_model_config
from finetune.logger import setup_logger


def encode_corpus_chunked(
    model: SentenceTransformer,
    citations: list[str],
    texts: list[str],
    passage_prefix: str,
    batch_size: int = 256,
    chunk_size: int = 50_000,
    passage_prompt_name: str = "",
) -> np.ndarray:
    """Encode corpus in chunks to manage memory. Returns (n_docs, dim) array."""
    all_embs = []
    for start in tqdm(range(0, len(texts), chunk_size), desc="Corpus chunks"):
        chunk = texts[start:start + chunk_size]
        if passage_prompt_name:
            embs = model.encode(chunk, prompt_name=passage_prompt_name,
                                batch_size=batch_size, show_progress_bar=True,
                                normalize_embeddings=True)
        else:
            embs = model.encode([passage_prefix + t for t in chunk],
                                batch_size=batch_size, show_progress_bar=True,
                                normalize_embeddings=True)
        all_embs.append(embs)
    return np.vstack(all_embs)


def retrieve_top_k(
    q_embs: np.ndarray,
    c_embs: np.ndarray,
    citations: list[str],
    top_k: int,
) -> list[list[str]]:
    """Dot product similarity, return fixed top-K citation lists per query."""
    results = []
    batch = 64
    for i in range(0, len(q_embs), batch):
        scores = q_embs[i:i+batch] @ c_embs.T  # (batch, n_corpus)
        top_idx = np.argsort(scores, axis=1)[:, ::-1][:, :top_k]
        for row in top_idx:
            results.append([citations[j] for j in row])
    return results


def retrieve_variable_k(
    q_embs: np.ndarray,
    c_embs: np.ndarray,
    citations: list[str],
    score_threshold: float,
    min_k: int = 1,
    max_k: int = 20,
) -> list[list[str]]:
    """
    Variable-K retrieval: returns all citations above score_threshold per query.
    Always returns at least min_k (prevents empty predictions which destroy F1).
    Caps at max_k to avoid excessive false positives.
    Calibrate score_threshold on val.csv before submitting (sweep 0.70–0.90).
    Note: assumes normalized embeddings (cosine = dot product).
    """
    results = []
    batch = 64
    for i in range(0, len(q_embs), batch):
        scores = q_embs[i:i+batch] @ c_embs.T  # (batch, n_corpus)
        for row_scores in scores:
            sorted_idx = np.argsort(row_scores)[::-1]
            forced = [citations[j] for j in sorted_idx[:min_k]]
            additional = [
                citations[j] for j in sorted_idx[min_k:max_k]
                if row_scores[j] >= score_threshold
            ]
            results.append(forced + additional)
    return results


def threshold_sweep(
    q_embs: np.ndarray,
    c_embs: np.ndarray,
    citations: list[str],
    gold_all: list[list[str]],
    thresholds: list[float],
    min_k: int = 1,
    max_k: int = 20,
) -> None:
    """Sweep thresholds on val set and print Macro F1 for each. Use before final submission."""
    import numpy as np

    def compute_f1(pred, gold):
        ps, gs = set(pred), set(gold)
        if not ps and not gs:
            return 1.0
        if not ps or not gs:
            return 0.0
        tp = len(ps & gs)
        p, r = tp / len(ps), tp / len(gs)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    scores_matrix = q_embs @ c_embs.T
    print("\nThreshold sweep on val:")
    print(f"  {'threshold':>10}  {'macro_f1':>10}  {'avg_cits':>10}")
    for t in thresholds:
        preds, lengths = [], []
        for row_scores in scores_matrix:
            sorted_idx = np.argsort(row_scores)[::-1]
            forced = [citations[j] for j in sorted_idx[:min_k]]
            additional = [
                citations[j] for j in sorted_idx[min_k:max_k]
                if row_scores[j] >= t
            ]
            p = forced + additional
            preds.append(p)
            lengths.append(len(p))
        f1 = float(np.mean([compute_f1(p, g) for p, g in zip(preds, gold_all)]))
        print(f"  {t:>10.2f}  {f1:>10.4f}  {np.mean(lengths):>10.1f}")


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
    parser.add_argument(
        "--model-type", choices=["e5-large", "gemma"], default="e5-large",
        help="Embedding model type — controls prefix/prompt strategy.",
    )
    parser.add_argument("--log-file", type=str, default=None,
                        help="Save all output to this file (default: finetune/logs/inference_{model}.txt)")
    parser.add_argument("--score-threshold", type=float, default=None,
                        help="Cosine similarity threshold for variable-K retrieval. "
                             "If set, overrides --top-k. Calibrate on val.csv (sweep 0.70–0.90). "
                             "Default: None (use fixed --top-k)")
    parser.add_argument("--min-k", type=int, default=cfg.inference.min_k,
                        help="Min citations per query for variable-K (default: 1)")
    parser.add_argument("--max-k", type=int, default=cfg.inference.max_k,
                        help="Max citations per query for variable-K (default: 20)")
    parser.add_argument("--sweep-val", action="store_true",
                        help="Before writing submission, sweep thresholds on val.csv to find best score_threshold")
    args = parser.parse_args()
    mcfg = get_model_config(args.model_type)
    log_file = args.log_file or f"finetune/logs/inference_{args.model}.txt"
    setup_logger(log_file)

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
    if mcfg.query_prompt_name:
        q_embs = model.encode(test_queries, prompt_name=mcfg.query_prompt_name,
                               batch_size=args.batch_size, show_progress_bar=False,
                               normalize_embeddings=True)
    else:
        q_embs = model.encode([mcfg.query_prefix + q for q in test_queries],
                               batch_size=args.batch_size, show_progress_bar=False,
                               normalize_embeddings=True)

    print(f"  Encoding corpus ({len(corpus_texts):,} docs)...")
    c_embs = encode_corpus_chunked(
        model=model,
        citations=corpus_citations,
        texts=corpus_texts,
        passage_prefix=mcfg.passage_prefix,
        passage_prompt_name=mcfg.passage_prompt_name,
        batch_size=args.batch_size,
        chunk_size=50_000,
    )

    # --- Optional: sweep thresholds on val.csv before writing submission ---
    if args.sweep_val and os.path.exists(cfg.data.val_csv):
        print("\nSweeping thresholds on val.csv...")
        val_df = pd.read_csv(cfg.data.val_csv)
        val_queries_sw = val_df["query"].tolist()
        gold_sw = []
        for _, row in val_df.iterrows():
            if pd.isna(row.get("gold_citations", None)):
                gold_sw.append([])
            else:
                gold_sw.append([c.strip() for c in str(row["gold_citations"]).split(";")])

        if mcfg.query_prompt_name:
            q_embs_val = model.encode(val_queries_sw, prompt_name=mcfg.query_prompt_name,
                                      batch_size=args.batch_size, normalize_embeddings=True,
                                      show_progress_bar=False)
        else:
            q_embs_val = model.encode([mcfg.query_prefix + q for q in val_queries_sw],
                                      batch_size=args.batch_size, normalize_embeddings=True,
                                      show_progress_bar=False)
        threshold_sweep(
            q_embs=q_embs_val,
            c_embs=c_embs,
            citations=corpus_citations,
            gold_all=gold_sw,
            thresholds=[round(t, 2) for t in np.arange(0.60, 0.92, 0.02)],
            min_k=args.min_k,
            max_k=args.max_k,
        )

    # --- Retrieve ---
    if args.score_threshold is not None:
        print(f"\n  Retrieving with score_threshold={args.score_threshold} "
              f"(min_k={args.min_k}, max_k={args.max_k})...")
        predictions = retrieve_variable_k(
            q_embs, c_embs, corpus_citations,
            score_threshold=args.score_threshold,
            min_k=args.min_k,
            max_k=args.max_k,
        )
    else:
        print(f"\n  Retrieving fixed top-{args.top_k} per query...")
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
