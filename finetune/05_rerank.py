"""
Stage 5: Re-rank bi-encoder top-N results with BGE cross-encoder.

Usage:
    # Sweep thresholds on val first to calibrate:
    python finetune/05_rerank.py --sweep-val

    # Generate reranked submission:
    python finetune/05_rerank.py \
        --bi-encoder-model scenario1-both \
        --reranker-path finetune/models/bge-reranker-v2-m3 \
        --rerank-top-n 50 \
        --score-threshold 0.0 \
        --output submission_reranked.csv

Notes:
- BGE reranker returns raw logits, NOT probabilities. score_threshold is a logit
  value (typical range -10 to +10). Always calibrate on val.csv first.
- Use max_length=512 for the cross-encoder (legal texts are long; unlike bi-encoder
  where 128 was OK, cross-encoder context matters for re-ranking accuracy).
- Model: BAAI/bge-reranker-v2-m3 (~560MB fp16, multilingual, Apache 2.0)
  Download: huggingface-cli download BAAI/bge-reranker-v2-m3 \
            --local-dir finetune/models/bge-reranker-v2-m3
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm

from finetune.config import cfg, get_model_config
from finetune.logger import setup_logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_corpus_chunked(
    model: SentenceTransformer,
    citations: list[str],
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


def retrieve_top_n(
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
    """
    Re-rank bi-encoder candidates with cross-encoder.
    Returns variable-length citation lists (above score_threshold, capped at max_k).
    score_threshold is a raw logit — calibrate on val.csv first.
    """
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
        additional = [
            cit for score, cit in sorted_pairs[min_k:max_k]
            if score >= score_threshold
        ]
        results.append(forced + additional)
    return results


def threshold_sweep_rerank(
    reranker: CrossEncoder,
    val_queries: list[str],
    candidates: list[list[str]],
    gold_all: list[list[str]],
    corpus_dict: dict[str, str],
    batch_size: int,
    thresholds: list[float],
    min_k: int,
    max_k: int,
) -> None:
    """Sweep reranker score thresholds on val set."""
    all_logits_per_query = []
    all_cands_per_query = []

    for query, cands in zip(val_queries, candidates):
        if not cands:
            all_logits_per_query.append([])
            all_cands_per_query.append([])
            continue
        pairs = [(query, corpus_dict.get(cit, "")) for cit in cands]
        logits = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        sorted_pairs = sorted(zip(logits, cands), key=lambda x: x[0], reverse=True)
        all_logits_per_query.append([s for s, _ in sorted_pairs])
        all_cands_per_query.append([c for _, c in sorted_pairs])

    def compute_f1(pred, gold):
        ps, gs = set(pred), set(gold)
        if not ps and not gs:
            return 1.0
        if not ps or not gs:
            return 0.0
        tp = len(ps & gs)
        p, r = tp / len(ps), tp / len(gs)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    print("\nReranker threshold sweep on val:")
    print(f"  {'threshold':>12}  {'macro_f1':>10}  {'avg_cits':>10}")
    for t in thresholds:
        preds, lengths = [], []
        for logits_sorted, cands_sorted in zip(all_logits_per_query, all_cands_per_query):
            forced = cands_sorted[:min_k]
            additional = [c for s, c in zip(logits_sorted[min_k:max_k],
                                             cands_sorted[min_k:max_k]) if s >= t]
            p = forced + additional
            preds.append(p)
            lengths.append(len(p))
        f1 = float(np.mean([compute_f1(p, g) for p, g in zip(preds, gold_all)]))
        print(f"  {t:>12.2f}  {f1:>10.4f}  {np.mean(lengths):>10.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bi-encoder-model", type=str, default="scenario1-original",
                        help="Bi-encoder run name under finetune/models/ (or full path)")
    parser.add_argument("--reranker-path", type=str,
                        default=cfg.rerank.reranker_model_path,
                        help="Path to cross-encoder model (BAAI/bge-reranker-v2-m3)")
    parser.add_argument("--rerank-top-n", type=int, default=cfg.rerank.rerank_top_n,
                        help="Bi-encoder candidates to feed to cross-encoder (default 50)")
    parser.add_argument("--score-threshold", type=float,
                        default=cfg.rerank.reranker_score_threshold,
                        help="Cross-encoder logit threshold (NOT probability). "
                             "Calibrate via --sweep-val first.")
    parser.add_argument("--min-k", type=int, default=cfg.rerank.min_k)
    parser.add_argument("--max-k", type=int, default=cfg.rerank.max_k)
    parser.add_argument("--no-court", action="store_true",
                        help="Skip court_considerations.csv (faster)")
    parser.add_argument("--output", type=str, default="submission_reranked.csv")
    parser.add_argument("--batch-size", type=int, default=cfg.rerank.reranker_batch_size)
    parser.add_argument("--model-type", choices=["e5-large", "gemma"], default="e5-large")
    parser.add_argument("--sweep-val", action="store_true",
                        help="Sweep reranker thresholds on val.csv and exit")
    parser.add_argument("--log-file", type=str, default=None)
    args = parser.parse_args()

    mcfg = get_model_config(args.model_type)
    log_file = args.log_file or f"finetune/logs/rerank_{Path(args.bi_encoder_model).name}.txt"
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
    print(f"  Re-ranking inference")
    print(f"  Bi-encoder: {bi_path}")
    print(f"  Reranker:   {args.reranker_path}")
    print(f"  Top-N:      {args.rerank_top_n}  →  threshold: {args.score_threshold}")
    print(f"  Output:     {args.output}")
    print(f"{'='*60}\n")

    # --- Load corpus ---
    print("[1/5] Loading corpus...")
    parts = []
    if os.path.exists(cfg.data.laws_csv):
        df = pd.read_csv(cfg.data.laws_csv)[["citation", "text"]].dropna()
        parts.append(df)
        print(f"  laws_de: {len(df):,} docs")
    if not args.no_court and os.path.exists(cfg.data.court_csv):
        df = pd.read_csv(cfg.data.court_csv)[["citation", "text"]].dropna()
        parts.append(df)
        print(f"  court_considerations: {len(df):,} docs")

    corpus_df = pd.concat(parts, ignore_index=True)
    corpus_citations = corpus_df["citation"].tolist()
    corpus_texts = corpus_df["text"].tolist()
    corpus_dict = dict(zip(corpus_citations, corpus_texts))
    print(f"  Total corpus: {len(corpus_citations):,} docs")

    # --- Load bi-encoder + encode ---
    print(f"\n[2/5] Loading bi-encoder: {bi_path}")
    bi_model = SentenceTransformer(bi_path)
    bi_model.max_seq_length = 128

    # --- Load test queries ---
    print("\n[3/5] Loading test.csv...")
    test_df = pd.read_csv("data/test.csv")
    test_queries = test_df["query"].tolist()
    test_ids = test_df["query_id"].tolist()
    print(f"  Test queries: {len(test_queries)}")

    # Encode queries
    if mcfg.query_prompt_name:
        q_embs = bi_model.encode(test_queries, prompt_name=mcfg.query_prompt_name,
                                  batch_size=256, normalize_embeddings=True,
                                  show_progress_bar=False)
    else:
        q_embs = bi_model.encode([mcfg.query_prefix + q for q in test_queries],
                                  batch_size=256, normalize_embeddings=True,
                                  show_progress_bar=False)

    # Encode corpus
    print(f"\n[4/5] Encoding corpus ({len(corpus_texts):,} docs)...")
    c_embs = encode_corpus_chunked(
        model=bi_model,
        citations=corpus_citations,
        texts=corpus_texts,
        passage_prefix=mcfg.passage_prefix,
        passage_prompt_name=mcfg.passage_prompt_name,
        batch_size=256,
    )

    # Retrieve top-N candidates
    print(f"  Retrieving top-{args.rerank_top_n} candidates per query...")
    candidates = retrieve_top_n(q_embs, c_embs, corpus_citations, args.rerank_top_n)

    # --- Load cross-encoder ---
    print(f"\n[5/5] Loading reranker: {args.reranker_path}")
    reranker = CrossEncoder(args.reranker_path, max_length=512)

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

        val_candidates = retrieve_top_n(q_embs_val, c_embs, corpus_citations, args.rerank_top_n)
        threshold_sweep_rerank(
            reranker=reranker,
            val_queries=val_queries_sw,
            candidates=val_candidates,
            gold_all=gold_sw,
            corpus_dict=corpus_dict,
            batch_size=args.batch_size,
            thresholds=[round(t, 1) for t in np.arange(-6.0, 6.0, 0.5)],
            min_k=args.min_k,
            max_k=args.max_k,
        )
        if args.output == "submission_reranked.csv":
            print("\nSweep complete. Re-run with --score-threshold <value> to write submission.")
            return

    # --- Re-rank ---
    print(f"\n  Re-ranking with threshold={args.score_threshold}...")
    predictions = rerank_predictions(
        reranker=reranker,
        queries=test_queries,
        candidates=candidates,
        corpus_dict=corpus_dict,
        batch_size=args.batch_size,
        score_threshold=args.score_threshold,
        min_k=args.min_k,
        max_k=args.max_k,
    )

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
