"""
Stage 3: Evaluate and compare all fine-tuned models on val.csv

Usage:
    # Compare all 3 scenarios vs base model
    python finetune/03_evaluate.py

    # Compare specific models
    python finetune/03_evaluate.py --models scenario1-original scenario2-translated scenario3-both

Output:
    Comparison table with Macro F1 (competition metric) + NDCG@10 + Recall@10
    for base model and each fine-tuned scenario side by side.

Metric: Macro F1 matches the competition evaluation exactly.
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


# ---------------------------------------------------------------------------
# Metric: Macro F1 (competition metric)
# ---------------------------------------------------------------------------

def compute_f1(predicted: list[str], gold: list[str]) -> float:
    pred_set = set(predicted)
    gold_set = set(gold)
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def macro_f1(all_predicted: list[list[str]], all_gold: list[list[str]]) -> float:
    return float(np.mean([compute_f1(p, g) for p, g in zip(all_predicted, all_gold)]))


def recall_at_k(all_predicted: list[list[str]], all_gold: list[list[str]], k: int) -> float:
    scores = []
    for pred, gold in zip(all_predicted, all_gold):
        gold_set = set(gold)
        if not gold_set:
            continue
        hit = len(set(pred[:k]) & gold_set)
        scores.append(hit / len(gold_set))
    return float(np.mean(scores)) if scores else 0.0


def ndcg_at_k(all_predicted: list[list[str]], all_gold: list[list[str]], k: int) -> float:
    scores = []
    for pred, gold in zip(all_predicted, all_gold):
        gold_set = set(gold)
        if not gold_set:
            continue
        dcg = sum(
            1 / np.log2(rank + 2)
            for rank, doc in enumerate(pred[:k])
            if doc in gold_set
        )
        ideal_dcg = sum(1 / np.log2(rank + 2) for rank in range(min(len(gold_set), k)))
        scores.append(dcg / ideal_dcg if ideal_dcg > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def load_corpus(laws_path: str, court_path: str, court_sample: int = 0) -> tuple[list[str], list[str]]:
    """Returns (citations, texts) lists.
    court_sample=0 skips court docs entirely (fast); set >0 to include a random sample.
    """
    parts = []
    if os.path.exists(laws_path):
        df = pd.read_csv(laws_path)
        parts.append(df[["citation", "text"]].dropna())
    if court_sample > 0 and os.path.exists(court_path):
        df = pd.read_csv(court_path)
        df = df[["citation", "text"]].dropna()
        if len(df) > court_sample:
            df = df.sample(n=court_sample, random_state=42)
        parts.append(df)
    corpus_df = pd.concat(parts, ignore_index=True)
    return corpus_df["citation"].tolist(), corpus_df["text"].tolist()


def retrieve_for_model(
    model: SentenceTransformer,
    val_queries: list[str],
    corpus_citations: list[str],
    corpus_texts: list[str],
    query_prefix: str,
    passage_prefix: str,
    top_k: int = 50,
    batch_size: int = 256,
    multi_gpu: bool = False,
    query_prompt_name: str = "",
    passage_prompt_name: str = "",
) -> list[list[str]]:
    """Encode queries + corpus, return top-K citation lists per query."""
    print(f"    Encoding {len(val_queries)} queries...")
    if query_prompt_name:
        q_embs = model.encode(val_queries, prompt_name=query_prompt_name,
                               batch_size=batch_size, show_progress_bar=False,
                               normalize_embeddings=True)
    else:
        q_embs = model.encode([query_prefix + q for q in val_queries],
                               batch_size=batch_size, show_progress_bar=False,
                               normalize_embeddings=True)

    prefixed_corpus = (
        corpus_texts if passage_prompt_name
        else [passage_prefix + t for t in corpus_texts]
    )

    print(f"    Encoding corpus ({len(corpus_texts):,} docs)...")
    encode_kwargs = dict(batch_size=batch_size, normalize_embeddings=True)
    if passage_prompt_name:
        encode_kwargs["prompt_name"] = passage_prompt_name
        corpus_input = corpus_texts
    else:
        corpus_input = prefixed_corpus

    if multi_gpu:
        import torch
        n_gpu = torch.cuda.device_count()
        devices = [f"cuda:{i}" for i in range(n_gpu)]
        pool = model.start_multi_process_pool(devices)
        c_embs = model.encode_multi_process(corpus_input, pool, batch_size=batch_size)
        model.stop_multi_process_pool(pool)
        c_embs = c_embs / np.linalg.norm(c_embs, axis=1, keepdims=True)
    else:
        c_embs = model.encode(corpus_input, show_progress_bar=True, **encode_kwargs)

    scores = q_embs @ c_embs.T  # (n_queries, n_corpus)
    top_indices = np.argsort(scores, axis=1)[:, ::-1][:, :top_k]

    return [[corpus_citations[i] for i in row] for row in top_indices]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Model run names under finetune/models/. Defaults to all 3 scenarios."
    )
    parser.add_argument("--top-k", type=int, default=20,
                        help="Retrieve this many docs per query for F1 computation")
    parser.add_argument("--court-sample", type=int, default=0,
                        help="Include N random court docs in corpus (default 0 = laws_de only, ~175k docs)")
    parser.add_argument("--multi-gpu", action="store_true",
                        help="Use all available GPUs for corpus encoding")
    parser.add_argument(
        "--model-type", choices=["e5-large", "gemma"], default="e5-large",
        help="Embedding model type — controls prefix/prompt strategy.",
    )
    parser.add_argument("--log-file", type=str, default="finetune/logs/evaluate.txt",
                        help="Save all output to this file")
    args = parser.parse_args()

    setup_logger(args.log_file)
    mcfg = get_model_config(args.model_type)
    model_names = args.models or [
        "scenario1-original",
        "scenario2-translated",
        "scenario3-both",
    ]
    models_root = Path(cfg.train.models_root)

    print(f"\n{'='*60}")
    print("  Evaluation on val.csv")
    print(f"  Top-K: {args.top_k}")
    print(f"{'='*60}\n")

    # --- Load val.csv ---
    val_df = pd.read_csv(cfg.data.val_csv)
    val_queries = val_df["query"].tolist()
    gold_all = []
    for _, row in val_df.iterrows():
        if pd.isna(row.get("gold_citations", None)):
            gold_all.append([])
        else:
            gold_all.append([c.strip() for c in str(row["gold_citations"]).split(";")])
    print(f"Val queries: {len(val_queries)}")

    # --- Load corpus ---
    print(f"Loading corpus (laws_de + {args.court_sample:,} court samples)...")
    corpus_citations, corpus_texts = load_corpus(cfg.data.laws_csv, cfg.data.court_csv, args.court_sample)

    # Always add gold citations from val.csv to corpus — they may come from court docs
    # not included in laws_de. Without this, F1=0 even if retrieval works correctly.
    citation_to_text_all = {}
    for path in [cfg.data.laws_csv, cfg.data.court_csv]:
        if os.path.exists(path):
            df = pd.read_csv(path)[["citation", "text"]].dropna()
            citation_to_text_all.update(dict(zip(df["citation"], df["text"])))

    existing = set(corpus_citations)
    for gold_list in gold_all:
        for cit in gold_list:
            if cit not in existing and cit in citation_to_text_all:
                corpus_citations.append(cit)
                corpus_texts.append(citation_to_text_all[cit])
                existing.add(cit)

    print(f"Corpus: {len(corpus_citations):,} docs (laws_de + gold citations from val.csv)\n")

    # --- Evaluate each model ---
    results = {}

    # Always include base model for comparison
    model_configs = [("base (no finetune)", mcfg.base_model)] + [
        (name, str(models_root / name / "final"))
        for name in model_names
    ]

    for label, model_path in model_configs:
        if label != "base (no finetune)" and not Path(model_path).exists():
            print(f"  SKIP {label}: model not found at {model_path}")
            continue

        print(f"Evaluating: {label}")
        model = SentenceTransformer(model_path)
        model.max_seq_length = 128  # match training setting; 93.9% docs fit fully

        retrieved = retrieve_for_model(
            model=model,
            val_queries=val_queries,
            corpus_citations=corpus_citations,
            corpus_texts=corpus_texts,
            query_prefix=mcfg.query_prefix,
            passage_prefix=mcfg.passage_prefix,
            query_prompt_name=mcfg.query_prompt_name,
            passage_prompt_name=mcfg.passage_prompt_name,
            top_k=args.top_k,
            batch_size=256,
            multi_gpu=args.multi_gpu,
        )

        results[label] = {
            "F1@5":      macro_f1([r[:5]  for r in retrieved], gold_all),
            "F1@10":     macro_f1([r[:10] for r in retrieved], gold_all),
            "F1@20":     macro_f1([r[:20] for r in retrieved], gold_all),
            "NDCG@10":   ndcg_at_k(retrieved, gold_all, k=10),
            "Recall@10": recall_at_k(retrieved, gold_all, k=10),
        }
        print(f"  F1@10={results[label]['F1@10']:.4f}  NDCG@10={results[label]['NDCG@10']:.4f}\n")

    # --- Print comparison table ---
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    cols = ["F1@5", "F1@10", "F1@20", "NDCG@10", "Recall@10"]
    col_w = 10
    name_w = 30

    header = f"{'Model':<{name_w}}" + "".join(f"{c:>{col_w}}" for c in cols)
    print(header)
    print("-" * len(header))

    for label, metrics in results.items():
        row = f"{label:<{name_w}}" + "".join(f"{metrics[c]:>{col_w}.4f}" for c in cols)
        print(row)

    print("="*80)

    # Highlight best scenario
    best = max(
        ((k, v) for k, v in results.items() if k != "base (no finetune)"),
        key=lambda x: x[1]["F1@10"],
        default=(None, None),
    )
    if best[0]:
        print(f"\nBest scenario by F1@10: {best[0]} ({best[1]['F1@10']:.4f})")
        base_f1 = results.get("base (no finetune)", {}).get("F1@10", 0)
        delta = best[1]["F1@10"] - base_f1
        print(f"Improvement over base: +{delta:.4f}")


if __name__ == "__main__":
    main()
