"""
Stage 1: Data Preparation for Swiss Legal Embedding Fine-tuning

Usage:
    python finetune/01_prepare_data.py --query-mode original
    python finetune/01_prepare_data.py --query-mode translated
    python finetune/01_prepare_data.py --query-mode both

What it does:
1. Load train.csv + corpus (laws_de.csv + court_considerations.csv)
2. Build (query, positive_doc_text) pairs from gold citations
3. Translate queries to English if needed (cached)
4. Mine hard negatives from corpus using base embedding model
5. Save HuggingFace Dataset to prepared_data/{query_mode}/

Sources:
- ShawhinT: dataset format (anchor, positive, negative) columns
- NVIDIA NeMo: hard neg mining concept, hard_neg_margin=0.95, corpus_chunk_size=50000
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

from finetune.config import cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_corpus(laws_path: str, court_path: str) -> dict[str, str]:
    """Build citation → text lookup from both corpus files."""
    parts = []
    for path in [laws_path, court_path]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            parts.append(df[["citation", "text"]].dropna())
            print(f"  Loaded {len(df):,} rows from {path}")
        else:
            print(f"  WARNING: {path} not found, skipping")
    corpus_df = pd.concat(parts, ignore_index=True)
    return dict(zip(corpus_df["citation"], corpus_df["text"]))


def expand_pairs(train_df: pd.DataFrame, corpus: dict[str, str]) -> tuple[list, list, int]:
    """
    For each train row, split gold_citations by ';' and join with corpus text.
    Returns:
        queries: list of query strings
        positives: list of positive doc texts
        missed: count of citations not found in corpus
    """
    queries, positives = [], []
    missed = 0
    for _, row in train_df.iterrows():
        if pd.isna(row.get("gold_citations", None)):
            continue
        citations = [c.strip() for c in str(row["gold_citations"]).split(";")]
        for cit in citations:
            if cit in corpus:
                queries.append(str(row["query"]))
                positives.append(corpus[cit])
            else:
                missed += 1
    return queries, positives, missed


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def load_translation_model(model_name: str):
    print(f"  Loading translation model: {model_name}")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return tokenizer, model


def translate_batch(texts: list[str], tokenizer, model, batch_size: int = 32) -> list[str]:
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
        batch = texts[i: i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            translated = model.generate(**inputs)
        decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)
        results.extend(decoded)
    return results


def translate_queries(queries: list[str], cache_path: str,
                      model_name: str, batch_size: int) -> list[str]:
    """Translate queries with a persistent cache to avoid re-translating."""
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"  Loaded {len(cache):,} cached translations")

    unique_to_translate = [q for q in set(queries) if q not in cache]
    if unique_to_translate:
        print(f"  Translating {len(unique_to_translate):,} unique queries...")
        tokenizer, model = load_translation_model(model_name)
        translated = translate_batch(unique_to_translate, tokenizer, model, batch_size)
        for src, tgt in zip(unique_to_translate, translated):
            cache[src] = tgt
        os.makedirs(Path(cache_path).parent, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        print(f"  Cache saved to {cache_path}")

    return [cache[q] for q in queries]


# ---------------------------------------------------------------------------
# Hard negative mining (NVIDIA NeMo concept, custom implementation)
# ---------------------------------------------------------------------------

def mine_hard_negatives(
    queries: list[str],
    positives: list[str],
    corpus: dict[str, str],
    model_name: str,
    hard_neg_per_query: int,
    hard_neg_margin: float,
    mining_batch_size: int,
    corpus_chunk_size: int,
    query_prefix: str,
    passage_prefix: str,
) -> list[str]:
    """
    For each query, retrieve top-K corpus docs, filter true positives, return hard negatives.

    Args:
        hard_neg_margin: docs with similarity > top_score * margin are excluded (NVIDIA: 0.95)
    """
    print("  Loading model for hard negative mining...")
    model = SentenceTransformer(model_name)

    # Build set of positive texts per query for filtering
    pos_set_per_query = {}
    for q, p in zip(queries, positives):
        pos_set_per_query.setdefault(q, set()).add(p)

    # Get unique queries
    unique_queries = list(set(queries))
    prefixed_queries = [query_prefix + q for q in unique_queries]

    print(f"  Encoding {len(unique_queries):,} unique queries...")
    query_embs = model.encode(
        prefixed_queries,
        batch_size=mining_batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Encode corpus in chunks (NVIDIA: corpus_chunk_size=50000)
    corpus_citations = list(corpus.keys())
    corpus_texts = list(corpus.values())
    prefixed_texts = [passage_prefix + t for t in corpus_texts]

    print(f"  Encoding corpus ({len(corpus_texts):,} docs) in chunks of {corpus_chunk_size:,}...")
    corpus_embs = []
    for start in tqdm(range(0, len(prefixed_texts), corpus_chunk_size), desc="Corpus chunks"):
        chunk = prefixed_texts[start: start + corpus_chunk_size]
        emb = model.encode(
            chunk,
            batch_size=mining_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        corpus_embs.append(emb)
    corpus_embs = np.concatenate(corpus_embs, axis=0)

    print("  Mining hard negatives...")
    # Build query → hard negatives map
    top_k = hard_neg_per_query * 10  # retrieve more, filter down
    query_to_hardnegs = {}

    # Batch score all queries at once (may be large; use chunks if OOM)
    scores = query_embs @ corpus_embs.T  # (n_queries, n_corpus)

    for i, q in enumerate(unique_queries):
        row_scores = scores[i]
        top_indices = np.argsort(row_scores)[::-1][:top_k]
        pos_texts = pos_set_per_query.get(q, set())
        top_score = row_scores[top_indices[0]]

        hard_negs = []
        for idx in top_indices:
            doc_text = corpus_texts[idx]
            doc_score = row_scores[idx]
            # Skip true positives and near-duplicate scores (NVIDIA margin check)
            if doc_text in pos_texts:
                continue
            if doc_score > top_score * hard_neg_margin:
                continue
            hard_negs.append(doc_text)
            if len(hard_negs) >= hard_neg_per_query:
                break
        query_to_hardnegs[q] = hard_negs

    # Expand to per-pair negatives (one negative per pair, cycling if few)
    negatives = []
    for q, p in zip(queries, positives):
        negs = query_to_hardnegs.get(q, [])
        if negs:
            negatives.append(negs[0])
        else:
            negatives.append("")  # fallback; filtered later

    return negatives


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query-mode",
        choices=["original", "translated", "both"],
        default="both",
        help=(
            "original: use train queries as-is (German/French/Italian). "
            "translated: translate all to English. "
            "both: use original + translated (doubles training pairs)."
        ),
    )
    parser.add_argument("--no-hard-negatives", action="store_true",
                        help="Skip hard negative mining (faster, lower quality)")
    args = parser.parse_args()

    query_mode = args.query_mode
    out_dir = Path(cfg.data.prepared_data_root) / query_mode
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Preparing data — mode: {query_mode}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}\n")

    # --- 1. Load train.csv ---
    print("[1/5] Loading train.csv...")
    train_df = pd.read_csv(cfg.data.train_csv)
    print(f"  {len(train_df):,} training rows")

    # --- 2. Load corpus ---
    print("[2/5] Loading corpus...")
    corpus = load_corpus(cfg.data.laws_csv, cfg.data.court_csv)
    print(f"  Total corpus size: {len(corpus):,} citations")

    # --- 3. Expand pairs ---
    print("[3/5] Building (query, positive) pairs...")
    queries_orig, positives, missed = expand_pairs(train_df, corpus)
    total = len(queries_orig) + missed
    print(f"  Pairs built: {len(queries_orig):,} / {total:,} (missed {missed:,} citations not in corpus)")

    # --- 4. Build query variants ---
    print(f"[4/5] Building query variants (mode={query_mode})...")
    final_queries, final_positives = [], []

    if query_mode in ("original", "both"):
        prefixed_orig = [cfg.model.query_prefix + q for q in queries_orig]
        final_queries.extend(prefixed_orig)
        final_positives.extend([cfg.model.passage_prefix + p for p in positives])
        print(f"  Added {len(prefixed_orig):,} original-language pairs")

    if query_mode in ("translated", "both"):
        print("  Translating queries to English...")
        queries_en = translate_queries(
            queries_orig,
            cache_path=cfg.data.translation_cache,
            model_name=cfg.data.translation_model,
            batch_size=cfg.data.translation_batch_size,
        )
        prefixed_en = [cfg.model.query_prefix + q for q in queries_en]
        final_queries.extend(prefixed_en)
        final_positives.extend([cfg.model.passage_prefix + p for p in positives])
        print(f"  Added {len(prefixed_en):,} English-translated pairs")

    print(f"  Total pairs: {len(final_queries):,}")

    # --- 5. Mine hard negatives ---
    mine = cfg.data.mine_hard_negatives and not args.no_hard_negatives
    if mine:
        print("[5/5] Mining hard negatives...")
        # Strip prefixes for mining (model adds them internally via encode)
        raw_queries = [q[len(cfg.model.query_prefix):] for q in final_queries]
        raw_positives = [p[len(cfg.model.passage_prefix):] for p in final_positives]
        negatives = mine_hard_negatives(
            queries=raw_queries,
            positives=raw_positives,
            corpus=corpus,
            model_name=cfg.model.base_model,
            hard_neg_per_query=cfg.data.hard_neg_per_query,
            hard_neg_margin=cfg.data.hard_neg_margin,
            mining_batch_size=cfg.data.mining_batch_size,
            corpus_chunk_size=cfg.data.corpus_chunk_size,
            query_prefix=cfg.model.query_prefix,
            passage_prefix=cfg.model.passage_prefix,
        )
        negatives = [cfg.model.passage_prefix + n for n in negatives]
    else:
        print("[5/5] Skipping hard negative mining")
        negatives = [""] * len(final_queries)

    # Filter rows with empty negatives
    if mine:
        before = len(final_queries)
        valid = [(q, p, n) for q, p, n in zip(final_queries, final_positives, negatives) if n.strip()]
        final_queries, final_positives, negatives = zip(*valid) if valid else ([], [], [])
        final_queries, final_positives, negatives = list(final_queries), list(final_positives), list(negatives)
        print(f"  Kept {len(final_queries):,} / {before:,} pairs with valid hard negatives")

    # --- 6. Save dataset ---
    print("\nSaving dataset...")
    data = {"anchor": final_queries, "positive": final_positives}
    if mine and negatives:
        data["negative"] = negatives

    # 90/10 split for train/eval (val.csv used for final evaluation)
    n = len(final_queries)
    split = int(n * 0.9)
    indices = np.random.permutation(n)
    train_idx, eval_idx = indices[:split], indices[split:]

    def subset(d, idx):
        return {k: [v[i] for i in idx] for k, v in d.items()}

    ds = DatasetDict({
        "train": Dataset.from_dict(subset(data, train_idx)),
        "eval":  Dataset.from_dict(subset(data, eval_idx)),
    })
    ds.save_to_disk(str(out_dir))

    print(f"\nDone! Dataset saved to {out_dir}")
    print(f"  Train: {len(ds['train']):,} pairs")
    print(f"  Eval:  {len(ds['eval']):,} pairs")
    print(f"  Columns: {ds['train'].column_names}")
    print("\nSample pair:")
    print(f"  anchor:   {ds['train']['anchor'][0][:120]}")
    print(f"  positive: {ds['train']['positive'][0][:120]}")
    if "negative" in ds["train"].column_names:
        print(f"  negative: {ds['train']['negative'][0][:120]}")


if __name__ == "__main__":
    main()
