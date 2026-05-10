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

from finetune.config import cfg, get_model_config, ModelConfig
from finetune.logger import setup_logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_corpus(path: str) -> dict[str, str]:
    """Build citation → text lookup from a CSV file (concatenated/clean or raw)."""
    if not path or not os.path.exists(path):
        print(f"  WARNING: {path} not found or empty, skipping")
        return {}
    
    df = pd.read_csv(path)
    # If using corpus_clean.csv, it has 'citation' and 'text'. 
    # If court_considerations.csv, it might have duplicates (handled by dictionary overwrite - NOT IDEAL).
    # We prefer the clean one.
    df = df[["citation", "text"]].dropna()
    print(f"  Loaded {len(df):,} rows from {path}")
    return dict(zip(df["citation"], df["text"]))


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

def get_device() -> str:
    """
    Detect usable device. Falls back to CPU if GPU compute capability is too low.
    Tesla P100 = sm_60, but recent PyTorch requires sm_70+.
    """
    if not torch.cuda.is_available():
        return "cpu"
    try:
        # Quick smoke test — if the GPU can't run kernels, this will raise
        _ = torch.zeros(1).cuda() + torch.zeros(1).cuda()
        return "cuda"
    except Exception as e:
        print(f"  WARNING: GPU unusable ({e}), falling back to CPU")
        return "cpu"


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
    query_prompt_name: str = "",
    passage_prompt_name: str = "",
    max_corpus_docs: int = 100_000,
    neg_similarity_threshold: float = 0.50,
    max_negs_per_pair: int = 0,
) -> list[str]:
    """
    For each query, retrieve top-K corpus docs, filter true positives, return hard negatives.

    Hard negatives are selected by cosine similarity threshold (not fixed count):
    only corpus docs with similarity >= neg_similarity_threshold qualify.

    Args:
        hard_neg_margin: (legacy, not used in threshold mode)
        neg_similarity_threshold: minimum cosine similarity to qualify as hard negative
        max_negs_per_pair: cap hard negatives per (query, positive) pair. 0 = no cap.
        max_corpus_docs: cap corpus size for mining to avoid OOM / excessive time on CPU
    """
    device = get_device()
    print(f"  Device for mining: {device}")
    print(f"  Loading model for hard negative mining...")

    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = 512

    # Adjust batch size for CPU (much smaller to avoid slowness spiral)
    encode_batch = mining_batch_size if device == "cuda" else 16

    # Build set of positive texts per query for filtering
    pos_set_per_query = {}
    for q, p in zip(queries, positives):
        pos_set_per_query.setdefault(q, set()).add(p)

    # Get unique queries
    unique_queries = list(set(queries))

    print(f"  Encoding {len(unique_queries):,} unique queries...")
    if query_prompt_name:
        query_embs = model.encode(
            unique_queries, prompt_name=query_prompt_name,
            batch_size=encode_batch, show_progress_bar=True, normalize_embeddings=True,
        )
    else:
        query_embs = model.encode(
            [query_prefix + q for q in unique_queries],
            batch_size=encode_batch, show_progress_bar=True, normalize_embeddings=True,
        )

    # Cap corpus size for mining — 2M+ docs is impractical on CPU
    # Prefer to keep docs that are actually gold citations (higher mining signal)
    corpus_citations = list(corpus.keys())
    corpus_texts = list(corpus.values())
    if len(corpus_citations) > max_corpus_docs:
        print(f"  Sampling corpus: {max_corpus_docs:,} / {len(corpus_citations):,} docs for mining")
        # Keep gold positive docs + random sample of the rest
        pos_cits = set()
        for q, p in zip(queries, positives):
            pos_cits.add(p)
        priority_idx = [i for i, t in enumerate(corpus_texts) if t in pos_cits]
        other_idx = [i for i in range(len(corpus_texts)) if i not in set(priority_idx)]
        np.random.shuffle(other_idx)
        keep_idx = priority_idx + other_idx[: max_corpus_docs - len(priority_idx)]
        corpus_citations = [corpus_citations[i] for i in keep_idx]
        corpus_texts = [corpus_texts[i] for i in keep_idx]
        print(f"  Mining corpus: {len(corpus_citations):,} docs "
              f"({len(priority_idx):,} gold positives + {len(keep_idx)-len(priority_idx):,} random)")

    print(f"  Encoding corpus ({len(corpus_texts):,} docs) in chunks of {corpus_chunk_size:,}...")
    corpus_embs = []
    for start in tqdm(range(0, len(corpus_texts), corpus_chunk_size), desc="Corpus chunks"):
        chunk = corpus_texts[start: start + corpus_chunk_size]
        if passage_prompt_name:
            emb = model.encode(
                chunk, prompt_name=passage_prompt_name,
                batch_size=encode_batch, show_progress_bar=False, normalize_embeddings=True,
            )
        else:
            emb = model.encode(
                [passage_prefix + t for t in chunk],
                batch_size=encode_batch, show_progress_bar=False, normalize_embeddings=True,
            )
        corpus_embs.append(emb)
    corpus_embs = np.concatenate(corpus_embs, axis=0)

    print(f"  Mining hard negatives (similarity >= {neg_similarity_threshold})...")
    # Score all queries against all corpus docs (full cosine sim matrix).
    # Threshold-based selection: keep all non-positive docs with similarity >= threshold.
    top_k = 500  # large enough pool to catch all above-threshold docs
    query_to_hardnegs = {}

    # Score in row-chunks to avoid OOM on large query × corpus matrix
    chunk_size_q = 256
    scores = np.empty((len(unique_queries), len(corpus_texts)), dtype=np.float32)
    for start in range(0, len(unique_queries), chunk_size_q):
        end = min(start + chunk_size_q, len(unique_queries))
        scores[start:end] = query_embs[start:end] @ corpus_embs.T

    min_negatives = 3  # always take at least this many regardless of threshold
    for i, q in enumerate(unique_queries):
        row_scores = scores[i]
        pos_texts = pos_set_per_query.get(q, set())

        # Sort descending and take top_k (wide enough to catch threshold qualifiers)
        top_indices = np.argsort(row_scores)[::-1][:top_k]

        # Scheme: minimum {min_negatives} always, plus all extra docs above threshold.
        # Step 1: gather top {min_negatives} non-positive docs (guaranteed minimum).
        # Step 2: continue scanning while similarity >= threshold to grab extras.
        hard_negs = []
        for idx in top_indices:
            doc_text = corpus_texts[idx]
            if doc_text in pos_texts:
                continue
            hard_negs.append(doc_text)
            if len(hard_negs) >= min_negatives and row_scores[idx] < neg_similarity_threshold:
                break  # minimum met and now below threshold → stop
        query_to_hardnegs[q] = hard_negs

        n_above = int((row_scores >= neg_similarity_threshold).sum())
        if n_above > top_k:
            print(f"  WARNING: {q[:60]}... has {n_above} docs >= {neg_similarity_threshold} "
                  f"(only scanned top {top_k}). Increase top_k if missing candidates.")

    # Expand rows: each hard negative becomes its own (anchor, positive, negative) row.
    expanded_queries, expanded_positives, expanded_negatives = [], [], []
    for q, p in zip(queries, positives):
        negs = query_to_hardnegs.get(q, [])
        if max_negs_per_pair and max_negs_per_pair > 0:
            negs = negs[:max_negs_per_pair]
        if negs:
            for neg in negs:
                expanded_queries.append(q)
                expanded_positives.append(p)
                expanded_negatives.append(neg)
        else:
            expanded_queries.append(q)
            expanded_positives.append(p)
            expanded_negatives.append("")  # filtered later

    return expanded_queries, expanded_positives, expanded_negatives


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
    parser.add_argument(
        "--model", choices=["e5-large", "gemma"], default="e5-large",
        help="Embedding model to use. Determines output dir and encoding strategy.",
    )
    parser.add_argument("--no-hard-negatives", action="store_true",
                        help="Skip hard negative mining (faster, lower quality)")
    parser.add_argument(
        "--max-corpus-for-mining", type=int, default=100_000,
        help=(
            "Max corpus docs to use for hard neg mining. "
            "Full corpus (2M+) is impractical on CPU. "
            "Default 100k: all gold positives + random sample. "
            "Use --max-corpus-for-mining 0 to use full corpus (slow!)."
        ),
    )
    parser.add_argument(
        "--court-sample", type=int, default=0,
        help=(
            "How many court_considerations docs to add to mining corpus (default: 0). "
            "data.md says you can skip court entirely to start. "
            "laws_de (175k) + gold citations already gives good hard negatives."
        ),
    )
    parser.add_argument("--log-file", type=str, default=None,
                        help="Save all output to this file (default: finetune/logs/prepare_{model}_{mode}.txt)")
    args = parser.parse_args()

    query_mode = args.query_mode
    model_type = args.model
    mcfg = get_model_config(model_type)

    log_file = args.log_file or f"finetune/logs/prepare_{model_type}_{query_mode}.txt"
    setup_logger(log_file)

    # Separate dirs per model so e5 and gemma data don't overwrite each other
    out_dir = Path(cfg.data.prepared_data_root) / model_type / query_mode
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Preparing data — mode: {query_mode}  model: {model_type}")
    print(f"  Base model: {mcfg.base_model}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}\n")

    # --- 1. Load train.csv ---
    print("[1/5] Loading train.csv...")
    train_df = pd.read_csv(cfg.data.train_csv)
    print(f"  {len(train_df):,} training rows")

    # --- 2. Load corpus ---
    print("[2/5] Loading corpus...")
    # Preferred: concatenated/clean corpus from preprocess.ipynb
    if os.path.exists(cfg.data.corpus_clean_csv):
        print(f"  Using clean/concatenated corpus: {cfg.data.corpus_clean_csv}")
        corpus = load_corpus(cfg.data.corpus_clean_csv)
    else:
        print(f"  WARNING: Clean corpus not found at {cfg.data.corpus_clean_csv}")
        print("  Falling back to raw laws + court (may have dictionary overwrites for chunks)")
        corpus_laws = load_corpus(cfg.data.laws_csv)
        corpus_court = load_corpus(cfg.data.court_csv)
        corpus = {**corpus_laws, **corpus_court}
    print(f"  Total corpus size: {len(corpus):,} unique citations")

    # --- 3. Expand pairs ---
    print("[3/5] Building (query, positive) pairs...")
    queries_orig, positives, missed = expand_pairs(train_df, corpus)
    total = len(queries_orig) + missed
    print(f"  Pairs built: {len(queries_orig):,} / {total:,} (missed {missed:,} citations not in corpus)")

    # --- 4. Build query variants (with optimization mapping) ---
    print(f"[4/5] Building query variants (mode={query_mode})...")
    
    # query_map: original_query -> translated_query (None if not translating)
    query_map = {q: None for q in set(queries_orig)}
    
    if query_mode in ("translated", "both"):
        print("  Translating queries to English...")
        unique_orig = list(query_map.keys())
        unique_en = translate_queries(
            unique_orig,
            cache_path=cfg.data.translation_cache,
            model_name=cfg.data.translation_model,
            batch_size=cfg.data.translation_batch_size,
        )
        for orig, en in zip(unique_orig, unique_en):
            query_map[orig] = en

    # final_training_pairs: list of (semantic_anchor, positive_text, original_query_for_mining)
    # We always mine based on the original query (German) and apply results to translations.
    # This ensures consistency and saves half the mining compute in 'both' mode.
    training_rows = []
    for q_orig, p in zip(queries_orig, positives):
        if query_mode in ("original", "both"):
            training_rows.append({"anchor": q_orig, "positive": p, "mine_key": q_orig})
        if query_mode in ("translated", "both"):
            training_rows.append({"anchor": query_map[q_orig], "positive": p, "mine_key": q_orig})
    
    print(f"  Total training rows to be expanded: {len(training_rows):,}")

    # --- 5. Mine hard negatives ---
    mine = cfg.data.mine_hard_negatives and not args.no_hard_negatives
    if mine:
        print("[5/5] Mining hard negatives...")
        
        # Build mining corpus logic (Smart Mining)
        # Use clean laws + gold citations as base for efficiency
        gold_texts = set(positives)
        mining_corpus = {}
        # 1. Add all laws (usually high quality distractors)
        if os.path.exists(cfg.data.laws_csv):
            mining_corpus.update(load_corpus(cfg.data.laws_csv))
        # 2. Add all gold positives from training
        for cit, text in corpus.items():
            if text in gold_texts:
                mining_corpus[cit] = text

        # 3. Add random court docs if requested
        if args.court_sample > 0 and os.path.exists(cfg.data.court_csv):
            court_df = pd.read_csv(cfg.data.court_csv, usecols=["citation", "text"]).dropna()
            court_df = court_df[~court_df["citation"].isin(mining_corpus)]
            if len(court_df) > args.court_sample:
                court_df = court_df.sample(n=args.court_sample, random_state=42)
            for _, row in court_df.iterrows():
                mining_corpus[row["citation"]] = row["text"]
            print(f"  Added {len(court_df):,} random court docs (--court-sample {args.court_sample})")

        print(f"  Smart mining corpus: {len(mining_corpus):,} docs (gold citations + laws + court sample)")

        # Unique queries to mine (the original ones)
        queries_to_mine = list(query_map.keys())
        # We need a dummy positive for each unique query to mine
        # (just pick one, used to filter self-positives)
        dummy_positives = []
        q_to_all_pos = {}
        for q, p in zip(queries_orig, positives):
            q_to_all_pos.setdefault(q, []).append(p)
        for q in queries_to_mine:
            dummy_positives.append(q_to_all_pos[q][0])

        mined_queries, _, mined_negatives_list = mine_hard_negatives(
            queries=queries_to_mine,
            positives=dummy_positives,
            corpus=mining_corpus,
            model_name=mcfg.base_model,
            hard_neg_per_query=cfg.data.hard_neg_per_query,
            hard_neg_margin=cfg.data.hard_neg_margin,
            mining_batch_size=cfg.data.mining_batch_size,
            corpus_chunk_size=cfg.data.corpus_chunk_size,
            query_prefix=mcfg.query_prefix,
            passage_prefix=mcfg.passage_prefix,
            query_prompt_name=mcfg.query_prompt_name,
            passage_prompt_name=mcfg.passage_prompt_name,
            max_corpus_docs=len(mining_corpus),
            neg_similarity_threshold=cfg.data.neg_similarity_threshold,
            max_negs_per_pair=cfg.data.max_negs_per_pair,
        )

        # Map unique queries back to their mined negatives
        # Since mine_hard_negatives returns expanded lists, we need to re-group
        # Actually, let's modify mine_hard_negatives to return a DICT q -> list[neg] 
        # or just handle the expanded output.
        # WAIT: mine_hard_negatives expands rows (anchor, pos, neg). 
        # Let's check its return values again.
        
        # NOTE: mine_hard_negatives returns (expanded_q, expanded_p, expanded_n)
        # We need to map q -> list of negatives.
        q_to_negs = {}
        for q, n in zip(mined_queries, mined_negatives_list):
            if n.strip():
                q_to_negs.setdefault(q, []).append(n)
        
        # Expand final dataset
        final_queries, final_positives, final_negatives = [], [], []
        row_cap = cfg.data.max_negs_per_pair if cfg.data.max_negs_per_pair > 0 else None
        for row in training_rows:
            negs = q_to_negs.get(row["mine_key"], [])
            if row_cap:
                negs = negs[:row_cap] 
            if negs:
                for n in negs:
                    final_queries.append(row["anchor"])
                    final_positives.append(row["positive"])
                    final_negatives.append(n)
            else:
                final_queries.append(row["anchor"])
                final_positives.append(row["positive"])
                final_negatives.append("")
        
        print(f"  Final expanded dataset: {len(final_queries):,} rows")
    else:
        print("[5/5] Skipping hard negative mining")
        final_queries = [row["anchor"] for row in training_rows]
        final_positives = [row["positive"] for row in training_rows]
        final_negatives = [""] * len(final_queries)

    # Filter rows with empty negatives
    if mine:
        before = len(final_queries)
        valid = [(q, p, n) for q, p, n in zip(final_queries, final_positives, final_negatives) if n.strip()]
        final_queries, final_positives, negatives = zip(*valid) if valid else ([], [], [])
        final_queries, final_positives, negatives = list(final_queries), list(final_positives), list(negatives)
        print(f"  Kept {len(final_queries):,} / {before:,} expanded rows with valid hard negatives")

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
