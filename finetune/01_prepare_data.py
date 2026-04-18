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

def load_corpus(laws_path: str, court_path: str) -> dict[str, str]:
    """Build citation → text lookup from corpus files. Pass empty string to skip a file."""
    parts = []
    for path in [laws_path, court_path]:
        if not path:
            continue
        if os.path.exists(path):
            df = pd.read_csv(path)
            parts.append(df[["citation", "text"]].dropna())
            print(f"  Loaded {len(df):,} rows from {path}")
        else:
            print(f"  WARNING: {path} not found, skipping")
    if not parts:
        return {}
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
) -> list[str]:
    """
    For each query, retrieve top-K corpus docs, filter true positives, return hard negatives.

    Args:
        hard_neg_margin: docs with similarity > top_score * margin are excluded (NVIDIA: 0.95)
        max_corpus_docs: cap corpus size for mining to avoid OOM / excessive time on CPU
    """
    device = get_device()
    print(f"  Device for mining: {device}")
    print(f"  Loading model for hard negative mining...")

    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = 128

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

    print("  Mining hard negatives...")
    # Build query → hard negatives map
    top_k = hard_neg_per_query * 10  # retrieve more, filter down
    query_to_hardnegs = {}

    # Score in row-chunks to avoid OOM on large query × corpus matrix
    chunk_size_q = 256  # score 256 queries at a time
    scores = np.empty((len(unique_queries), len(corpus_texts)), dtype=np.float32)
    for start in range(0, len(unique_queries), chunk_size_q):
        end = min(start + chunk_size_q, len(unique_queries))
        scores[start:end] = query_embs[start:end] @ corpus_embs.T

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
    corpus = load_corpus(cfg.data.laws_csv, cfg.data.court_csv)
    print(f"  Total corpus size: {len(corpus):,} citations")

    # --- 3. Expand pairs ---
    print("[3/5] Building (query, positive) pairs...")
    queries_orig, positives, missed = expand_pairs(train_df, corpus)
    total = len(queries_orig) + missed
    print(f"  Pairs built: {len(queries_orig):,} / {total:,} (missed {missed:,} citations not in corpus)")

    # --- Build smart mining corpus ---
    # Strategy (from user insight + data.md):
    #   1. All gold citations from train.csv  → pasti relevan, harus ada (~3-5k unik)
    #   2. All laws_de.csv                   → hanya 175k, manageable, semua pasal relevan
    #   3. court_considerations: data.md says "you can start without this" — skip or small sample
    # Total: ~180k vs 2.16M sebelumnya
    gold_texts = set(positives)  # teks dari gold citations yang sudah matched
    laws_corpus = load_corpus(cfg.data.laws_csv, "")  # laws saja
    mining_corpus = dict(laws_corpus)  # mulai dari semua laws (175k)
    # Pastikan semua gold citation texts masuk ke mining corpus
    for cit, text in corpus.items():
        if text in gold_texts:
            mining_corpus[cit] = text
    court_sample = args.court_sample
    if court_sample > 0 and os.path.exists(cfg.data.court_csv):
        court_df = pd.read_csv(cfg.data.court_csv, nrows=court_sample).dropna(subset=["citation", "text"])
        for _, row in court_df.iterrows():
            if row["citation"] not in mining_corpus:
                mining_corpus[row["citation"]] = row["text"]
        print(f"  Added {court_sample:,} court docs sample to mining corpus")
    print(f"  Smart mining corpus: {len(mining_corpus):,} docs "
          f"(gold citations + all laws + {court_sample:,} court sample)")

    # --- 4. Build query variants ---
    # Store raw text (no prefixes/prompts) — applied at training time
    print(f"[4/5] Building query variants (mode={query_mode})...")
    final_queries, final_positives = [], []

    if query_mode in ("original", "both"):
        final_queries.extend(queries_orig)
        final_positives.extend(positives)
        print(f"  Added {len(queries_orig):,} original-language pairs")

    if query_mode in ("translated", "both"):
        print("  Translating queries to English...")
        queries_en = translate_queries(
            queries_orig,
            cache_path=cfg.data.translation_cache,
            model_name=cfg.data.translation_model,
            batch_size=cfg.data.translation_batch_size,
        )
        final_queries.extend(queries_en)
        final_positives.extend(positives)
        print(f"  Added {len(queries_en):,} English-translated pairs")

    print(f"  Total pairs: {len(final_queries):,}")

    # --- 5. Mine hard negatives ---
    mine = cfg.data.mine_hard_negatives and not args.no_hard_negatives
    if mine:
        print("[5/5] Mining hard negatives...")
        negatives = mine_hard_negatives(
            queries=final_queries,
            positives=final_positives,
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
        )
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
