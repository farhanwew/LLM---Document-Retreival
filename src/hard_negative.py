"""
Hard negative mining.

Requires:
  1. FAISS index already built (run: python src/indexer.py)

Produces data/hard_negative_dataset.json with anchor/positive/negative triplets.
"""
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import config
from data_utils import load_train, load_corpus, parse_citations, build_citation_to_text
from embedder import Embedder
from indexer import FaissIndexer


def mine_hard_negatives(
    train: pd.DataFrame,
    indexer: FaissIndexer,
    embedder: Embedder,
    citation_to_text: dict,
    top_k_candidates: int = config.HARD_NEG_TOP_K,
    n_hard_neg: int = config.HARD_NEG_PER_QUERY,
) -> dict:
    """
    For each training query:
      1. Retrieve top_k_candidates from FAISS
      2. Remove gold citations → remaining are hard negative candidates
      3. Keep top n_hard_neg
      4. Build triplets: (anchor_query, positive_text, hard_negative_text)

    Expects train queries to be in English (run translate.py first).
    """
    dataset: dict[str, list] = {"anchor": [], "positive": [], "negative": []}
    skipped = 0

    queries = train["query"].tolist()
    query_embs = embedder.encode_queries(queries)
    scores, indices = indexer.search(query_embs, top_k=top_k_candidates)
    retrieved_citations = indexer.get_citations(indices)

    for i, row in tqdm(train.iterrows(), total=len(train), desc="mining"):
        query = row["query"]
        gold_citations = parse_citations(row["gold_citations"])

        if not gold_citations:
            skipped += 1
            continue

        gold_set = set(gold_citations)
        hard_neg_candidates = [
            c for c in retrieved_citations[i] if c not in gold_set
        ][:n_hard_neg]

        if not hard_neg_candidates:
            skipped += 1
            continue

        for gold_cit in gold_citations:
            gold_text = citation_to_text.get(gold_cit)
            if not gold_text:
                continue
            for neg_cit in hard_neg_candidates:
                neg_text = citation_to_text.get(neg_cit)
                if not neg_text:
                    continue
                dataset["anchor"].append(query)
                dataset["positive"].append(gold_text)
                dataset["negative"].append(neg_text)

    print(f"\n[hard_neg] triplets   : {len(dataset['anchor'])}")
    print(f"[hard_neg] skipped    : {skipped}")
    return dataset


def save_dataset(dataset: dict, path: Path = config.HARD_NEG_DATASET_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(dataset, f)
    print(f"[hard_neg] saved → {path}")


def load_dataset(path: Path = config.HARD_NEG_DATASET_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    train = load_train()
    corpus = load_corpus()
    citation_to_text = build_citation_to_text(corpus)

    embedder = Embedder()
    indexer = FaissIndexer()
    indexer.load()

    dataset = mine_hard_negatives(train, indexer, embedder, citation_to_text)
    save_dataset(dataset)
    print("Hard negative mining done. Run trainer.py next.")
