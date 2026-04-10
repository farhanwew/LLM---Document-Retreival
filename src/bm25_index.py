"""
BM25 first-pass retrieval over the full corpus.

Build once (CPU, ~15-20 min), then search in milliseconds.
Output: top-500 candidates per query → fed into MILCO reranker.

Usage:
    python src/bm25_index.py
"""
import re
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from pathlib import Path
import config
from data_utils import load_corpus


class BM25Index:
    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.citations: list[str] = []
        self.texts: list[str] = []

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Lowercase + split on non-alphanumeric. Filters empty tokens."""
        return [t for t in re.split(r"\W+", text.lower()) if t]

    def build(self, corpus: pd.DataFrame):
        self.citations = corpus["citation"].tolist()
        self.texts = corpus["text"].tolist()

        print(f"[bm25] tokenizing {len(self.texts):,} docs ...")
        tokenized = [
            self.tokenize(t)
            for t in tqdm(self.texts, desc="tokenizing")
        ]
        print("[bm25] building BM25Okapi index ...")
        self.bm25 = BM25Okapi(tokenized)
        print("[bm25] index ready")

    def save(self, path: Path = config.BM25_INDEX_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"bm25": self.bm25, "citations": self.citations, "texts": self.texts},
                f,
            )
        print(f"[bm25] saved → {path}")

    def load(self, path: Path = config.BM25_INDEX_PATH):
        print(f"[bm25] loading from {path} ...")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.citations = data["citations"]
        self.texts = data["texts"]
        print(f"[bm25] loaded {len(self.citations):,} docs")

    def search(self, query: str, top_k: int = config.BM25_TOP_K) -> list[dict]:
        """Return top-k candidates as list of {citation, text, bm25_score}."""
        tokens = self.tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [
            {
                "citation": self.citations[i],
                "text": self.texts[i],
                "bm25_score": float(scores[i]),
            }
            for i in top_idx
        ]

    def search_batch(
        self, queries: list[str], top_k: int = config.BM25_TOP_K
    ) -> list[list[dict]]:
        return [self.search(q, top_k=top_k) for q in tqdm(queries, desc="bm25 search")]


if __name__ == "__main__":
    corpus = load_corpus()
    index = BM25Index()
    index.build(corpus)
    index.save()
    print("BM25 index built and saved.")
