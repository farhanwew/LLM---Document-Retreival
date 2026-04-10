"""
BM25 first-pass retrieval over the full corpus.

Uses bm25s — sparse matrix BM25, handles millions of docs in seconds.
Build once (CPU, ~10-15 min), search in milliseconds.

Usage:
    python src/bm25_index.py
"""
import bm25s
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm
import config
from data_utils import load_corpus


class BM25Index:
    def __init__(self):
        self.retriever: bm25s.BM25 | None = None
        self.citations: list[str] = []
        self.texts: list[str] = []

    def build(self, corpus: pd.DataFrame):
        self.citations = corpus["citation"].tolist()
        self.texts = corpus["text"].tolist()

        print(f"[bm25] tokenizing {len(self.texts):,} docs ...")
        corpus_tokens = bm25s.tokenize(self.texts, show_progress=True)

        print("[bm25] building index ...")
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens, show_progress=True)
        print("[bm25] index ready")

    def save(self, path: Path = config.BM25_INDEX_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        # Save bm25s retriever + citations/texts separately
        self.retriever.save(str(path) + ".bm25s", corpus=self.texts)
        with open(str(path) + ".meta.pkl", "wb") as f:
            pickle.dump({"citations": self.citations}, f)
        print(f"[bm25] saved → {path}")

    def load(self, path: Path = config.BM25_INDEX_PATH):
        print(f"[bm25] loading from {path} ...")
        self.retriever = bm25s.BM25.load(str(path) + ".bm25s", load_corpus=True)
        self.texts = self.retriever.corpus
        with open(str(path) + ".meta.pkl", "rb") as f:
            data = pickle.load(f)
        self.citations = data["citations"]
        print(f"[bm25] loaded {len(self.citations):,} docs")

    def search(self, query: str, top_k: int = config.BM25_TOP_K) -> list[dict]:
        """Return top-k candidates as list of {citation, text, bm25_score}."""
        query_tokens = bm25s.tokenize([query], show_progress=False)
        results, scores = self.retriever.retrieve(query_tokens, k=min(top_k, len(self.citations)))
        # results[0] = doc indices for first query, scores[0] = scores
        return [
            {
                "citation": self.citations[idx],
                "text": self.texts[idx],
                "bm25_score": float(scores[0][i]),
            }
            for i, idx in enumerate(results[0])
        ]

    def search_batch(
        self, queries: list[str], top_k: int = config.BM25_TOP_K
    ) -> list[list[dict]]:
        """Batch search — faster than calling search() in a loop."""
        query_tokens = bm25s.tokenize(queries, show_progress=False)
        results, scores = self.retriever.retrieve(
            query_tokens, k=min(top_k, len(self.citations))
        )
        return [
            [
                {
                    "citation": self.citations[idx],
                    "text": self.texts[idx],
                    "bm25_score": float(scores[q][i]),
                }
                for i, idx in enumerate(results[q])
            ]
            for q in range(len(queries))
        ]


if __name__ == "__main__":
    corpus = load_corpus()
    index = BM25Index()
    index.build(corpus)
    index.save()
    print("BM25 index built and saved.")
