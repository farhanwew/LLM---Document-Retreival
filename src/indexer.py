import numpy as np
import scipy.sparse as sp
import pandas as pd
import config
from embedder import Embedder
from data_utils import load_corpus


class SparseIndexer:
    """
    Build, save, load, and search a scipy CSR sparse index from MILCO embeddings.

    Index shape: (n_docs, vocab_size) — typically ~500 MB for 1M docs.
    Search: query_sparse @ index.T  →  (n_queries, n_docs) scores.
    """

    def __init__(self):
        self.index: sp.csr_matrix | None = None
        self.citations: list[str] = []

    def build(self, corpus: pd.DataFrame, embedder: Embedder):
        print("[indexer] encoding corpus ...")
        texts = corpus["text"].tolist()
        self.citations = corpus["citation"].tolist()

        self.index = embedder.encode_corpus(texts)
        print(
            f"[indexer] index built: {self.index.shape[0]:,} docs, "
            f"vocab={self.index.shape[1]:,}, "
            f"nnz={self.index.nnz:,}"
        )

    def save(self):
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        sp.save_npz(str(config.SPARSE_INDEX_PATH), self.index)
        np.save(str(config.CORPUS_CITATIONS_PATH), np.array(self.citations))
        print(f"[indexer] saved → {config.SPARSE_INDEX_PATH}")

    def load(self):
        print(f"[indexer] loading from {config.SPARSE_INDEX_PATH} ...")
        self.index = sp.load_npz(str(config.SPARSE_INDEX_PATH))
        self.citations = np.load(
            str(config.CORPUS_CITATIONS_PATH), allow_pickle=True
        ).tolist()
        print(f"[indexer] loaded {self.index.shape[0]:,} docs")

    def search(
        self, query_sparse: sp.csr_matrix, top_k: int = config.RETRIEVAL_TOP_K
    ) -> tuple[list[list[str]], np.ndarray]:
        """
        Score all docs against each query and return top-k.

        Returns:
            citations : (n_queries,) list of top-k citation strings
            scores    : (n_queries, top_k) float32 scores
        """
        # (n_queries, n_docs) — sparse dot product
        scores_matrix = (query_sparse @ self.index.T).toarray().astype(np.float32)

        top_k_actual = min(top_k, scores_matrix.shape[1])
        # argpartition: top-k (unordered), then sort
        top_idx = np.argpartition(scores_matrix, -top_k_actual, axis=1)[:, -top_k_actual:]

        all_citations, all_scores = [], []
        for i, idx_row in enumerate(top_idx):
            order = np.argsort(scores_matrix[i, idx_row])[::-1]
            sorted_idx = idx_row[order]
            all_citations.append([self.citations[j] for j in sorted_idx])
            all_scores.append(scores_matrix[i, sorted_idx])

        return all_citations, np.array(all_scores)


if __name__ == "__main__":
    corpus = load_corpus()
    embedder = Embedder()
    indexer = SparseIndexer()
    indexer.build(corpus, embedder)
    indexer.save()
    print("Sparse index built and saved.")
