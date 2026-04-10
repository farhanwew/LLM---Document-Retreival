import numpy as np
import faiss
import pandas as pd
import config
from embedder import Embedder
from data_utils import load_corpus


class FaissIndexer:
    """
    Build, save, load, and search a FAISS flat inner-product index.
    Since embeddings are L2-normalized, inner product == cosine similarity.
    """

    def __init__(self):
        self.index = None
        self.citations: list[str] = []

    def build(self, corpus: pd.DataFrame, embedder: Embedder):
        """Embed corpus texts and build FAISS index."""
        print("[indexer] encoding corpus ...")
        texts = corpus["text"].tolist()
        self.citations = corpus["citation"].tolist()

        embeddings = embedder.encode_corpus(texts)

        dim = embeddings.shape[1]
        print(f"[indexer] building FAISS IndexFlatIP (dim={dim}) ...")
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        print(f"[indexer] index built: {self.index.ntotal} vectors")

    def save(self):
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(config.FAISS_INDEX_PATH))
        np.save(str(config.CORPUS_CITATIONS_PATH), np.array(self.citations))
        print(f"[indexer] saved → {config.FAISS_INDEX_PATH}")

    def load(self):
        print(f"[indexer] loading index from {config.FAISS_INDEX_PATH} ...")
        self.index = faiss.read_index(str(config.FAISS_INDEX_PATH))
        self.citations = np.load(
            str(config.CORPUS_CITATIONS_PATH), allow_pickle=True
        ).tolist()
        print(f"[indexer] loaded {self.index.ntotal} vectors")

    def search(
        self, query_embeddings: np.ndarray, top_k: int = config.RETRIEVAL_TOP_K
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            scores  : (n_queries, top_k) float32
            indices : (n_queries, top_k) int64
        """
        scores, indices = self.index.search(query_embeddings, top_k)
        return scores, indices

    def get_citations(self, indices: np.ndarray) -> list[list[str]]:
        """Convert index positions → citation strings."""
        return [
            [self.citations[i] for i in row if i >= 0]
            for row in indices
        ]


if __name__ == "__main__":
    corpus = load_corpus()
    embedder = Embedder()
    indexer = FaissIndexer()
    indexer.build(corpus, embedder)
    indexer.save()
    print("Index built and saved.")
