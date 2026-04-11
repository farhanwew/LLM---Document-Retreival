"""
MILCO Index: Learned Sparse Retrieval for Cross-Lingual Search.

Replaces BM25. Maps German/French/Italian documents to a shared English lexical space.
Search is done by dot product between sparse query vector and sparse document matrix.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pickle
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent))
import config
from embedder import Embedder


class MILCOIndex:
    def __init__(self, embedder: Embedder = None):
        self.embedder = embedder
        self.doc_matrix: sp.csr_matrix | None = None
        self.citations: list[str] = []
        self.texts: list[str] = []

    def build(self, corpus: pd.DataFrame, batch_size: int = config.ENCODE_BATCH_SIZE):
        """Encode entire corpus using MILCO and store as sparse matrix.
        
        This might take hours for 2M docs on a single GPU.
        """
        if self.embedder is None:
            self.embedder = Embedder()
            
        self.citations = corpus["citation"].tolist()
        self.texts = corpus["text"].tolist()
        
        print(f"[milco-index] encoding {len(self.texts):,} docs ...")
        
        # We can use the existing embedder's encode_documents function
        # which already handles batching and memory management.
        self.doc_matrix = self.embedder.encode_documents(self.texts)
        print(f"[milco-index] index built: {self.doc_matrix.shape}")

    def save(self, path: Path = config.MILCO_INDEX_PATH, meta_path: Path = config.MILCO_META_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        sp.save_npz(path, self.doc_matrix)
        with open(meta_path, "wb") as f:
            pickle.dump({"citations": self.citations, "texts": self.texts}, f)
        print(f"[milco-index] saved → {path}")

    def load(self, path: Path = config.MILCO_INDEX_PATH, meta_path: Path = config.MILCO_META_PATH):
        if not path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Index not found at {path}")
            
        print(f"[milco-index] loading from {path} ...")
        self.doc_matrix = sp.load_npz(path)
        with open(meta_path, "rb") as f:
            data = pickle.load(f)
        self.citations = data["citations"]
        self.texts = data["texts"]
        print(f"[milco-index] loaded {len(self.citations):,} docs")

    def search_batch(self, query_vectors: sp.csr_matrix, top_k: int = 100) -> list[list[dict]]:
        """Fast sparse dot product search for a batch of queries."""
        # Dot product: (n_queries, vocab) @ (vocab, n_docs) = (n_queries, n_docs)
        # Using .T for dot product
        scores_matrix = query_vectors @ self.doc_matrix.T
        
        results = []
        for i in range(scores_matrix.shape[0]):
            row = scores_matrix[i].toarray()[0]
            # Get top-k indices using argpartition (faster than sort for top-k)
            k = min(top_k, len(row))
            idx = np.argpartition(row, -k)[-k:]
            # Sort the k indices by score
            idx = idx[np.argsort(row[idx])[::-1]]
            
            query_results = [
                {
                    "citation": self.citations[j],
                    "text": self.texts[j],
                    "score": float(row[j])
                }
                for j in idx if row[j] > 0
            ]
            results.append(query_results)
            
        return results


if __name__ == "__main__":
    from data_utils import load_corpus, load_train, load_mini_corpus
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini", action="store_true", help="Build mini index for testing")
    args = parser.parse_args()
    
    embedder = Embedder()
    index = MILCOIndex(embedder)
    
    if args.mini:
        train = load_train()
        corpus = load_mini_corpus(train)
        index.build(corpus)
        index.save(config.MODELS_DIR / "milco_mini.npz", config.MODELS_DIR / "milco_mini_meta.pkl")
    else:
        corpus = load_corpus()
        index.build(corpus)
        index.save()
