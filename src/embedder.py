import numpy as np
import torch
import scipy.sparse as sp
from tqdm import tqdm
from transformers import AutoModel
import config


class Embedder:
    """
    MILCO-based sparse embedder for cross-lingual retrieval.

    encode_queries()  → sparse CSR matrix (n_queries, vocab_size)
    encode_corpus()   → sparse CSR matrix (n_docs, vocab_size)

    Scoring: query_sparse @ doc_sparse.T  (sparse dot product)
    """

    def __init__(self, model_name_or_path: str = config.BASE_MODEL):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[embedder] loading MILCO: {model_name_or_path} on {device}")
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device
        print("[embedder] ready")

    def encode_queries(self, queries: list[str]) -> sp.csr_matrix:
        """Encode queries → sparse CSR matrix (n_queries, vocab_size)."""
        with torch.no_grad():
            sparse_out = self.model.encode_query(queries)
        return self._to_scipy(sparse_out)

    def encode_corpus(
        self,
        texts: list[str],
        batch_size: int = config.ENCODE_BATCH_SIZE,
    ) -> sp.csr_matrix:
        """Encode corpus documents → sparse CSR matrix (n_docs, vocab_size).

        Processes in batches and builds the full sparse matrix in one shot
        to avoid slow repeated vstack calls.
        """
        all_rows, all_cols, all_vals = [], [], []
        vocab_size = None
        row_offset = 0

        for i in tqdm(range(0, len(texts), batch_size), desc="encoding corpus"):
            batch = texts[i : i + batch_size]
            with torch.no_grad():
                sparse_out = self.model.encode_document(batch)

            sparse_out = sparse_out.coalesce().cpu()
            torch.cuda.empty_cache()
            indices = sparse_out.indices().numpy()  # (2, nnz)
            values = sparse_out.values().numpy()    # (nnz,)

            all_rows.append(indices[0] + row_offset)
            all_cols.append(indices[1])
            all_vals.append(values)

            vocab_size = sparse_out.shape[1]
            row_offset += len(batch)

        rows = np.concatenate(all_rows)
        cols = np.concatenate(all_cols)
        vals = np.concatenate(all_vals)

        return sp.csr_matrix(
            (vals, (rows, cols)),
            shape=(len(texts), vocab_size),
            dtype=np.float32,
        )

    def _to_scipy(self, sparse_tensor: torch.Tensor) -> sp.csr_matrix:
        """Convert a 2D torch sparse tensor → scipy CSR matrix."""
        sparse_tensor = sparse_tensor.coalesce().cpu()
        indices = sparse_tensor.indices().numpy()
        values = sparse_tensor.values().numpy()
        shape = tuple(sparse_tensor.shape)
        return sp.csr_matrix(
            (values, (indices[0], indices[1])),
            shape=shape,
            dtype=np.float32,
        )

    def save(self, path: str):
        self.model.save_pretrained(str(path))
        print(f"[embedder] model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "Embedder":
        return cls(model_name_or_path=str(path))
