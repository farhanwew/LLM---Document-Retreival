import numpy as np
import torch
import scipy.sparse as sp
from transformers import AutoModel
import config


class Embedder:
    """
    MILCO embedder — used only at query time to rerank BM25 candidates.

    No full corpus pre-encoding needed. encode_documents() handles
    small batches of BM25 candidates (~500 docs per query).
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
        """Encode queries → sparse CSR (n_queries, vocab_size)."""
        with torch.no_grad():
            sparse_out = self.model.encode_query(queries)
        return self._to_scipy(sparse_out)

    def encode_documents(self, texts: list[str]) -> sp.csr_matrix:
        """Encode a small batch of documents → sparse CSR (n_docs, vocab_size).

        Used for reranking BM25 candidates (~500 docs), not full corpus.
        """
        all_rows, all_cols, all_vals = [], [], []
        vocab_size = None
        row_offset = 0
        batch_size = config.ENCODE_BATCH_SIZE

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            with torch.no_grad():
                sparse_out = self.model.encode_document(batch)
            torch.cuda.empty_cache()

            sparse_out = sparse_out.coalesce().cpu()
            indices = sparse_out.indices().numpy()
            values = sparse_out.values().numpy()

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
        sparse_tensor = sparse_tensor.coalesce().cpu()
        indices = sparse_tensor.indices().numpy()
        values = sparse_tensor.values().numpy()
        shape = tuple(sparse_tensor.shape)
        return sp.csr_matrix(
            (values, (indices[0], indices[1])),
            shape=shape,
            dtype=np.float32,
        )
