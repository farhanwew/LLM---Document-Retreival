import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import config


class Embedder:
    """
    Wrapper around SentenceTransformer for encoding queries and corpus texts.

    Queries are encoded with a task instruction prefix (Qwen3-Embedding best practice).
    Corpus documents are encoded without any prefix.
    Embeddings are always L2-normalized for cosine similarity via FAISS IndexFlatIP.
    """

    def __init__(self, model_name_or_path: str = config.BASE_MODEL):
        print(f"[embedder] loading model: {model_name_or_path}")
        self.model = SentenceTransformer(model_name_or_path)
        print("[embedder] model loaded")

    def encode(
        self,
        texts: list[str],
        batch_size: int = config.ENCODE_BATCH_SIZE,
        show_progress: bool = True,
        prompt: str | None = None,
    ) -> np.ndarray:
        """Encode texts → normalized float32 numpy array."""
        kwargs = dict(
            batch_size=batch_size,
            normalize_embeddings=config.NORMALIZE_EMBEDDINGS,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        if prompt is not None:
            kwargs["prompt"] = prompt
        embeddings = self.model.encode(texts, **kwargs)
        return embeddings.astype(np.float32)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        """Encode queries with task instruction prefix."""
        return self.encode(
            queries,
            batch_size=32,
            show_progress=False,
            prompt=config.QUERY_INSTRUCTION,
        )

    def encode_corpus(self, texts: list[str]) -> np.ndarray:
        """Encode corpus documents (no prefix, large batch, with progress bar)."""
        return self.encode(texts, batch_size=config.ENCODE_BATCH_SIZE, show_progress=True)

    def save(self, path: str | Path):
        self.model.save(str(path))
        print(f"[embedder] model saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "Embedder":
        return cls(model_name_or_path=str(path))
