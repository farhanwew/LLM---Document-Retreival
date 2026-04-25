"""
Fine-tuning configuration for Swiss legal embedding model.
Adjust paths and flags before running each stage.
"""
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DataConfig:
    # --- Input data paths (download from Kaggle first) ---
    train_csv: str = "data/train.csv"
    val_csv: str = "data/val.csv"
    laws_csv: str = "data/laws_de.csv"
    court_csv: str = "data/court_considerations.csv"
    corpus_clean_csv: str = "data/corpus_clean.csv"

    # --- Output root ---
    prepared_data_root: str = "finetune/prepared_data"

    # --- Translation ---
    # Helsinki-NLP/opus-mt-mul-en handles DE/FR/IT → EN, offline-capable (~300MB)
    translation_model: str = "Helsinki-NLP/opus-mt-mul-en"
    translation_cache: str = "finetune/translation_cache.json"
    translation_batch_size: int = 32

    # --- Hard negative mining (NVIDIA NeMo concept) ---
    mine_hard_negatives: bool = True
    hard_neg_per_query: int = 4        # 1 pos + 4 neg = 5 passages per query (NVIDIA: train_n_passages=5)
    hard_neg_margin: float = 0.95      # From NVIDIA config
    mining_batch_size: int = 512  # no backprop during mining → can use large batch
    corpus_chunk_size: int = 100_000    # From NVIDIA config


@dataclass
class ModelConfig:
    base_model: str = "intfloat/multilingual-e5-large"

    # Prefix-based models (e.g. multilingual-e5): prepend to text before encoding
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "

    # Prompt-name models (e.g. EmbeddingGemma): use model.encode(prompt_name=...)
    # If non-empty, prompt_name takes priority over prefix
    query_prompt_name: str = ""
    passage_prompt_name: str = ""

    max_seq_length: int = 128


# --- Available model configs ---
MODEL_CONFIGS: dict[str, ModelConfig] = {
    "e5-large": ModelConfig(
        base_model="intfloat/multilingual-e5-large",
        query_prefix="query: ",
        passage_prefix="passage: ",
        max_seq_length=128,
    ),
    "gemma": ModelConfig(
        base_model="google/embeddinggemma-300m",
        query_prefix="",
        passage_prefix="",
        query_prompt_name="Retrieval-query",
        passage_prompt_name="Retrieval-document",
        max_seq_length=256,
    ),
}


def get_model_config(model_type: str) -> ModelConfig:
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model '{model_type}'. Choose from: {list(MODEL_CONFIGS)}")
    return MODEL_CONFIGS[model_type]


@dataclass
class TrainConfig:
    models_root: str = "finetune/models"

    # Hyperparams: ShawhinT framework + NVIDIA values
    num_epochs: int = 3
    batch_size: int = 16  # per-device; seq_len=128 makes e5-large fit on P100 16GB
    learning_rate: float = 1e-5        # NVIDIA: 1e-5 for large models
    warmup_ratio: float = 0.1          # ShawhinT: 0.1
    weight_decay: float = 0.01         # NVIDIA: 0.01
    lr_scheduler_type: str = "cosine"  # NVIDIA: cosine decay

    eval_steps: int = 100
    logging_steps: int = 50


@dataclass
class InferenceConfig:
    # Variable-K retrieval (threshold-based, better Macro F1 than fixed top-K)
    # Calibrate score_threshold on val.csv by sweeping 0.70–0.90 in steps of 0.02
    score_threshold: float = 0.78   # cosine similarity cutoff (normalized embeddings)
    min_k: int = 1                  # always return at least this many citations
    max_k: int = 20                 # cap to avoid excessive false positives


@dataclass
class RerankConfig:
    reranker_model_path: str = "finetune/models/bge-reranker-v2-m3"
    rerank_top_n: int = 50          # bi-encoder candidates fed to cross-encoder
    reranker_batch_size: int = 32
    # BGE reranker returns raw logits (not 0-1 probabilities); calibrate on val
    reranker_score_threshold: float = 0.0   # logit threshold; sweep on val
    min_k: int = 1
    max_k: int = 20


@dataclass
class FinetuneConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)


# Singleton for easy import (defaults to e5-large)
cfg = FinetuneConfig()
