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
    corpus_chunk_size: int = 50_000    # From NVIDIA config


@dataclass
class ModelConfig:
    # Multilingual, strong cross-lingual alignment, ~560M params
    base_model: str = "intfloat/multilingual-e5-large"

    # Required prefixes for multilingual-e5 (from NVIDIA recipe)
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "

    # 128 covers 93.9% of corpus docs (mean=57 tokens); hardcoded in 02_finetune.py
    max_seq_length: int = 256


@dataclass
class TrainConfig:
    models_root: str = "finetune/models"

    # Hyperparams: ShawhinT framework + NVIDIA values
    num_epochs: int = 5
    batch_size: int = 16  # per-device; seq_len=128 makes e5-large fit on P100 16GB
    learning_rate: float = 1e-5        # NVIDIA: 1e-5 for large models
    warmup_ratio: float = 0.1          # ShawhinT: 0.1
    weight_decay: float = 0.01         # NVIDIA: 0.01
    lr_scheduler_type: str = "cosine"  # NVIDIA: cosine decay

    eval_steps: int = 100
    logging_steps: int = 50


@dataclass
class FinetuneConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# Singleton for easy import
cfg = FinetuneConfig()
