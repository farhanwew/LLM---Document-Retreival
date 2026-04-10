from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent   # project root (parent of src/)
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "val.csv"
TEST_PATH = DATA_DIR / "test.csv"
LAWS_PATH = DATA_DIR / "laws_de.csv"
COURT_PATH = DATA_DIR / "court_considerations.csv"

CORPUS_EMBEDDINGS_PATH = MODELS_DIR / "corpus_embeddings.npy"
CORPUS_CITATIONS_PATH = MODELS_DIR / "corpus_citations.npy"
FAISS_INDEX_PATH = MODELS_DIR / "corpus.index"

FINETUNED_MODEL_PATH = MODELS_DIR / "qwen-swiss-legal"
HARD_NEG_DATASET_PATH = DATA_DIR / "hard_negative_dataset.json"

SUBMISSION_PATH = OUTPUTS_DIR / "submission.csv"

# ── Model ──────────────────────────────────────────────────────────────────
# Qwen3-Embedding sizes vs Kaggle GPU memory (T4 = 16 GB):
#   Qwen/Qwen3-Embedding-0.6B  → ~2 GB VRAM, dim=1024  ✓ safe for T4
#   Qwen/Qwen3-Embedding-4B    → ~8 GB VRAM            ✓ fits on T4
#   Qwen/Qwen3-Embedding       → ~16+ GB VRAM          ✗ too large for T4
BASE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DIM = 1024

# Qwen3-Embedding performs better with a task instruction prepended to queries.
# Documents are encoded without any prefix.
QUERY_INSTRUCTION = (
    "Instruct: Given an English legal question, retrieve the most relevant "
    "Swiss legal sources (statutes and court decisions).\nQuery: "
)

# ── Retrieval ──────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K = 20          # fallback fixed top-k
HARD_NEG_TOP_K = 50           # candidates to mine hard negatives from
HARD_NEG_PER_QUERY = 3        # hard negatives per anchor

# Adaptive threshold: cut citations where score drops by more than this fraction
# of the top score. None = use fixed RETRIEVAL_TOP_K.
ADAPTIVE_GAP_FRACTION = 0.15

# ── Training ───────────────────────────────────────────────────────────────
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 3e-5
WARMUP_STEPS = 100
FP16 = True

# ── Encoding ───────────────────────────────────────────────────────────────
ENCODE_BATCH_SIZE = 256       # safe now that MAX_SEQ_LENGTH is capped
MAX_SEQ_LENGTH = 512          # truncate to this many tokens (Qwen3 default is 32k → OOM)
NORMALIZE_EMBEDDINGS = True   # required for cosine similarity with FAISS IP
