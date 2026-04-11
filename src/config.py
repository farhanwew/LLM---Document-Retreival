from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "val.csv"
TEST_PATH = DATA_DIR / "test.csv"
LAWS_PATH = DATA_DIR / "laws_de.csv"
COURT_PATH = DATA_DIR / "court_considerations.csv"

BM25_INDEX_PATH = MODELS_DIR / "bm25_index.pkl"
MILCO_INDEX_PATH = MODELS_DIR / "milco_index.npz"
MILCO_META_PATH = MODELS_DIR / "milco_meta.pkl"

SUBMISSION_PATH = OUTPUTS_DIR / "submission.csv"

# ── Model ──────────────────────────────────────────────────────────────────
# MILCO only used at query time to rerank BM25 candidates — no full corpus encoding
BASE_MODEL = "omai-research/milco-650m"

# ── Corpus filtering ───────────────────────────────────────────────────────
CORPUS_MIN_YEAR = 2004   # drop court decisions older than this

# ── Retrieval ──────────────────────────────────────────────────────────────
BM25_TOP_K = 500          # candidates from BM25 first pass
RERANK_TOP_K = 20         # final citations after MILCO rerank

# Adaptive threshold: cut where score drops > this fraction of top score
ADAPTIVE_GAP_FRACTION = 0.15

# ── Encoding ───────────────────────────────────────────────────────────────
ENCODE_BATCH_SIZE = 64    # reduced from 64 to avoid OOM
