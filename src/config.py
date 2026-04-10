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

SPARSE_INDEX_PATH = MODELS_DIR / "corpus_sparse.npz"
CORPUS_CITATIONS_PATH = MODELS_DIR / "corpus_citations.npy"

SUBMISSION_PATH = OUTPUTS_DIR / "submission.csv"

# ── Model ──────────────────────────────────────────────────────────────────
# MILCO: multilingual learned sparse retrieval
#   milco-300m → lighter, faster encoding
#   milco-650m → better quality (recommended)
BASE_MODEL = "omai-research/milco-650m"

# ── Corpus filtering ───────────────────────────────────────────────────────
# Court decisions older than this year are unlikely to appear in gold citations
# BGE vol 130 ≈ year 2004
CORPUS_MIN_YEAR = 2004

# ── Retrieval ──────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K = 100         # candidates from sparse search
FINAL_TOP_K = 20              # after adaptive cutoff

# Adaptive threshold: cut where score drops > this fraction of top score
ADAPTIVE_GAP_FRACTION = 0.15

# ── Encoding ───────────────────────────────────────────────────────────────
ENCODE_BATCH_SIZE = 128
