# config.py
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
SAMPLE_DIR = DATA_DIR / "sample_stocks"
ASSETS_DIR = ROOT / "assets"
MODELS_DIR = ROOT / "models_store"
SKLEARN_DIR = MODELS_DIR / "sklearn"
TF_DIR = MODELS_DIR / "tensorflow"
REPORTS_DIR = ROOT / "reports" / "generated_reports"

LOTTIE_BULL = ASSETS_DIR / "lottie_bull.json"
LOTTIE_BEAR = ASSETS_DIR / "lottie_bear.json"

RANDOM_SEED = 42
DEFAULT_PERIOD = "5y"
DEFAULT_INTERVAL = "1d"

# Ensure dirs
for d in [DATA_DIR, UPLOAD_DIR, SAMPLE_DIR, ASSETS_DIR, SKLEARN_DIR, TF_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
