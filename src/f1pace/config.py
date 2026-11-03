"""Shared project paths, constants, and directory setup."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
TABLE_DIR = ROOT / "reports" / "tables"
ARCHIVE_CATALOG_PATH = Path(__file__).with_name("archive_catalog.json")

SEED = 42
TRAIN_YEARS = [2023, 2024]
TEST_YEARS = [2025]
SESSION = "R"

DRY_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}
GREEN_TRACK_STATUS = "1"          # keep only laps whose TrackStatus is exactly "1"
OUTLIER_FACTOR = 1.07             # drop laps slower than 107% of race clean median
WET_COMPOUND_SHARE_MAX = 0.05     # race excluded if >5% of laps on INTERMEDIATE/WET
N_LAGS_REQUIRED = 3
N_LAGS_MAX = 5
CV_FOLDS = 5
TYRE_AGE_BUCKETS = [0, 5, 10, 20, 30, 100]   # labels: 1-5, 6-10, 11-20, 21-30, 31+


def ensure_dirs() -> None:
    """Create every data/output directory if missing."""
    for d in (CACHE_DIR, PROCESSED_DIR, MODELS_DIR, FIG_DIR, TABLE_DIR):
        d.mkdir(parents=True, exist_ok=True)
