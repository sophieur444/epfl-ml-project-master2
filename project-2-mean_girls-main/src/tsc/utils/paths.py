# src/tsc/utils/paths.py

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TSC = ROOT / "src" / "tsc"

DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_TWITTER = DATA_RAW / "twitter-datasets"
DATA_INTERMEDIATE = DATA / "intermediate"
DATA_PROCESSED = DATA / "processed"
RESULTS = ROOT / "results"

GLOVE = TSC / "models" / "glove"
TFIDF = TSC / "models" / "tfidf"
DEBERTA = TSC / "models" / "deberta"
DISTILBERT = TSC / "models" / "distilbert"
