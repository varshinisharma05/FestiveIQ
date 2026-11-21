"""Central settings. Every other file imports paths and constants from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
RAW_CSV = DATA_DIR / "online_retail_II.csv"   # put the Kaggle file here
DB_PATH = DATA_DIR / "retail.db"              # created by clean_load.py

# Forecast test window: train before this date, test from this date on.
# The window contains the Oct-Dec 2011 festive peak, the hardest part to predict.
TEST_START = "2011-10-03"
LAST_COMPLETE_WEEK_END = "2011-12-04"   # data ends 9 Dec 2011 (incomplete week)

# ASSUMPTION (not in the data): gross margin used by the pricing simulator.
ASSUMED_MARGIN = 0.30

OUT_DIR.mkdir(exist_ok=True)
