import sys
from pathlib import Path

# Set up project root in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "synthetic_training_dataset_FINAL_clean.csv"
PROCESSED_TRAIN_PATH = DATA_DIR / "processed" / "train.csv"
PROCESSED_TEST_PATH = DATA_DIR / "processed" / "test.csv"
MODEL_DIR = BASE_DIR / "models"

# Ensure directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)

# Target & Split settings
TARGET_COL = "simulated_delay_min"
CANCELLED_COL = "simulated_cancelled_flag"
SPLIT_DATE = "2023-10-31"

RANDOM_STATE = 42
