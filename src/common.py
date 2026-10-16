from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
TARGET_COLUMN = "loan_status"


def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        return None
    return joblib.load(path)
