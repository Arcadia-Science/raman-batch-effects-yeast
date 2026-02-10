import joblib

from raman_batch_effects import utils

REPO_ROOT = utils.find_repo_root()

cache = joblib.Memory(location=REPO_ROOT / "cache" / "joblib")
