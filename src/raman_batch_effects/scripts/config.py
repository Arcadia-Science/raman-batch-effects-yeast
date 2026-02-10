from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from raman_batch_effects.utils import get_data_dirpath

RANDOM_STATE = 42
CROP_REGION = (300, 1800)


@dataclass(frozen=True)
class YeastConfig:
    """Configuration for loading and processing yeast spectra data."""

    data_dirpath: Path = get_data_dirpath() / "august-2025-spectra"
    crop_region: tuple[int, int] = CROP_REGION


DEFAULT_SVC_MODEL = SVC(
    C=100,
    probability=True,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)

DEFAULT_RF_MODEL = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
)


def get_output_dir(script_name: str) -> Path:
    """Get the output directory for a script, creating it if necessary."""
    todays_date = datetime.today().strftime("%Y-%m-%d")
    output_dir = Path("output") / todays_date / script_name
    return output_dir
