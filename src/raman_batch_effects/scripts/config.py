from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from raman_batch_effects.utils import get_data_dirpath

RANDOM_STATE = 42
CROP_REGION = (300, 1800)

# This determines the order of the strains in the plots of mean spectra
# and in the confusion matrices. It uses the original strain names (from the platemaps).
STRAIN_ORDER = [
    # cerevisiae strains
    "BY4741",
    "YGL058",
    "YNL141",
    # pombe strains
    "SP286",
    "ED666",
    "RAD6",
    "PDF1",
    "DEA2",
    "ARSG",
]

# The names used for each strain in all of the plots.
STRAIN_DISPLAY_NAMES = {
    "BY4741": "wild-type haploid (S. cerevisiae)",
    "YGL058": "rad6Δ (S. cerevisiae)",
    "YNL141": "aah1Δ (S. cerevisiae)",
    "ED666": "wild-type haploid (S. pombe)",
    "SP286": "wild-type diploid (S. pombe)",
    "DEA2": "dea2Δ (S. pombe)",
    "RAD6": "rhp6Δ (S. pombe)",  # "RAD6" was a mistake in the platemap.
    "PDF1": "pdf1Δ (S. pombe)",
    "ARSG": "asgΔ (S. pombe)",  # "ARSG" was a typo in the platemap.
}


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
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
