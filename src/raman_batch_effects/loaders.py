import dataclasses
from pathlib import Path

import pandas as pd
import ramanspy
from tqdm import tqdm

from raman_batch_effects import batch_correction, spectrum_utils
from raman_batch_effects.cache import cache
from raman_batch_effects.datasets import RamanDataset
from raman_batch_effects.utils import get_data_dirpath

PLATEMAP_FILEPATH_NOVEMBER_2025 = (
    get_data_dirpath() / "november-2025-platemaps" / "2025-11-14-yeast_raman_3_days_plate_map.csv"
)

PLATEMAP_FILEPATHS_AUGUST_2025 = {
    1: get_data_dirpath() / "august-2025-platemaps" / "08-06_yeast-mutants.csv",
    2: get_data_dirpath() / "august-2025-platemaps" / "08-07_yeast-mutants.csv",
    3: get_data_dirpath() / "august-2025-platemaps" / "08-12_yeast-mutants.csv",
}

SUBDIRECTORY_NAMES_AUGUST_2025 = {
    1: "2025-08-06",
    2: "2025-08-07",
    3: "2025-08-12",
}

SUBDIRECTORY_NAMES_NOVEMBER_2025 = {
    1: "2025-11-11_yeast_raman_day1",
    2: "2025-11-12_yeast_raman_day2",
    3: "2025-11-13_yeast_raman_day3",
}


def parse_open_raman_file(file_path: str | Path) -> ramanspy.Spectrum:
    """
    Parse an OpenRaman CSV file and return a `ramanspy.Spectrum` instance.
    """
    df = pd.read_csv(file_path)
    df.dropna(axis=0, how="any", inplace=True)

    # Earlier datasets used the "Pixel" column for the spectral axis,
    # while later datasets use the "Wavenumber" column.
    spectral_axis_column = "Pixel"
    if spectral_axis_column not in df.columns:
        spectral_axis_column = "Wavenumber"

    spectrum = ramanspy.Spectrum(
        spectral_data=df["Intensity"].values,
        spectral_axis=df[spectral_axis_column].values,
    )

    return spectrum


def get_preprocessing_pipeline() -> ramanspy.preprocessing.Pipeline:
    """
    Get the preprocessing pipeline used for all spectra.
    """
    return ramanspy.preprocessing.Pipeline(
        [
            # Crop to 300 cm⁻¹ to eliminate the sharp shoulder at low wavenumbers.
            ramanspy.preprocessing.misc.Cropper(region=(300, None)),
            ramanspy.preprocessing.despike.WhitakerHayes(),
        ]
    )


def _remove_zero_padding_from_well_id(well_id: str) -> str:
    """
    Remove zero padding from a well ID.
    """
    row, column = well_id[0], int(well_id[1:])
    return f"{row}{column}"


def load_platemap_august_2025() -> pd.DataFrame:
    """
    Load the platemaps for the yeast dataset from August 2025.
    These are stored as three separate files, one for each day.
    """
    platemaps = []
    for day, filepath in PLATEMAP_FILEPATHS_AUGUST_2025.items():
        platemap = pd.read_csv(filepath)
        platemap["day"] = day
        platemaps.append(platemap[["well_id", "day", "strain", "species"]])

    platemap = pd.concat(platemaps)

    platemap["well_id"] = platemap["well_id"].apply(_remove_zero_padding_from_well_id)

    return platemap


def load_platemap_november_2025() -> pd.DataFrame:
    """
    Load the platemap for the yeast dataset from November 2025.
    """
    platemap = pd.read_csv(PLATEMAP_FILEPATH_NOVEMBER_2025)

    platemap.rename(
        columns={
            "Well": "well_id",
            "Day 1": "day_1",
            "Day 2": "day_2",
            "Day 3": "day_3",
        },
        inplace=True,
    )

    platemap = platemap[["well_id", "day_1", "day_2", "day_3"]]

    # Remove parentheses from strain names.
    for day in ["day_1", "day_2", "day_3"]:
        platemap[day] = platemap[day].str.split(" ").str[0]

    # HACK: Add back the (ts) appendix to the one temperature-sensitive strain.
    platemap = platemap.map(lambda x: x + " (ts)" if x == "sec6" else x)

    # Check that each day has the same number of strains.
    for day in ["day_1", "day_2", "day_3"]:
        for other_day in ["day_1", "day_2", "day_3"]:
            if other_day == day:
                continue
            if platemap[day].nunique() != platemap[other_day].nunique():
                raise ValueError(
                    f"{day} has {platemap[day].nunique()} strains, "
                    f"but {other_day} has {platemap[other_day].nunique()} strains."
                )

    # Flatten the platemap into a single dataframe with a "day" column.
    day_platemaps = []
    for day in [1, 2, 3]:
        day_platemap = platemap[["well_id", f"day_{day}"]].copy()
        day_platemap["day"] = day
        day_platemap.rename(columns={f"day_{day}": "strain"}, inplace=True)
        day_platemaps.append(day_platemap)

    platemap = pd.concat(day_platemaps)

    # Add rows for empty-well controls (always well A5), as these are not in the original platemap.
    empty_wells = pd.DataFrame(
        {"well_id": ["A5"] * 3, "day": [1, 2, 3], "strain": ["empty-well"] * 3}
    )
    platemap = pd.concat([platemap, empty_wells])

    # Add species column (not in original platemap but needed for consistency)
    platemap["species"] = "yeast"

    return platemap


@cache.cache()
def load_yeast_spectra(data_dirpath: str | Path) -> RamanDataset:
    """
    Load yeast Raman spectra from August and/or November 2025 acquisitions.

    Arguments:
        data_dirpath: Path to the root data directory containing acquisition subdirectories.

    Returns:
        A RamanDataset containing the spectra with metadata including:
        - day: 1, 2, or 3
        - well_id: e.g., "A1", "B3"
        - strain: strain name from platemap
        - species: "yeast"
        - filepath: path to the CSV file
    """
    data_dirpath = Path(data_dirpath)

    dataset = RamanDataset()

    platemap_august = load_platemap_august_2025()

    for day in tqdm([1, 2, 3], desc="Loading August 2025"):
        subdirectory_path = data_dirpath / SUBDIRECTORY_NAMES_AUGUST_2025[day] / "yeast_mut"

        if not subdirectory_path.exists():
            print(f"Warning: directory not found: {subdirectory_path}")
            continue

        for filepath in subdirectory_path.glob("*.csv"):
            well_id = filepath.name.split("-")[0]

            # Look up metadata from platemap
            platemap_row = platemap_august.loc[
                (platemap_august.well_id == well_id) & (platemap_august.day == day)
            ]

            if platemap_row.empty:
                print(f"Warning: well {well_id} on day {day} not found in August platemap")
                continue

            platemap_row = platemap_row.iloc[0]

            spectrum = parse_open_raman_file(filepath)
            spectrum = get_preprocessing_pipeline().apply(spectrum)

            dataset.add_spectrum(
                spectrum,
                date="august-2025",
                day=day,
                well_id=well_id,
                strain=platemap_row.strain,
                species=platemap_row.species,
                filepath=str(filepath),
            )

    return dataset

    # TODO: remove the code below if we decide not to use the November 2025 dataset.
    platemap_november = load_platemap_november_2025()

    for day in tqdm([1, 2, 3], desc="Loading November 2025"):
        subdirectory_path = (
            data_dirpath / SUBDIRECTORY_NAMES_NOVEMBER_2025[day] / f"spectra_day{day}"
        )

        if not subdirectory_path.exists():
            print(f"Warning: directory not found: {subdirectory_path}")
            continue

        for filepath in subdirectory_path.glob("*.csv"):
            well_id = filepath.name.split("-")[0]

            platemap_row = platemap_november.loc[
                (platemap_november.well_id == well_id) & (platemap_november.day == day)
            ]

            if platemap_row.empty:
                print(f"Warning: well {well_id} on day {day} not found in November platemap")
                continue

            platemap_row = platemap_row.iloc[0]

            spectrum = parse_open_raman_file(filepath)
            spectrum = get_preprocessing_pipeline().apply(spectrum)

            dataset.add_spectrum(
                spectrum,
                date="november-2025",
                day=day,
                well_id=well_id,
                strain=platemap_row.strain,
                species=platemap_row.species,
                filepath=str(filepath),
            )

    return dataset


@cache.cache()
def load_background_spectra(data_dirpath: str | Path) -> RamanDataset:
    """
    Load background/dark spectra for yeast acquisitions.

    Arguments:
        data_dirpath: Path to the root data directory.

    Returns:
        A RamanDataset containing background spectra with metadata including:
        - day: 1, 2, or 3
    """
    data_dirpath = Path(data_dirpath)

    dataset = RamanDataset()

    for day in [1, 2, 3]:
        filepath = data_dirpath / SUBDIRECTORY_NAMES_AUGUST_2025[day] / "dark" / "Default.csv"

        if not filepath.exists():
            print(f"Warning: background file not found: {filepath}")
            continue

        spectrum = parse_open_raman_file(filepath)
        spectrum = get_preprocessing_pipeline().apply(spectrum)
        dataset.add_spectrum(spectrum, date="august-2025", day=day)

    # TODO: remove the code below if we decide not to use the November 2025 dataset.
    for day in [1, 2, 3]:
        filepath = data_dirpath / SUBDIRECTORY_NAMES_NOVEMBER_2025[day] / "SS" / "Default.csv"

        if not filepath.exists():
            print(f"Warning: background file not found: {filepath}")
            continue

        spectrum = parse_open_raman_file(filepath)
        spectrum = get_preprocessing_pipeline().apply(spectrum)
        dataset.add_spectrum(spectrum, date="november-2025", day=day)

    return dataset


def subtract_background_spectra(
    dataset: RamanDataset, background_dataset: RamanDataset
) -> RamanDataset:
    """
    Subtract acquisition-specific consensus background spectra from all spectra.

    For each acquisition (august-2025, november-2025), computes a consensus background
    spectrum from the backgrounds for that acquisition, then subtracts it from all
    spectra from that acquisition.

    Arguments:
        dataset: RamanDataset containing spectra to correct.
        background_dataset: RamanDataset containing background spectra.
        mean_center: If True, mean-center background spectra before averaging.

    Returns:
        New RamanDataset with background-subtracted spectra.
    """
    background_subtracted_dataset = RamanDataset()

    for date in dataset.metadata.date.unique():
        background_dataset_for_date = background_dataset.filter(date=date)

        if len(background_dataset_for_date) == 0:
            print(f"Warning: no background spectra found for date {date}")
            continue

        background_spectra, _ = background_dataset_for_date.to_matrix()
        mean_background_spectrum = background_spectra.mean(axis=0)

        dataset_for_date = dataset.filter(date=date)
        for spectrum, metadata in dataset_for_date:
            background_subtracted_spectrum = ramanspy.Spectrum(
                spectrum.spectral_data - mean_background_spectrum,
                spectrum.spectral_axis,
            )
            background_subtracted_dataset.add_spectrum(
                background_subtracted_spectrum, **metadata.to_dict()
            )

    return background_subtracted_dataset


def process_spectra(
    dataset: RamanDataset,
    crop_region: tuple[int, int],
    modpoly_poly_order: int | None = None,
) -> RamanDataset:
    """
    Process yeast spectra by subtracting a baseline, smoothing, and normalizing.

    Arguments:
        dataset: RamanDataset containing spectra to process.
        crop_region: The region of the spectrum to crop to before processing.
        modpoly_poly_order: Polynomial order for ModPoly baseline correction.

    Returns:
        New RamanDataset with processed spectra.
    """
    processing_steps = [
        ramanspy.preprocessing.misc.Cropper(region=crop_region),
        ramanspy.preprocessing.denoise.SavGol(window_length=5, polyorder=3),
        ramanspy.preprocessing.baseline.ModPoly(poly_order=modpoly_poly_order),
        ramanspy.preprocessing.normalise.AUC(),
    ]
    return dataset.apply(processing_steps)


def identify_dim_spectra(
    dataset: RamanDataset, raw_dataset: RamanDataset, threshold: float = 1000.0
) -> RamanDataset:
    """
    Identify dim spectra using mean intensity from raw (background-subtracted) dataset.
    Adds an "is_dim" column to the metadata.

    Arguments:
        dataset: The processed RamanDataset to annotate.
        raw_dataset: The raw (background-subtracted but not fully processed) RamanDataset
                     used to compute mean intensities.
        threshold: Mean intensity threshold below which spectra are considered dim.

    Returns:
        The input dataset with "is_dim" column added to metadata.
    """
    dataset.metadata["is_dim"] = False
    for spectrum, metadata_row in raw_dataset:
        mean_intensity = spectrum.spectral_data.mean()
        if mean_intensity < threshold:
            mask = dataset.metadata.filepath == metadata_row.filepath
            dataset.metadata.loc[mask, "is_dim"] = True

    return dataset


def identify_outlier_spectra(dataset: RamanDataset) -> RamanDataset:
    """
    Identify outlier spectra within each day-strain group.
    Adds an "is_outlier" column to the metadata.

    This is a convenience wrapper around spectrum_utils.identify_outlier_spectra
    that groups by date, day, and strain.

    Arguments:
        dataset: RamanDataset to annotate.

    Returns:
        The input dataset with "is_outlier" column added to metadata.
    """
    return spectrum_utils.identify_outlier_spectra(dataset, group_by=["date", "day", "strain"])


@dataclasses.dataclass
class YeastDatasets:
    """
    This class namespaces the different versions of the yeast dataset
    produced by the `load_and_process_spectra` function.
    """

    # The original raw dataset.
    raw: RamanDataset

    # The dataset after subtracting the background or "dark" spectra.
    background_subtracted: RamanDataset

    # The dataset after processing.
    processed: RamanDataset

    # The processed dataset after removing dim spectra.
    processed_no_dim: RamanDataset

    # The processed dataset after removing dim spectra and outlier spectra.
    processed_no_dim_no_outliers: RamanDataset

    # A copy of the dataset used for batch correction.
    uncorrected: RamanDataset | None = None

    # The batch-corrected datasets (for plate-level batch effects).
    corrected_lmm: RamanDataset | None = None
    corrected_combat: RamanDataset | None = None

    def construct_composite_metadata_columns(self) -> None:
        """
        Construct composite metadata columns for the datasets.
        """
        for field in dataclasses.fields(self):
            dataset = getattr(self, field.name)
            if dataset is None:
                continue
            metadata = dataset.metadata
            # Create a single plate_id column to use for batch correction.
            metadata["plate_id"] = metadata.date + "-" + metadata.day.apply(lambda x: f"day-{x}")


@cache.cache
def load_and_process_spectra(
    data_dirpath: str | Path, crop_region: tuple[int, int]
) -> tuple[YeastDatasets, RamanDataset]:
    """
    Load and process the yeast spectra.
    """
    raw_dataset = load_yeast_spectra(data_dirpath)
    background_dataset = load_background_spectra(data_dirpath)
    background_subtracted_dataset = subtract_background_spectra(raw_dataset, background_dataset)

    processed_dataset = process_spectra(
        background_subtracted_dataset, crop_region=crop_region, modpoly_poly_order=5
    )

    # Remove dim spectra.
    # # Note: it is important to do this *before* identifying outliers,
    # as dim spectra diminish the effectiveness of the outlier identification algorithm.
    processed_dataset = identify_dim_spectra(
        processed_dataset, background_subtracted_dataset, threshold=1000.0
    )
    processed_dataset_no_dim = processed_dataset.loc(~processed_dataset.metadata.is_dim)
    print(
        f"Number of dim spectra removed: {len(processed_dataset) - len(processed_dataset_no_dim)}"
    )

    # Remove outlier spectra.
    processed_dataset_no_dim = identify_outlier_spectra(processed_dataset_no_dim)
    processed_dataset_no_dim_no_outliers = processed_dataset_no_dim.loc(
        ~processed_dataset_no_dim.metadata.is_outlier
    )
    print(
        f"Number of outlier spectra removed: "
        f"{len(processed_dataset_no_dim) - len(processed_dataset_no_dim_no_outliers)}"
    )

    print(f"Final dataset length: {len(processed_dataset_no_dim_no_outliers)} spectra")

    datasets = YeastDatasets(
        raw=raw_dataset,
        background_subtracted=background_subtracted_dataset,
        processed=processed_dataset,
        processed_no_dim=processed_dataset_no_dim,
        processed_no_dim_no_outliers=processed_dataset_no_dim_no_outliers,
    )

    datasets.construct_composite_metadata_columns()

    # We batch-correct the final cleaned dataset.
    datasets.uncorrected = datasets.processed_no_dim_no_outliers.copy()

    # Remove empty-well spectra prior to batch correction.
    datasets.uncorrected = datasets.uncorrected.loc(
        datasets.uncorrected.metadata.strain != "empty-well"
    )

    datasets.corrected_lmm = batch_correction.correct_batch_effects_lmm(
        datasets.uncorrected,
        batch_column="plate_id",
        fixed_effect_column=None,
    )

    datasets.corrected_combat = batch_correction.correct_batch_effects_combat(
        datasets.uncorrected,
        batch_column="plate_id",
        fixed_effect_column=None,
    )

    return datasets, background_dataset
