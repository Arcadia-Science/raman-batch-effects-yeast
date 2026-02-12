import argparse
import shutil
from pathlib import Path

import arcadia_pycolor as apc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from raman_batch_effects import loaders, plotting
from raman_batch_effects.cache import cache
from raman_batch_effects.cross_validation import (
    calc_confusion_matrix_lobo,
    calc_confusion_matrix_loo,
)
from raman_batch_effects.scripts import config
from raman_batch_effects.scripts.config import YeastConfig, get_output_dir

apc.mpl.setup()

CONFIG = YeastConfig()
OUTPUT_DIR = get_output_dir(Path(__file__).stem)

AUGUST_2025 = "august-2025"

# Manually-curated order for displaying confusion matrices.
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


def plot_kfold_cv_strain_prediction(dataset, suffix: str = "", overwrite: bool = False):
    """
    Generate confusion matrix for k-fold cross-validation to predict strain.

    Uses 5-fold cross-validation with random partitioning of all spectra.
    """
    X, labels = dataset.filter(date=AUGUST_2025).to_matrix()

    target_labels = labels.species + "-" + labels.strain

    confusion_matrix, unique_labels, ovr_auc, importances = calc_confusion_matrix_loo(
        X,
        Y=target_labels,
        model=config.DEFAULT_RF_MODEL,
        n_folds=5,
    )

    unique_strains = [label.split("-")[1] for label in unique_labels]
    sort_order = [unique_strains.index(strain) for strain in STRAIN_ORDER]

    plotting.plot_confusion_matrix(
        confusion_matrix[np.ix_(sort_order, sort_order)],
        [unique_labels[ind] for ind in sort_order],
        figsize=(12, 10),
        show_cell_counts=False,
        cmap=mpl.cm.Blues,
    )

    plt.tight_layout()
    plotting.utils.save_figure(
        OUTPUT_DIR / f"cf--kfold-cv--strain-prediction{suffix}.pdf",
        overwrite=overwrite,
    )


def plot_lodo_cv_strain_prediction(dataset, suffix: str = "", overwrite: bool = False):
    """
    Generate confusion matrix for leave-one-day-out cross-validation to predict strain.

    Trains on two days and tests on the third, rotating through all three days.
    """
    X, labels = dataset.filter(date=AUGUST_2025).to_matrix()

    target_labels = labels.species + "-" + labels.strain

    confusion_matrix, unique_labels, ovr_auc, feature_importances = calc_confusion_matrix_lobo(
        X,
        Y=target_labels,
        batch_labels=labels.day.values,
        model=config.DEFAULT_RF_MODEL,
    )

    unique_strains = [label.split("-")[1] for label in unique_labels]
    sort_order = [unique_strains.index(strain) for strain in STRAIN_ORDER]

    plotting.plot_confusion_matrix(
        confusion_matrix[np.ix_(sort_order, sort_order)],
        [unique_labels[ind] for ind in sort_order],
        figsize=(12, 10),
        show_cell_counts=False,
        cmap=mpl.cm.Blues,
    )

    plt.tight_layout()
    plotting.utils.save_figure(
        OUTPUT_DIR / f"cf--lodo-cv--strain-prediction{suffix}.pdf",
        overwrite=overwrite,
    )


def plot_loso_cv_day_prediction(dataset, suffix: str = "", overwrite: bool = False):
    """
    Generate confusion matrix for leave-one-strain-out cross-validation to predict day.

    This is an "adversarial" test - we shouldn't be able to predict plate/day identity
    if there are no batch effects. High accuracy indicates strong batch effects.
    """
    X, labels = dataset.filter(date=AUGUST_2025).to_matrix()

    target_labels = labels.day.astype(str)

    confusion_matrix, unique_labels, _, feature_importances = calc_confusion_matrix_lobo(
        X,
        Y=target_labels,
        batch_labels=labels.strain.values,
        model=config.DEFAULT_RF_MODEL,
    )

    plotting.plot_confusion_matrix(
        confusion_matrix,
        [f"Plate-{day}" for day in unique_labels],
        figsize=(6, 5),
        show_cell_counts=True,
        cmap=mpl.cm.Blues,
    )

    plt.tight_layout()
    plotting.utils.save_figure(
        OUTPUT_DIR / f"cf--loso-cv--day-prediction{suffix}.pdf",
        overwrite=overwrite,
    )


def plot_lodo_cv_species_prediction(dataset, suffix: str = "", overwrite: bool = False):
    """
    Generate confusion matrix for leave-one-day-out cross-validation to predict species.

    Tests whether species-level differences are strong enough to generalize across batches.
    """
    dataset_filtered = dataset.filter(date=AUGUST_2025)

    plotting.plot_lobo_cv_results(
        dataset_filtered,
        y_column="species",
        batch_column="day",
        model=config.DEFAULT_RF_MODEL,
        force_confusion_matrix=True,
        show_cell_counts=True,
        cmap=mpl.cm.Blues,
    )

    plt.tight_layout()
    plotting.utils.save_figure(
        OUTPUT_DIR / f"cf--lodo-cv--species-prediction{suffix}.pdf",
        overwrite=overwrite,
    )


def plot_loso_cv_day_prediction_with_wrapper(dataset, suffix: str = "", overwrite: bool = False):
    """
    Generate confusion matrix for leave-one-strain-out CV to predict day using wrapper function.

    Alternative visualization using the plot_lobo_cv_results wrapper.
    """
    dataset_filtered = dataset.filter(date=AUGUST_2025)

    plotting.plot_lobo_cv_results(
        dataset_filtered,
        y_column="day",
        batch_column="strain",
        model=config.DEFAULT_RF_MODEL,
    )

    plt.tight_layout()
    plotting.utils.save_figure(
        OUTPUT_DIR / f"cf--loso-cv--day-prediction-wrapper{suffix}.pdf",
        overwrite=overwrite,
    )


def main(overwrite: bool = False):
    """
    Generate all cross-validation confusion matrix plots.

    Creates two versions of each plot: one using uncorrected data and one using
    batch-corrected data.
    """
    datasets, _ = loaders.load_and_process_spectra(CONFIG.data_dirpath, CONFIG.crop_region)

    # Generate plots for both uncorrected and corrected datasets
    for dataset_name, dataset in [
        ("uncorrected", datasets.uncorrected),
        ("corrected", datasets.corrected),
    ]:
        suffix = f"--{dataset_name}"
        print(f"\n=== Generating {dataset_name} plots ===")

        print("Generating k-fold CV strain prediction plot...")
        plot_kfold_cv_strain_prediction(dataset, suffix=suffix, overwrite=overwrite)

        print("Generating leave-one-day-out CV strain prediction plot...")
        plot_lodo_cv_strain_prediction(dataset, suffix=suffix, overwrite=overwrite)

        print("Generating leave-one-strain-out CV day prediction plot...")
        plot_loso_cv_day_prediction(dataset, suffix=suffix, overwrite=overwrite)

        print("Generating leave-one-day-out CV species prediction plot...")
        plot_lodo_cv_species_prediction(dataset, suffix=suffix, overwrite=overwrite)

        print("Generating leave-one-strain-out CV day prediction plot (wrapper)...")
        plot_loso_cv_day_prediction_with_wrapper(dataset, suffix=suffix, overwrite=overwrite)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--clear", action="store_true", help="Clear joblib cache before running")
    args = parser.parse_args()

    if args.clear:
        print("Clearing joblib cache...")
        cache.clear()
        print("Cache cleared.")

    if args.reset:
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    main(overwrite=args.overwrite)
