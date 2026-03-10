import argparse
import shutil
from pathlib import Path

import arcadia_pycolor as apc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml

from raman_batch_effects import loaders, plotting
from raman_batch_effects.cache import cache
from raman_batch_effects.cross_validation import (
    CVResults,
    calc_confusion_matrix_kfold,
    calc_confusion_matrix_lobo,
)
from raman_batch_effects.scripts import config
from raman_batch_effects.scripts.config import STRAIN_DISPLAY_NAMES, STRAIN_ORDER

apc.mpl.setup()

CONFIG = config.YeastConfig()
OUTPUT_DIR = config.get_output_dir(Path(__file__).stem)

AUGUST_2025 = "august-2025"


def _format_metrics_report(
    task_description,
    cv_strategy,
    cv_results: CVResults,
):
    """Format classification metrics as a YAML report with raw per-fold values."""
    metric_fields = {
        "per_fold_accuracy": "accuracy",
        "per_fold_macro_precision": "macro_precision",
        "per_fold_macro_recall": "macro_recall",
        "per_fold_macro_f1": "macro_f1",
        "per_fold_weighted_f1": "weighted_f1",
        "per_fold_ovr_auc": "ovr_auc",
        "per_fold_ovr_mcc": "ovr_mcc",
        "per_fold_mcc": "mcc",
    }

    per_fold = {}
    for field, key in metric_fields.items():
        values = getattr(cv_results, field)
        if not values:
            continue
        per_fold[key] = [round(float(v), 4) for v in values]

    report = {
        "task": task_description,
        "cv_strategy": cv_strategy,
        "n_folds": len(cv_results.per_fold_accuracy),
        "n_classes": len(cv_results.unique_labels),
        "n_samples": int(len(cv_results.aggregate_y_true)),
        "per_fold_metrics": per_fold,
    }

    return yaml.dump(report, sort_keys=False, default_flow_style=False)


def _save_metrics(filepath, report_text, overwrite=False):
    """Save metrics report to a text file."""
    filepath = Path(filepath)
    if filepath.exists() and not overwrite:
        print(f"  Skipping {filepath.name} (file exists)")
        return
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(report_text)
    print(f"  Saved metrics to {filepath.name}")


def plot_kfold_cv_strain_prediction(dataset, model, suffix: str = "", overwrite: bool = False):
    """
    Generate confusion matrix for k-fold cross-validation to predict strain.

    Uses 5-fold cross-validation with random partitioning of all spectra.
    """
    X, labels = dataset.filter(date=AUGUST_2025).to_matrix()

    target_labels = labels.species + "-" + labels.strain

    cv_results = calc_confusion_matrix_kfold(
        X,
        Y=target_labels,
        model=model,
        n_folds=5,
    )

    unique_strains = [label.split("-")[1] for label in cv_results.unique_labels]
    sort_order = [unique_strains.index(strain) for strain in STRAIN_ORDER]

    display_labels = [
        STRAIN_DISPLAY_NAMES.get(unique_strains[ind], unique_strains[ind])
        for ind in sort_order
    ]

    plotting.plot_confusion_matrix(
        cv_results.confusion_matrix[np.ix_(sort_order, sort_order)],
        display_labels,
        figsize=(12, 10),
        show_cell_counts=False,
        cmap=mpl.cm.Blues,
    )

    plt.tight_layout()
    plotting.utils.save_figure(
        OUTPUT_DIR / f"cf--kfold-cv--strain-prediction{suffix}.pdf",
        overwrite=overwrite,
    )

    report = _format_metrics_report(
        task_description="Strain prediction",
        cv_strategy="5-fold cross-validation",
        cv_results=cv_results,
    )
    _save_metrics(
        OUTPUT_DIR / f"metrics--kfold-cv--strain-prediction{suffix}.yaml",
        report,
        overwrite=overwrite,
    )


def plot_lodo_cv_strain_prediction(dataset, model, suffix: str = "", overwrite: bool = False):
    """
    Generate confusion matrix for leave-one-day-out cross-validation to predict strain.

    Trains on two days and tests on the third, rotating through all three days.
    """
    X, labels = dataset.filter(date=AUGUST_2025).to_matrix()

    target_labels = labels.species + "-" + labels.strain

    cv_results = calc_confusion_matrix_lobo(
        X,
        Y=target_labels,
        batch_labels=labels.day.values,
        model=model,
    )

    unique_strains = [label.split("-")[1] for label in cv_results.unique_labels]
    sort_order = [unique_strains.index(strain) for strain in STRAIN_ORDER]

    display_labels = [
        STRAIN_DISPLAY_NAMES.get(unique_strains[ind], unique_strains[ind])
        for ind in sort_order
    ]

    plotting.plot_confusion_matrix(
        cv_results.confusion_matrix[np.ix_(sort_order, sort_order)],
        display_labels,
        figsize=(12, 10),
        show_cell_counts=False,
        cmap=mpl.cm.Blues,
    )

    plt.tight_layout()
    plotting.utils.save_figure(
        OUTPUT_DIR / f"cf--lodo-cv--strain-prediction{suffix}.pdf",
        overwrite=overwrite,
    )

    report = _format_metrics_report(
        task_description="Strain prediction",
        cv_strategy="Leave-one-day-out cross-validation",
        cv_results=cv_results,
    )
    _save_metrics(
        OUTPUT_DIR / f"metrics--lodo-cv--strain-prediction{suffix}.yaml",
        report,
        overwrite=overwrite,
    )


def plot_loso_cv_day_prediction(dataset, model, suffix: str = "", overwrite: bool = False):
    """
    Generate confusion matrix for leave-one-strain-out cross-validation to predict day.

    This is an "adversarial" test - we shouldn't be able to predict plate/day identity
    if there are no batch effects. High accuracy indicates strong batch effects.
    """
    X, labels = dataset.filter(date=AUGUST_2025).to_matrix()

    target_labels = labels.day.astype(str)

    cv_results = calc_confusion_matrix_lobo(
        X,
        Y=target_labels,
        batch_labels=labels.strain.values,
        model=model,
    )

    plotting.plot_confusion_matrix(
        cv_results.confusion_matrix,
        [f"Plate-{day}" for day in cv_results.unique_labels],
        figsize=(6, 5),
        show_cell_counts=True,
        cmap=mpl.cm.Blues,
    )

    plt.tight_layout()
    plotting.utils.save_figure(
        OUTPUT_DIR / f"cf--loso-cv--day-prediction{suffix}.pdf",
        overwrite=overwrite,
    )

    report = _format_metrics_report(
        task_description="Day/plate prediction",
        cv_strategy="Leave-one-strain-out cross-validation",
        cv_results=cv_results,
    )
    _save_metrics(
        OUTPUT_DIR / f"metrics--loso-cv--day-prediction{suffix}.yaml",
        report,
        overwrite=overwrite,
    )


def plot_lodo_cv_species_prediction(dataset, model, suffix: str = "", overwrite: bool = False):
    """
    Generate confusion matrix for leave-one-day-out cross-validation to predict species.

    Tests whether species-level differences are strong enough to generalize across batches.
    """
    dataset_filtered = dataset.filter(date=AUGUST_2025)

    cv_results = plotting.plot_lobo_cv_results(
        dataset_filtered,
        y_column="species",
        batch_column="day",
        model=model,
        force_confusion_matrix=False,
    )

    plt.tight_layout()
    plotting.utils.save_figure(
        OUTPUT_DIR / f"cf--lodo-cv--species-prediction{suffix}.pdf",
        overwrite=overwrite,
    )

    # `plot_lobo_cv_results` returns None when plotting a ROC curve for binary classification,
    # so run `calc_confusion_matrix_lobo` to get the CV results object.
    if cv_results is None:
        X, labels = dataset_filtered.to_matrix()
        cv_results = calc_confusion_matrix_lobo(
            X,
            Y=labels["species"].values,
            batch_labels=labels["day"].values,
            model=model,
        )

    report = _format_metrics_report(
        task_description="Species prediction",
        cv_strategy="Leave-one-day-out cross-validation",
        cv_results=cv_results,
    )
    _save_metrics(
        OUTPUT_DIR / f"metrics--lodo-cv--species-prediction{suffix}.yaml",
        report,
        overwrite=overwrite,
    )


def plot_loso_cv_day_prediction_with_wrapper(
    dataset, model, suffix: str = "", overwrite: bool = False
):
    """
    Generate confusion matrix for leave-one-strain-out CV to predict day,
    using the `plot_lobo_cv_results` wrapper function instead of plotting
    the confusion matrix directly.
    """
    dataset_filtered = dataset.filter(date=AUGUST_2025)

    _ = plotting.plot_lobo_cv_results(
        dataset_filtered,
        y_column="day",
        batch_column="strain",
        model=model,
    )

    plt.tight_layout()
    plotting.utils.save_figure(
        OUTPUT_DIR / f"cf--loso-cv--day-prediction-wrapper{suffix}.pdf",
        overwrite=overwrite,
    )


def main(overwrite: bool = False):
    """
    Generate all cross-validation confusion matrix plots.

    Creates two versions of each plot for each model: one using uncorrected data
    and one using batch-corrected data.
    """
    datasets, _ = loaders.load_and_process_spectra(CONFIG.data_dirpath, CONFIG.crop_region)

    for model_name, model in [("rf", config.DEFAULT_RF_MODEL), ("svc", config.DEFAULT_SVC_MODEL)]:
        for dataset_name, dataset in [
            ("uncorrected", datasets.uncorrected),
            ("corrected-lmm", datasets.corrected_lmm),
            ("corrected-combat", datasets.corrected_combat),
        ]:
            suffix = f"--{model_name}--{dataset_name}"
            print(f"\n=== Generating {model_name} / {dataset_name} plots ===")

            print("Generating k-fold CV strain prediction plot...")
            plot_kfold_cv_strain_prediction(dataset, model, suffix=suffix, overwrite=overwrite)

            print("Generating leave-one-day-out CV strain prediction plot...")
            plot_lodo_cv_strain_prediction(dataset, model, suffix=suffix, overwrite=overwrite)

            print("Generating leave-one-strain-out CV day prediction plot...")
            plot_loso_cv_day_prediction(dataset, model, suffix=suffix, overwrite=overwrite)

            print("Generating leave-one-day-out CV species prediction plot...")
            plot_lodo_cv_species_prediction(dataset, model, suffix=suffix, overwrite=overwrite)

            print("Generating leave-one-strain-out CV day prediction plot (wrapper)...")
            plot_loso_cv_day_prediction_with_wrapper(
                dataset, model, suffix=suffix, overwrite=overwrite
            )

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
