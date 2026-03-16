import argparse
import shutil
from pathlib import Path

import arcadia_pycolor as apc
import matplotlib.pyplot as plt

from raman_batch_effects import loaders, utils
from raman_batch_effects.cache import cache
from raman_batch_effects.scripts.config import (
    STRAIN_DISPLAY_NAMES,
    STRAIN_ORDER,
    YeastConfig,
    get_output_dir,
)

apc.mpl.setup()

CONFIG = YeastConfig()
OUTPUT_DIR = get_output_dir(Path(__file__).stem)

AUGUST_2025 = "august-2025"

# Species labels for grouping strains
CEREVISIAE_SPECIES_LABELS = ["cerevisiae", "Saccharomyces cerevisiae"]
POMBE_SPECIES_LABELS = ["pombe", "Schizosaccharomyces pombe"]


def plot_mean_spectra_by_strain_and_species(dataset, date: str, overwrite: bool = False):
    """
    Plot mean spectra for each strain across all days, grouped by species.

    Creates a figure with two panels (rows):
    - Top panel: cerevisiae strains
    - Bottom panel: pombe strains
    """
    X, labels = dataset.filter(date=date).to_matrix()

    # Identify cerevisiae and pombe strains, ordered by STRAIN_ORDER.
    available_strains = set(labels.strain.unique())
    cerevisiae_strains = []
    pombe_strains = []

    for strain in STRAIN_ORDER:
        if strain not in available_strains:
            continue
        strain_species = labels[labels.strain == strain].species.iloc[0].lower()
        if "cerevisiae" in strain_species:
            cerevisiae_strains.append(strain)
        elif "pombe" in strain_species:
            pombe_strains.append(strain)

    # Create figure with 2 rows
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Color palette
    colors = apc.palettes.all_palettes[4]

    # Plot cerevisiae strains in top panel
    ax = axes[0]
    for strain, color in zip(cerevisiae_strains, colors, strict=False):
        mask = labels.strain == strain
        if mask.sum() == 0:
            continue
        X_masked = X[mask]
        ax.plot(
            dataset.wavenumbers,
            X_masked.mean(axis=0),
            label=STRAIN_DISPLAY_NAMES.get(strain, strain),
            lw=1.5,
            alpha=0.9,
            color=color,
        )

    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    ax.set_ylabel("Intensity (a.u.)", fontsize=12)
    ax.set_title("Mean spectra by strain  |  S. cerevisiae  |  All plates", fontsize=13)
    apc.mpl.style_plot(ax, monospaced_axes="both")

    # Plot pombe strains in bottom panel
    ax = axes[1]
    for strain, color in zip(pombe_strains, colors, strict=False):
        mask = labels.strain == strain
        if mask.sum() == 0:
            continue
        X_masked = X[mask]
        ax.plot(
            dataset.wavenumbers,
            X_masked.mean(axis=0),
            label=STRAIN_DISPLAY_NAMES.get(strain, strain),
            lw=1.5,
            alpha=0.9,
            color=color,
        )

    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    ax.set_xlabel("Raman shift (cm$^{-1}$)", fontsize=12)
    ax.set_ylabel("Intensity (a.u.)", fontsize=12)
    ax.set_title("Mean spectra by strain  |  S. pombe  |  All plates", fontsize=13)
    apc.mpl.style_plot(ax, monospaced_axes="both")

    plt.tight_layout()
    utils.save_figure(
        OUTPUT_DIR / f"mean-spectra-by-strain-by-species--{date}.pdf",
        overwrite=overwrite,
    )


def main(overwrite: bool = False):
    """
    Generate plots of all spectra and mean spectra for yeast datasets.

    This script produces:
    - All spectra for each well in plate-layout format
    - All spectra for each strain and day
    - Mean spectra per strain for each date (day 3 only)
    - Mean spectra per day (all strains)
    - Mean spectra per day-species pair
    """

    date_to_plot = AUGUST_2025
    datasets, _ = loaders.load_and_process_spectra(CONFIG.data_dirpath, CONFIG.crop_region)

    clean_dataset = datasets.processed_no_dim_no_outliers

    # Plot all spectra in a grid of subplots with strains in rows and days in columns.
    X, labels = clean_dataset.filter(date=date_to_plot).to_matrix()
    available_strains = set(labels.strain.unique())
    strains = [s for s in STRAIN_ORDER if s in available_strains]
    num_strains = len(strains)

    _fig, axs = plt.subplots(num_strains, 3, figsize=(16, num_strains * 2))
    if num_strains == 1:
        axs = axs.reshape(1, -1)

    for ind, strain in enumerate(strains):
        for day in range(1, 4):
            ax = axs[ind, day - 1]
            mask = (labels.strain == strain) & (labels.day == day)
            for row in X[mask]:
                ax.plot(row, color="k", alpha=0.1, lw=0.5)
            ax.set_title(f"{STRAIN_DISPLAY_NAMES.get(strain, strain)}  |  plate{day}", fontsize=10)

            if ind == num_strains - 1:
                ax.set_xlabel("Wavenumber index", fontsize=10)
            else:
                ax.set_xlabel("")
                ax.set_xticks([])

            if day == 1:
                ax.set_ylabel("Intensity (a.u.)", fontsize=10)
            else:
                ax.set_ylabel("")
                ax.set_yticks([])

    plt.tight_layout()
    utils.save_figure(
        OUTPUT_DIR / f"all-spectra-by-strain-and-day--{date_to_plot}.png",
        overwrite=overwrite,
    )

    # Plot mean spectra for each strain overlaid, with dataset date by row.
    dates = [
        AUGUST_2025,
    ]

    for day in range(1, 4):
        _fig, axes = plt.subplots(len(dates), 1, figsize=(16, 8))
        if len(dates) == 1:
            axes = [axes]

        for ax, date in zip(axes, dates, strict=False):
            date_dataset = clean_dataset.filter(date=date)
            X_date, labels_date = date_dataset.to_matrix()

            colors = apc.palettes.all_palettes[4]
            available_strains_date = set(labels_date.strain.unique())
            ordered_strains = [s for s in STRAIN_ORDER if s in available_strains_date]
            for strain, color in zip(ordered_strains, colors, strict=False):
                mask = (labels_date.strain == strain) & (labels_date.day == day)
                if mask.sum() == 0:
                    continue
                X_masked = X_date[mask]
                ax.plot(
                    clean_dataset.wavenumbers,
                    X_masked.mean(axis=0),
                    label=STRAIN_DISPLAY_NAMES.get(strain, strain),
                    lw=1,
                    alpha=1,
                    color=color,
                )

            ax.legend(loc="upper right", fontsize=10, bbox_to_anchor=(1.15, 1))
            ax.set_xlabel("Raman shift (cm$^{-1}$)")
            ax.set_ylabel("Intensity (a.u.)")
            ax.set_title(f"Mean spectra  |  Plate {day}  |  {date}")
            apc.mpl.style_plot(ax, monospaced_axes="both")

        plt.tight_layout()
        utils.save_figure(
            OUTPUT_DIR / f"mean-spectra-by-strain--day{day}--both-dates.png",
            overwrite=overwrite,
        )

    # Plot mean spectra for each day-species pair.
    date_to_plot = AUGUST_2025
    plt.figure(figsize=(16, 4))
    colors = apc.palettes.all_palettes[4]

    X, labels = clean_dataset.filter(date=date_to_plot).to_matrix()

    batch_labels = labels.day.astype(str) + "-" + labels.species.values
    for batch_label, color in zip(sorted(batch_labels.unique()), colors, strict=False):
        mask = batch_labels == batch_label
        X_masked = X[mask]
        plt.plot(
            clean_dataset.wavenumbers,
            X_masked.mean(axis=0),
            label=batch_label,
            lw=1,
            alpha=1,
            color=color,
        )

    plt.legend(loc="upper right", fontsize=12)
    apc.mpl.style_plot(plt.gca(), monospaced_axes="both")
    plt.tight_layout()
    utils.save_figure(
        OUTPUT_DIR / f"mean-spectra-by-day-species--{date_to_plot}.png",
        overwrite=overwrite,
    )

    # Plot mean spectra for each strain (across all days), grouped by species
    date_to_plot = AUGUST_2025
    print("Generating mean spectra by strain and species plot...")
    plot_mean_spectra_by_strain_and_species(clean_dataset, date=date_to_plot, overwrite=overwrite)


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
