import argparse
import shutil
from pathlib import Path

import arcadia_pycolor as apc
import matplotlib.pyplot as plt

from raman_batch_effects import loaders, utils
from raman_batch_effects import plotting as yeast_plotting
from raman_batch_effects.scripts.config import YeastConfig, get_output_dir

apc.mpl.setup()

CONFIG = YeastConfig()
OUTPUT_DIR = get_output_dir(Path(__file__).stem)

AUGUST_2025 = "august-2025"


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

    background_subtracted_dataset = datasets.background_subtracted
    clean_dataset = datasets.processed_no_dim_no_outliers

    # Plot all spectra per well in plate-layout format for August 2025.
    for day in range(1, 4):
        for dataset_type, dataset in [
            ("background_subtracted", background_subtracted_dataset),
            ("processed", clean_dataset),
        ]:
            yeast_plotting.plot_plate_layout(
                dataset.filter(date=date_to_plot),
                day=day,
                ylim=(0, 60000) if dataset_type == "background_subtracted" else (-0.005, 0.005),
                alpha=0.9 if dataset_type == "background_subtracted" else 0.1,
                figsize=(16, 8),
            )
            utils.save_figure(
                OUTPUT_DIR
                / f"all-spectra-in-plate-layout--{dataset_type}-{date_to_plot}-day{day}.png",
                overwrite=overwrite,
            )

    # Plot all spectra in a grid of subplots with strains in rows and days in columns.
    X, labels = clean_dataset.filter(date=date_to_plot).to_matrix()
    strains = sorted(labels.strain.unique())
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
            ax.set_title(f"{strain}  |  day{day}", fontsize=10)

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
            for strain, color in zip(sorted(labels_date.strain.unique()), colors, strict=False):
                if strain == "YPD":
                    continue
                mask = (labels_date.strain == strain) & (labels_date.day == day)
                if mask.sum() == 0:
                    continue
                X_masked = X_date[mask]
                ax.plot(
                    clean_dataset.wavenumbers,
                    X_masked.mean(axis=0),
                    label=strain,
                    lw=1,
                    alpha=1,
                    color=color,
                )

            ax.legend(loc="upper right", fontsize=10, bbox_to_anchor=(1.15, 1))
            ax.set_xlabel("Raman shift (cm$^{-1}$)")
            ax.set_ylabel("Intensity (a.u.)")
            ax.set_title(f"Mean spectra  |  Day {day}  |  {date}")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if args.reset:
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    main(overwrite=args.overwrite)
