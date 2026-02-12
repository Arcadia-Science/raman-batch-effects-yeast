import argparse
import shutil
from pathlib import Path

import arcadia_pycolor as apc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA

from raman_batch_effects import loaders, utils
from raman_batch_effects.cache import cache
from raman_batch_effects.scripts.config import YeastConfig, get_output_dir

apc.mpl.setup()

CONFIG = YeastConfig()
OUTPUT_DIR = get_output_dir(Path(__file__).stem)


def _plot_embedding_colored_by_variable(ax, embedding: np.ndarray, color_values: np.ndarray):
    """Plot 2D embedding colored by a categorical variable."""
    unique_values = np.unique(color_values)
    unique_values = sorted([v for v in unique_values if pd.notna(v)])

    colors = apc.palettes.all_palettes[0][: len(unique_values)]

    for value, color in zip(unique_values, colors, strict=False):
        mask = color_values == value
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            color=utils.lighten_hex_color(color),
            edgecolor=utils.darken_hex_color(color),
            s=20,
            alpha=0.7,
            linewidths=0.7,
            label=str(value),
        )

    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_embedding_grid(
    dataset,
    output_prefix: str,
    embedding_type: str,
    color_columns: list[str],
    date_names: list[str],
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    overwrite: bool = False,
):
    """
    Plot 2D embeddings for different dates colored by different variables in a 2D subplot grid.

    Rows of the subplot grid correspond to dates (august-2025, november-2025).
    Columns of the subplot grid correspond to coloring variables.

    Arguments:
        dataset: RamanDataset to analyze.
        output_prefix: Prefix for output filename (e.g., "uncorrected", "corrected").
        embedding_type: Type of embedding to use ("umap" or "pca").
        color_columns: List of metadata columns to use for coloring.
        date_names: List of date names to analyze.
        n_neighbors: Number of neighbors for UMAP (ignored for PCA).
        min_dist: Minimum distance for UMAP (ignored for PCA).
        random_state: Random state for reproducibility.
        overwrite: Whether to overwrite existing figures.
    """
    num_rows = len(date_names)
    num_cols = len(color_columns)

    fig_width = 4 * num_cols
    fig_height = 4 * num_rows
    _fig, axs = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))

    if num_rows == 1 and num_cols == 1:
        axs = np.array([[axs]])
    elif num_rows == 1:
        axs = axs.reshape(1, -1)
    elif num_cols == 1:
        axs = axs.reshape(-1, 1)

    if embedding_type == "umap":
        xlabel = "UMAP 1"
        ylabel = "UMAP 2"
    elif embedding_type == "pca":
        xlabel = "PC1"
        ylabel = "PC2"
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")

    for row_idx, date_name in enumerate(date_names):
        dataset_subset = dataset.filter(date=date_name)
        X, labels_df = dataset_subset.to_matrix()

        if embedding_type == "umap":
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=random_state,
                n_components=2,
            )
        else:
            reducer = PCA(n_components=2, random_state=random_state)

        embedding = reducer.fit_transform(X)

        for col_idx, color_column in enumerate(color_columns):
            ax = axs[row_idx, col_idx]

            color_values = labels_df[color_column].values
            error_message = None

            if len(np.unique(color_values)) < 2:
                error_message = f"Error:\nLess than two unique values in {color_column}"
            else:
                try:
                    _plot_embedding_colored_by_variable(ax, embedding, color_values)
                except Exception as e:
                    error_message = f"Uncaught error:\n{str(e)[:50]}\n{str(e)[50:100]}"

            if error_message is not None:
                ax.text(
                    0.5,
                    0.5,
                    error_message[:50],
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                )
                ax.axis("off")
            else:
                if col_idx == 0:
                    ax.set_ylabel(ylabel, fontsize=12)

                if row_idx == num_rows - 1:
                    ax.set_xlabel(xlabel, fontsize=12)

            if row_idx == 0:
                ax.set_title(f"Colored by `{color_column}`\n{date_name}", fontsize=12)
            else:
                ax.set_title(date_name, fontsize=12)

            ax.tick_params(labelsize=10)

    plt.tight_layout()
    figure_filepath = OUTPUT_DIR / f"{output_prefix}--{embedding_type.upper()}-embeddings.png"
    utils.save_figure(figure_filepath, overwrite=overwrite)


def main(overwrite: bool = False):
    """
    Generate UMAP and PCA embeddings for yeast datasets and visualize them,
    colored by different variables.

    The results are visualized as a 2D subplot grid where:
    - Rows correspond to dates (august-2025, november-2025)
    - Columns correspond to coloring variables (metadata columns)

    Creates separate PNG files for UMAP and PCA embeddings, for both
    uncorrected and corrected datasets.
    """
    datasets, _ = loaders.load_and_process_spectra(CONFIG.data_dirpath, CONFIG.crop_region)

    color_columns = [
        "day",
        "species",
        "strain",
    ]
    date_names = [
        "august-2025",
    ]

    for dataset_type in ["uncorrected", "corrected"]:
        dataset = getattr(datasets, dataset_type)

        for embedding_type in ["umap", "pca"]:
            print(f"Generating {embedding_type} embeddings for {dataset_type} dataset...")
            plot_embedding_grid(
                dataset=dataset,
                output_prefix=dataset_type,
                embedding_type=embedding_type,
                color_columns=color_columns,
                date_names=date_names,
                overwrite=overwrite,
                min_dist=0.1,
                n_neighbors=15,
            )


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
