from collections.abc import Callable
from dataclasses import dataclass

import arcadia_pycolor as apc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import auc

from raman_batch_effects import utils
from raman_batch_effects.cross_validation import calc_confusion_matrix_lobo, calc_roc_lobo
from raman_batch_effects.datasets import RamanDataset

DEFAULT_COLOR_PALETTE = apc.palettes.all_palettes[0]


def plot_spectrum_with_other_values(wavenumbers, spectrum, other_values, other_label, ax=None):
    """
    Plot a spectrum with other values on a twin y-axis.

    Arguments:
        wavenumbers: Wavenumber values for x-axis.
        spectrum: Spectrum intensity values.
        other_values: Other values to plot on right y-axis (e.g., feature importance).
        other_label: Label for the right y-axis.
        ax: Optional matplotlib axis to plot on.
    """
    if ax is None:
        plt.figure(figsize=(15, 5))
        ax = plt.gca()

    ax_right = ax.twinx()

    ax.plot(wavenumbers, spectrum, lw=1, color="black")
    ax.set_ylabel("Mean Intensity")

    ax_right.plot(wavenumbers, other_values, lw=1, color="red")
    ax_right.set_ylabel(other_label)

    ax.set_xlabel("Wavenumber (cm$^{-1}$)")

    apc.mpl.style_plot(ax, monospaced_axes="both")
    apc.mpl.style_plot(ax_right, monospaced_axes="both")

    # Ensure the right axis is visible.
    ax_right.spines["right"].set_visible(True)


def plot_roc_curve(fpr, tpr, color=DEFAULT_COLOR_PALETTE[0], ax=None):
    """
    Plot a single ROC curve.

    Arguments:
        fpr: False positive rates.
        tpr: True positive rates.
        color: Color for the ROC curve.
        ax: Optional matplotlib axis to plot on.
    """
    if ax is None:
        plt.figure()
        ax = plt.gca()

    ax.plot(
        fpr,
        tpr,
        lw=1,
        label=f"Area = {auc(fpr, tpr):.2f}",
        color=color,
    )
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

    apc.mpl.style_plot(axes=ax, monospaced_axes="both")


def plot_roc_curves(fprs, tprs, labels, colors=DEFAULT_COLOR_PALETTE[:2]):
    """
    Plot one or more ROC curves on the same axes.

    Arguments:
        fprs: List of false positive rate arrays.
        tprs: List of true positive rate arrays.
        labels: List of labels for each curve.
        colors: List of colors for each curve.
    """
    plt.figure()

    for fpr, tpr, label, color in zip(fprs, tprs, labels, colors, strict=False):
        plt.plot(
            fpr,
            tpr,
            lw=1,
            label=f"{label} (AUC = {auc(fpr, tpr):.2f})",
            color=color,
        )

    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")

    apc.mpl.style_plot(monospaced_axes="both")


def plot_confusion_matrix(
    confusion_matrix,
    labels,
    show_cell_counts=True,
    ax=None,
    figsize=None,
    show_colorbar=True,
    ovr_auc=None,
    cmap=None,
):
    """
    Plot a confusion matrix as a heatmap.

    Arguments:
        confusion_matrix: Confusion matrix as numpy array.
        labels: List of class labels.
        show_cell_counts: If True, show counts and normalized rates in cells.
        ax: Optional matplotlib axis to plot on.
        figsize: Optional figure size tuple.
        show_colorbar: If True, show colorbar.
        ovr_auc: Optional One-vs-Rest AUC score to display on the plot.
    """

    num_labels = len(labels)

    if ax is None:
        plt.figure(figsize=(figsize or (6, 6)))
        ax = plt.gca()

    cm_normalized = confusion_matrix.astype("float") / confusion_matrix.sum(axis=1)[:, np.newaxis]

    im = ax.imshow(
        cm_normalized,
        cmap=cmap or (apc.gradients.reds.reverse()).to_mpl_cmap(),
        vmin=0,
        vmax=1,
        aspect="auto",
    )

    for i in range(num_labels):
        for j in range(num_labels):
            if not show_cell_counts:
                continue
            ax.text(
                j,
                i,
                f"{confusion_matrix[i, j]}\n({cm_normalized[i, j]:.2f})",
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )

    ax.set_xticks(np.arange(num_labels))
    ax.set_yticks(np.arange(num_labels))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Normalized Rate", rotation=270, labelpad=20)

    if ovr_auc is not None:
        ax.text(
            0.98,
            1.00,
            f"AUC = {ovr_auc:.2f}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="bottom",
            horizontalalignment="right",
        )


def plot_confusion_matrices_lobo(confusion_matrices, unique_labels, figsize=None, cmap="Blues"):
    """
    Plot confusion matrices for each batch from leave-one-batch-out CV.

    Args:
        confusion_matrices: Dictionary mapping batch labels to confusion matrices
        unique_labels: Sorted list of unique class labels
        figsize: Optional tuple (width, height) for figure size. If None, automatically determined.
        cmap: Colormap for the heatmap

    Returns:
        fig: matplotlib Figure object
        axes: Array of matplotlib Axes objects
    """

    n_batches = len(confusion_matrices)
    n_cols = min(4, n_batches)
    n_rows = (n_batches + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (n_cols * 4, n_rows * 3.5)

    _, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for idx, (batch_label, cm) in enumerate(sorted(confusion_matrices.items())):
        ax = axes[idx]

        # Plot the confusion matrix
        ax.imshow(cm, interpolation="nearest", cmap=cmap)
        ax.set_title(f"{batch_label}", fontsize=14)

        # Set tick marks
        tick_marks = np.arange(len(unique_labels))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(unique_labels, rotation=45, ha="right", fontsize=12)
        ax.set_yticklabels(unique_labels, fontsize=12)

        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10,
                )

    # Hide unused subplots
    for idx in range(n_batches, len(axes)):
        axes[idx].axis("off")

    # Hide axis labels on all but the first plot
    for idx in range(n_batches):
        if idx != 0:
            axes[idx].set_ylabel("")
            axes[idx].set_xlabel("")

    plt.tight_layout()


def _identify_continuous_segments(wavenumbers):
    """
    Identify continuous segments in wavenumbers to avoid plotting across gaps.

    Arguments:
        wavenumbers: Array of wavenumber values.

    Returns:
        List of (start_idx, end_idx) tuples for each continuous segment.
    """
    continuous_segments = []
    if len(wavenumbers) == 0:
        return continuous_segments

    min_segment_size = np.median(np.diff(wavenumbers)) * 2
    segment_start = 0
    for idx in range(1, len(wavenumbers)):
        if wavenumbers[idx] - wavenumbers[idx - 1] > min_segment_size:
            continuous_segments.append((segment_start, idx))
            segment_start = idx
    continuous_segments.append((segment_start, len(wavenumbers)))

    return continuous_segments


def _plot_mean_spectra(
    ax,
    X,
    Y,
    wavenumbers,
    continuous_segments,
    positive_class_label=None,
    negative_class_label=None,
):
    """
    Plot mean spectra for each class on the given axis.

    Arguments:
        ax: Matplotlib axis to plot on.
        X: Data matrix (samples × features).
        Y: Target labels.
        wavenumbers: Wavenumber array corresponding to features in X.
        continuous_segments: List of (start_idx, end_idx) tuples for continuous segments.
        positive_class_label: Optional label for positive class (for boolean Y).
        negative_class_label: Optional label for negative class (for boolean Y).
    """
    colors = apc.palettes.all_palettes[0]
    for ind, y_value in enumerate(sorted(np.unique(Y))):
        y_mask = Y == y_value
        mean_spectrum = X[y_mask].mean(axis=0)

        if y_value in [True, False]:
            spectrum_label = positive_class_label if y_value else negative_class_label
        else:
            spectrum_label = y_value

        for seg_start, seg_end in continuous_segments:
            ax.plot(
                wavenumbers[seg_start:seg_end],
                mean_spectrum[seg_start:seg_end],
                lw=1,
                color=colors[ind],
                label=(spectrum_label if seg_start == continuous_segments[0][0] else None),
            )

    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylabel("Mean Intensity")
    ax.set_xticklabels([])
    apc.mpl.style_plot(ax, monospaced_axes="both")


def _plot_feature_importances(ax, wavenumbers, importances, continuous_segments):
    """
    Plot feature importances on the given axis.

    Arguments:
        ax: Matplotlib axis to plot on.
        wavenumbers: Wavenumber array.
        importances: Feature importance values.
        continuous_segments: List of (start_idx, end_idx) tuples for continuous segments.
    """
    for seg_start, seg_end in continuous_segments:
        ax.plot(
            wavenumbers[seg_start:seg_end],
            importances[seg_start:seg_end],
            lw=1,
            color="black",
        )
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Feature importance")
    apc.mpl.style_plot(ax, monospaced_axes="both")


def plot_lobo_cv_results(
    dataset: RamanDataset,
    y_column: str | Callable,
    batch_column: str | Callable,
    model: BaseEstimator,
    figsize: tuple[float, float] = (15, 5),
    force_confusion_matrix: bool = False,
    show_cell_counts: bool = False,
    cmap: str | None = None,
    wavenumber_regions: list[tuple[int, int]] | None = None,
    positive_class_label: str | None = None,
    negative_class_label: str | None = None,
    title: str | None = None,
):
    """
    Plot LOBO CV results with classifier performance and feature importance.

    Creates a figure with three subplots:
    - Left: ROC curve (for binary classification) or confusion matrix (for multi-class).
    - Right top: Mean spectra for each class.
    - Right bottom: Feature importances.

    Arguments:
        dataset: RamanDataset to plot results for.
        y_column: Either a string column name or a callable that accepts the labels DataFrame
            and returns target labels (e.g., lambda df: df.column == "value").
        batch_column: Either a string column name or a callable that accepts the labels DataFrame
            and returns batch labels for LOBO CV.
        model: Optional sklearn model to use for CV.
        figsize: Figure size as (width, height) tuple.
        force_confusion_matrix: If True, force the use of a confusion matrix,
            even if the classification is binary.
        show_cell_counts: If True, show counts in cells of the confusion matrix.
        cmap: Colormap for the confusion matrix.
        wavenumber_regions: Optional list of (min, max) tuples specifying wavenumber regions to use.
            E.g., [(500, 1000), (1500, 2000)]. If provided, crops X to only these regions.
        positive_class_label: Label for the positive class.
            Only used for binary classification.
        negative_class_label: Label for the negative class.
            Only used for binary classification.
        title: Optional title for the figure.
    """
    X, labels = dataset.to_matrix()
    wavenumbers = dataset.wavenumbers

    if wavenumber_regions is not None:
        mask = np.zeros(len(wavenumbers), dtype=bool)
        for min_wavenumber, max_wavenumber in wavenumber_regions:
            mask |= (wavenumbers >= min_wavenumber) & (wavenumbers <= max_wavenumber)
        X = X[:, mask]
        wavenumbers = wavenumbers[mask]

    if callable(y_column):
        Y = y_column(labels)
    else:
        Y = labels[y_column].values

    if callable(batch_column):
        batch_labels = batch_column(labels)
    else:
        batch_labels = labels[batch_column].values

    unique_values = np.unique(Y)
    is_binary = len(unique_values) == 2

    # Create figure with custom layout: left column (1 plot) and right column (2 stacked plots).
    fig = plt.figure(figsize=figsize)
    gridspec = fig.add_gridspec(2, 2, width_ratios=[1.1, 2], hspace=0.3, wspace=0.3)
    ax_left = fig.add_subplot(gridspec[:, 0])
    ax_right_top = fig.add_subplot(gridspec[0, 1])
    ax_right_bottom = fig.add_subplot(gridspec[1, 1])

    if is_binary and not force_confusion_matrix:
        # Binary classification: use ROC curve.
        Y_binary = (Y == unique_values[1]).astype(int)
        fpr, tpr, importances = calc_roc_lobo(X, Y_binary, batch_labels=batch_labels, model=model)
        plot_roc_curve(fpr, tpr, ax=ax_left)

    else:
        # Multi-class classification: use confusion matrix.
        confusion_matrix, unique_labels, ovr_auc, importances = calc_confusion_matrix_lobo(
            X, Y, batch_labels=batch_labels, model=model
        )
        plot_confusion_matrix(
            confusion_matrix,
            unique_labels,
            ax=ax_left,
            show_cell_counts=show_cell_counts,
            show_colorbar=False,
            ovr_auc=ovr_auc,
            cmap=cmap,
        )

    if importances is None or len(importances) == 0:
        importances = np.zeros(X.shape[1])

    continuous_segments = _identify_continuous_segments(wavenumbers)

    _plot_mean_spectra(
        ax_right_top,
        X,
        Y,
        wavenumbers,
        continuous_segments,
        positive_class_label=positive_class_label,
        negative_class_label=negative_class_label,
    )

    _plot_feature_importances(ax_right_bottom, wavenumbers, importances, continuous_segments)

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()


@dataclass
class LoboCvRowConfig:
    """
    Configuration for a single row in a multi-row LOBO CV plot produced by
    `plot_lobo_cv_results_multirow`.

    This matches the interface of `plot_lobo_cv_results` and allows the multirow
    function to handle data preparation internally.
    """

    dataset: RamanDataset
    y_column: str | Callable
    batch_column: str | Callable
    wavenumber_regions: list[tuple[int, int]] | None
    row_label: str


def plot_lobo_cv_results_multirow(
    row_configs: list[LoboCvRowConfig],
    model,
    figure_title: str,
    output_filepath,
    overwrite: bool = False,
):
    """
    Create a multi-row figure with LOBO CV results.
    This is like `plot_lobo_cv_results` but for multiple rows.

    Each row shows the same 3-subplot layout (ROC/confusion matrix, mean spectra,
    feature importances) for different datasets or conditions.

    Arguments:
        row_configs: List of LoboCvRowConfig objects, one per row.
        model: sklearn model to use for cross-validation.
        figure_title: Title for the entire figure.
        output_filepath: Path where the figure will be saved.
        overwrite: If True, overwrite existing output files.

    Returns:
        None. Saves the figure to output_filepath.
    """
    if output_filepath.exists() and not overwrite:
        print(f"Skipping {output_filepath.name} (file exists and overwrite=False)")
        return

    num_rows = len(row_configs)
    fig_height = 5 * num_rows
    fig = plt.figure(figsize=(15, fig_height))

    # Create subplot layout for entire figure using gridspec.
    # Leave extra space at top for row titles and overall figure title.
    gs = fig.add_gridspec(
        num_rows,
        2,
        width_ratios=[1.1, 2],
        hspace=0.4,
        wspace=0.3,
        top=0.90,
        bottom=0.05,
    )

    for row_idx, config in enumerate(row_configs):
        # Prepare data for this row.
        X, labels = config.dataset.to_matrix()
        wavenumbers = config.dataset.wavenumbers

        # Apply wavenumber cropping if specified.
        if config.wavenumber_regions is not None:
            mask = np.zeros(len(wavenumbers), dtype=bool)
            for min_wn, max_wn in config.wavenumber_regions:
                mask |= (wavenumbers >= min_wn) & (wavenumbers <= max_wn)
            X = X[:, mask]
            wavenumbers = wavenumbers[mask]

        if callable(config.y_column):
            Y = config.y_column(labels)
        else:
            Y = labels[config.y_column].values

        if callable(config.batch_column):
            batch_labels = config.batch_column(labels)
        else:
            batch_labels = labels[config.batch_column].values

        unique_values = np.unique(Y)
        is_binary = len(unique_values) == 2

        ax_left = fig.add_subplot(gs[row_idx, 0])

        gs_right = gs[row_idx, 1].subgridspec(2, 1, hspace=0.3)
        ax_right_top = fig.add_subplot(gs_right[0])
        ax_right_bottom = fig.add_subplot(gs_right[1])

        # Perform cross-validation and plot results.
        if is_binary:
            Y_binary = (Y == unique_values[1]).astype(int)
            fpr, tpr, importances = calc_roc_lobo(
                X, Y_binary, batch_labels=batch_labels, model=model
            )
            plot_roc_curve(fpr, tpr, ax=ax_left)
        else:
            confusion_matrix, unique_labels, ovr_auc, importances = calc_confusion_matrix_lobo(
                X, Y, batch_labels=batch_labels, model=model
            )
            plot_confusion_matrix(
                confusion_matrix,
                unique_labels,
                ax=ax_left,
                show_cell_counts=False,
                show_colorbar=False,
                ovr_auc=ovr_auc,
            )

        if importances is None or len(importances) == 0:
            importances = np.zeros(X.shape[1])

        continuous_segments = _identify_continuous_segments(wavenumbers)
        _plot_mean_spectra(ax_right_top, X, Y, wavenumbers, continuous_segments)
        _plot_feature_importances(ax_right_bottom, wavenumbers, importances, continuous_segments)

        # Add horizontal row title above the row, centered.
        # Get the bbox of the right subplot to determine vertical position.
        row_title_y = ax_right_top.get_position().y1 + 0.01
        fig.text(
            0.5,
            row_title_y,
            config.row_label,
            fontsize=13,
            verticalalignment="bottom",
            horizontalalignment="center",
            fontweight="bold",
            transform=fig.transFigure,
        )

    # Add overall figure title at the very top.
    plt.suptitle(figure_title, fontsize=16, y=0.96)
    plt.tight_layout()
    utils.save_figure(output_filepath, overwrite=overwrite, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Saved {output_filepath.name}")
