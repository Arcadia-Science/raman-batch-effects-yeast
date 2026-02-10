import arcadia_pycolor as apc
import matplotlib.pyplot as plt

from raman_batch_effects.datasets import RamanDataset

LABEL_SEP = "  |  "


def plot_plate_layout(
    dataset: RamanDataset,
    day: int,
    date: str = None,
    ylim: tuple[float, float] = None,
    alpha: float = 0.1,
    figsize: tuple[float, float] = None,
):
    """
    Plot spectra in a 2D grid matching the 96-well plate layout.

    Creates a subplot for each well on the specified day, positioned to mirror
    the physical plate layout. Each subplot shows all spectra from that well
    overlaid in black with transparency.

    Args:
        dataset: RamanDataset containing the spectra to plot.
        day: Day number to plot (1, 2, or 3).
        date: Optional date filter ("august-2025" or "november-2025").
              If None, plots all dates for the given day.
        ylim: Optional y-axis limits as (ymin, ymax). If None, uses auto-scaling.
        alpha: Transparency for plotted spectra (0-1).
        figsize: Optional figure size as (width, height). If None, uses (16, 12).

    Example:
        >>> from raman_eda.yeast import loaders, plotting
        >>> dataset = loaders.load_yeast_spectra(data_dirpath)
        >>> plotting.plot_plate_layout(dataset, day=1, ylim=(-0.005, 0.01))
        >>> plotting.plot_plate_layout(dataset, day=1, date="november-2025")
    """
    if figsize is None:
        figsize = (16, 12)

    # Filter to the requested day (and optionally date)
    if date is not None:
        day_dataset = dataset.filter(day=day, date=date)
    else:
        day_dataset = dataset.filter(day=day)

    if len(day_dataset) == 0:
        print(f"Warning: no spectra found for day {day}" + (f", date {date}" if date else ""))
        return

    metadata = day_dataset.metadata

    # Get unique rows and columns from well IDs
    rows = sorted(metadata.well_id.apply(lambda x: x[0]).unique())
    cols = sorted(metadata.well_id.apply(lambda x: int(x[1:])).unique())

    # Create the figure and subplots
    fig, axes = plt.subplots(len(rows), len(cols), figsize=figsize, sharex=True, sharey=True)

    # Handle case where we have a 1D array of axes
    if len(rows) == 1 or len(cols) == 1:
        axes = axes.reshape(len(rows), len(cols))

    # Plot spectra for each well
    for row_idx, row in enumerate(rows):
        for col_idx, col in enumerate(cols):
            ax = axes[row_idx, col_idx]

            # Format well ID (e.g., "A1", "A4")
            well_id = f"{row}{col}"

            # Get all spectra for this well
            well_dataset = day_dataset.filter(well_id=well_id)

            if len(well_dataset) > 0:
                # Plot all spectra for this well
                for spectrum, _ in well_dataset:
                    ax.plot(
                        spectrum.spectral_axis,
                        spectrum.spectral_data,
                        lw=0.5,
                        alpha=alpha,
                        color="black",
                    )

                # Get strain name from metadata
                strain = well_dataset.metadata.iloc[0].strain
                title = LABEL_SEP.join([well_id, strain])
                ax.set_title(title, fontsize=14, pad=2)

                # Apply y-limits if specified
                if ylim is not None:
                    ax.set_ylim(*ylim)
            else:
                # Empty well - turn off axis
                ax.axis("off")

            # Style the plot
            ax.tick_params(labelsize=6)

            # Only show x-axis labels on bottom row
            if row_idx == len(rows) - 1:
                ax.set_xlabel("wavenumber (cm$^{-1}$)", fontsize=12)

            # Only show y-axis labels on leftmost column
            if col_idx == 0:
                ax.set_ylabel("Intensity", fontsize=12)

            apc.mpl.style_plot(ax, monospaced_axes="both")

    # Add title with date info if provided
    title = f"Day {day}"
    if date is not None:
        title += f" ({date})"
    fig.suptitle(title, fontsize=14, y=0.995)

    plt.tight_layout()
