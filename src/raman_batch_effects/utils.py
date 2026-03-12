import colorsys
from pathlib import Path

import matplotlib.pyplot as plt


def lighten_hex_color(hex_color: str, amount: float = 0.3) -> str:
    hex_color = hex_color.lstrip("#")
    red, green, blue = [int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]

    # Convert to HLS to modify lightness
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)

    # Increase lightness toward white.
    lightness = 1 - (1 - lightness) * (1 - amount)

    red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)

    return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"


def darken_hex_color(hex_color: str, amount: float = 0.3) -> str:
    hex_color = hex_color.lstrip("#")
    red, green, blue = [int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]

    # Convert to HLS to modify lightness
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)

    # Decrease lightness toward black.
    lightness = lightness * (1 - amount)

    red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)

    return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"


def save_figure(
    filepath: str | Path,
    dpi: int | None = None,
    bbox_inches: str = "tight",
    verbose: bool = True,
    overwrite: bool = False,
) -> None:
    """
    Save the current matplotlib figure with consistent settings.

    Parameters
    ----------
    filepath : str or Path
        Path where the figure should be saved.
    dpi : int or None
        Resolution in dots per inch. If None, uses 150 for PDFs and 300 for raster formats.
    bbox_inches : str
        Bounding box setting for the saved figure.
    verbose : bool
        Whether to print a message when saving the figure.
    overwrite : bool
        Whether to overwrite existing files. If False and file exists, skips saving.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Build the list of paths to save: always the requested path,
    # plus a PNG sibling if the requested path is a PDF.
    paths = [filepath]
    if filepath.suffix.lower() == ".pdf":
        paths.append(filepath.with_suffix(".png"))

    all_exist = all(p.exists() for p in paths)
    if not overwrite and all_exist:
        if verbose:
            print(f"Skipping '{filepath.stem}' (already exists)")
        plt.close()
        return

    for path in paths:
        # Set DPI based on file extension if not explicitly provided
        file_dpi = dpi
        if file_dpi is None:
            if path.suffix.lower() == ".pdf":
                file_dpi = 150  # Lower DPI for vector PDFs to reduce file size
            else:
                file_dpi = 300  # Higher DPI for raster formats (PNG, JPG, etc.)

        # For PDFs, use settings that optimize for vector output
        if path.suffix.lower() == ".pdf":
            plt.savefig(
                path,
                dpi=file_dpi,
                bbox_inches=bbox_inches,
                facecolor="white",
                format="pdf",
                metadata={"Creator": "matplotlib", "Producer": "matplotlib"},
            )
        else:
            plt.savefig(path, dpi=file_dpi, bbox_inches=bbox_inches, facecolor="white")

        if verbose:
            print(f"  Saved '{path.name}'")

    plt.close()


def find_repo_root() -> Path:
    dirpath = Path(__file__).parent
    while True:
        if (dirpath / "pyproject.toml").exists():
            return dirpath
        dirpath = dirpath.parent
        if dirpath == Path("/"):
            raise ValueError("Repo root not found")


def get_data_dirpath() -> Path:
    return find_repo_root() / "data"
