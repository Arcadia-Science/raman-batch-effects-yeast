import colorsys
from pathlib import Path

import matplotlib.pyplot as plt


def lighten_hex_color(hex_color: str, amount: float = 0.3) -> str:
    hex_color = hex_color.lstrip("#")
    red, green, blue = [int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]

    # Convert to HLS to modify lightness
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)

    # increase lightness toward white
    lightness = 1 - (1 - lightness) * (1 - amount)

    red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)

    return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"


def darken_hex_color(hex_color: str, amount: float = 0.3) -> str:
    hex_color = hex_color.lstrip("#")
    red, green, blue = [int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]

    # Convert to HLS to modify lightness
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)

    # decrease lightness toward black
    lightness = lightness * (1 - amount)

    red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)

    return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"


def save_figure(
    filepath: str | Path,
    dpi: int = 300,
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
    dpi : int
        Resolution in dots per inch.
    bbox_inches : str
        Bounding box setting for the saved figure.
    verbose : bool
        Whether to print a message when saving the figure.
    overwrite : bool
        Whether to overwrite existing files. If False and file exists, skips saving.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite and filepath.exists():
        if verbose:
            print(f"Skipping '{filepath}' (already exists)")
        plt.close()
        return

    plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, facecolor="white")
    plt.close()

    if verbose:
        print(f"Saved figure to '{filepath}'")


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
