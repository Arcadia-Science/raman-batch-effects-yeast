"""
Copy publication figures from the script output directory to figs/ directory.
"""

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).parent
FIGS_DIR = REPO_ROOT / "figs"


def get_latest_date_dir() -> str | None:
    """Find the most recent date directory in output/."""
    output_root = REPO_ROOT / "output"
    if not output_root.exists():
        return None

    # Get all subdirectories that look like dates (YYYY-MM-DD format)
    date_dirs = [
        d.name
        for d in output_root.iterdir()
        if d.is_dir() and len(d.name) == 10 and d.name[4] == "-"
    ]

    if not date_dirs:
        return None

    # Sort and return the most recent
    return sorted(date_dirs)[-1]


def get_figure_mappings(output_dir: Path) -> list[tuple[Path, str]]:
    """Get the list of figure mappings from source to destination."""
    return [
        (
            output_dir / "plot_spectra" / "mean-spectra-by-strain-by-species--august-2025.pdf",
            "fig1.pdf",
        ),
        (
            output_dir
            / "plot_cross_validation"
            / "cf--kfold-cv--strain-prediction--uncorrected.pdf",
            "fig2.pdf",
        ),
        (
            output_dir
            / "plot_cross_validation"
            / "cf--lodo-cv--strain-prediction--uncorrected.pdf",
            "fig3.pdf",
        ),
        (
            output_dir / "plot_cross_validation" / "cf--loso-cv--day-prediction--uncorrected.pdf",
            "fig4a.pdf",
        ),
        (
            output_dir / "plot_cross_validation" / "cf--loso-cv--day-prediction--corrected.pdf",
            "fig4b.pdf",
        ),
        (
            output_dir / "plot_cross_validation" / "cf--lodo-cv--strain-prediction--corrected.pdf",
            "fig5.pdf",
        ),
        (
            output_dir / "plot_cross_validation" / "cf--lodo-cv--species-prediction--corrected.pdf",
            "fig6.pdf",
        ),
    ]


def main(date: str, overwrite: bool):
    """Copy figures to figs/ directory."""
    # Handle 'latest' option
    if date == "latest":
        date = get_latest_date_dir()
        if date is None:
            print("Error: No date directories found in output/")
            return
        print(f"Using latest date directory: {date}")

    output_dir = REPO_ROOT / "output" / date

    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        return

    # Create figs directory if it doesn't exist
    FIGS_DIR.mkdir(exist_ok=True)
    print(f"Created/verified figs directory: {FIGS_DIR}")
    print(f"Source directory: {output_dir}\n")

    figure_mappings = get_figure_mappings(output_dir)

    # Copy each figure
    for source, dest_name in figure_mappings:
        dest = FIGS_DIR / dest_name

        if not source.exists():
            print(f"⚠️  Warning: Source file not found: {source}")
            continue

        if dest.exists() and not overwrite:
            print(f"⊘ Skipped {dest_name} (already exists, use --overwrite to replace)")
            continue

        shutil.copy2(source, dest)
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"✓ Copied {dest_name} ({size_mb:.1f} MB)")

    print(f"\nAll figures copied to {FIGS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="Date subdirectory in output/ to copy from (use 'latest' for most recent)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in figs/ directory",
    )
    args = parser.parse_args()

    main(date=args.date, overwrite=args.overwrite)
