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
            "mean-spectra.pdf",
        ),
        # Confusion matrices for strain prediction.
        (
            output_dir
            / "plot_cross_validation"
            / "cf--kfold-cv--strain-prediction--rf--uncorrected.pdf",
            "cf-kfold-strain.pdf",
        ),
        (
            output_dir
            / "plot_cross_validation"
            / "cf--lodo-cv--strain-prediction--rf--uncorrected.pdf",
            "cf-lodo-strain-uncorrected.pdf",
        ),
        (
            output_dir
            / "plot_cross_validation"
            / "cf--lodo-cv--strain-prediction--rf--corrected-lmm.pdf",
            "cf-lodo-strain-corrected.pdf",
        ),
        # Confusion matrices for day prediction.
        (
            output_dir
            / "plot_cross_validation"
            / "cf--loso-cv--day-prediction--rf--uncorrected.pdf",
            "cf-loso-plate-panel-a-uncorrected.pdf",
        ),
        (
            output_dir
            / "plot_cross_validation"
            / "cf--loso-cv--day-prediction--rf--corrected-lmm.pdf",
            "cf-loso-plate-panel-b-corrected.pdf",
        ),
        # Confusion matrices for species prediction.
        (
            output_dir
            / "plot_cross_validation"
            / "cf--lodo-cv--species-prediction--rf--uncorrected.pdf",
            "cf-lodo-species-panel-a-uncorrected.pdf",
        ),
        (
            output_dir
            / "plot_cross_validation"
            / "cf--lodo-cv--species-prediction--rf--corrected-lmm.pdf",
            "cf-lodo-species-panel-b-corrected.pdf",
        ),
        # ROC curve and feature importances for species prediction.
        (
            output_dir
            / "plot_cross_validation"
            / "cf--lodo-cv--species-prediction-roc-curve--rf--corrected-lmm.pdf",
            "cf-lodo-species-corrected-feature-importances.pdf",
        ),
        # Summary heatmap of MCC values.
        (
            output_dir / "aggregate_cv_metrics" / "cv_mcc_heatmap.pdf",
            "cv-mcc-heatmap.pdf",
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
        # Add date prefix to destination filename
        dest_name_with_date = f"{date}-{dest_name}"
        dest = FIGS_DIR / dest_name_with_date

        if not source.exists():
            print(f"⚠️  Warning: Source file not found: {source}")
            continue

        if dest.exists() and not overwrite:
            print(f"⊘ Skipped {dest_name_with_date} (already exists, use --overwrite to replace)")
            continue

        shutil.copy2(source, dest)
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"✓ Copied {dest_name_with_date} ({size_mb:.1f} MB)")

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
