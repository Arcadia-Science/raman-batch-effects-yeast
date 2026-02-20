import argparse
import shutil
from pathlib import Path

from raman_batch_effects import loaders
from raman_batch_effects.cache import cache
from raman_batch_effects.scripts.config import YeastConfig, get_output_dir

CONFIG = YeastConfig()
OUTPUT_DIR = get_output_dir(Path(__file__).stem)

AUGUST_2025 = "august-2025"


def count_spectra(dataset, date: str) -> str:
    """
    Return a formatted string with counts of spectra per strain and day.
    """
    _, labels = dataset.filter(date=date).to_matrix()

    lines = []

    # Total count
    lines.append(f"Total spectra: {len(labels)}")
    lines.append("")

    # Counts per strain
    lines.append("Counts per strain:")
    for strain in sorted(labels.strain.unique()):
        count = (labels.strain == strain).sum()
        lines.append(f"  {strain:>12s}: {count}")
    lines.append("")

    # Counts per day
    lines.append("Counts per day:")
    for day in sorted(labels.day.unique()):
        count = (labels.day == day).sum()
        lines.append(f"  Day {day}: {count}")
    lines.append("")

    # Counts per strain and day
    lines.append("Counts per strain and day:")
    days = sorted(labels.day.unique())
    header = f"  {'strain':>12s}" + "".join(f"  Day {d}" for d in days)
    lines.append(header)
    for strain in sorted(labels.strain.unique()):
        row = f"  {strain:>12s}"
        for day in days:
            count = ((labels.strain == strain) & (labels.day == day)).sum()
            row += f"  {count:>5d}"
        lines.append(row)

    return "\n".join(lines)


def main(overwrite: bool = False):
    """
    Print and save counts of spectra per strain and day
    for the raw dataset and the final corrected dataset.
    """
    datasets, _ = loaders.load_and_process_spectra(CONFIG.data_dirpath, CONFIG.crop_region)

    date_to_print = AUGUST_2025

    sections = [
        ("Raw dataset", datasets.raw),
        ("Final corrected dataset", datasets.corrected),
    ]

    output_lines = []
    for section_name, dataset in sections:
        header = f"=== {section_name} ({date_to_print}) ==="
        counts = count_spectra(dataset, date=date_to_print)
        section_text = f"{header}\n{counts}\n"
        output_lines.append(section_text)
        print(section_text)

    # Save to file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_filepath = OUTPUT_DIR / f"spectra-counts--{date_to_print}.txt"

    if output_filepath.exists() and not overwrite:
        print(f"Skipping '{output_filepath}' (already exists)")
        return

    output_filepath.write_text("\n".join(output_lines))
    print(f"Saved counts to '{output_filepath}'")


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
