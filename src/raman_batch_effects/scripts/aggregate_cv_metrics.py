"""Aggregate per-experiment YAML metrics files into a single summary YAML.

Usage:
    python -m raman_batch_effects.scripts.aggregate_cv_metrics
"""

import csv
import glob
import statistics
import sys
from pathlib import Path

import yaml

from raman_batch_effects.scripts.config import get_output_dir

INPUT_DIR = get_output_dir("plot_cross_validation")
OUTPUT_DIR = get_output_dir(Path(__file__).stem)


def _fmt(values):
    """Format a list of per-fold values as 'median (min, max)'."""
    if not values:
        return "n/a"
    med = statistics.median(values)
    return f"{med:.4f} ({min(values):.4f}, {max(values):.4f})"


def main():
    yaml_files = sorted(glob.glob(str(INPUT_DIR / "metrics--*.yaml")))
    if not yaml_files:
        print(f"No metrics YAML files found in {OUTPUT_DIR}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for filepath in yaml_files:
        with open(filepath) as f:
            data = yaml.safe_load(f)
        data["source_file"] = Path(filepath).stem
        rows.append(data)

    summary = []
    for row in rows:
        # Extract model name and dataset from the filename suffix.
        # Filename format: metrics--<cv>--<task>--<model>--<dataset>
        name = row["source_file"]
        parts = name.split("--")
        dataset = (
            parts[-1]
            if parts[-1] in ("uncorrected", "corrected-lmm", "corrected-combat")
            else "unknown"
        )
        model = parts[-2] if len(parts) >= 2 else "unknown"

        metrics = row.get("per_fold_metrics", {})
        summary.append(
            {
                "task": row["task"],
                "model": model,
                "dataset": dataset,
                "cv_strategy": row["cv_strategy"],
                "n_folds": row["n_folds"],
                "n_classes": row["n_classes"],
                "accuracy": _fmt(metrics.get("accuracy", [])),
                "mcc": _fmt(metrics.get("mcc", [])),
            }
        )

    output_path = OUTPUT_DIR / "cv_metrics_summary.yaml"
    output_path.write_text(yaml.dump(summary, sort_keys=False, default_flow_style=False))
    print(f"Wrote {len(summary)} entries to {output_path}")

    # Build a pivot CSV: rows = task (with cv_strategy), columns = model/dataset variants.
    # Collect unique row keys and column keys in insertion order.
    row_keys = dict.fromkeys(f"{e['task']} ({e['cv_strategy']})" for e in summary)
    col_keys = dict.fromkeys(f"{e['model']} / {e['dataset']}" for e in summary)
    pivot = {}
    for entry in summary:
        row_key = f"{entry['task']} ({entry['cv_strategy']})"
        col_key = f"{entry['model']} / {entry['dataset']}"
        pivot[(row_key, col_key)] = entry["mcc"]

    csv_path = OUTPUT_DIR / "cv_mcc_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task"] + list(col_keys))
        for row_key in row_keys:
            writer.writerow([row_key] + [pivot.get((row_key, col), "") for col in col_keys])
    print(f"Wrote pivot CSV to {csv_path}")


if __name__ == "__main__":
    main()
