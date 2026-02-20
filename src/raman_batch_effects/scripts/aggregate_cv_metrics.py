"""Aggregate per-experiment YAML metrics files into a single summary YAML.

Usage:
    python -m raman_batch_effects.scripts.aggregate_cv_metrics
"""

import glob
import statistics
import sys
from pathlib import Path

import yaml

from raman_batch_effects.scripts.config import get_output_dir

OUTPUT_DIR = get_output_dir("plot_cross_validation")


def _fmt(values):
    """Format a list of per-fold values as 'median (min, max)'."""
    if not values:
        return "n/a"
    med = statistics.median(values)
    return f"{med:.4f} ({min(values):.4f}, {max(values):.4f})"


def main():
    yaml_files = sorted(glob.glob(str(OUTPUT_DIR / "metrics--*.yaml")))
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
        # Extract "corrected" or "uncorrected" from the filename.
        name = row["source_file"]
        if name.endswith("--corrected"):
            dataset = "corrected"
        elif name.endswith("--uncorrected"):
            dataset = "uncorrected"
        else:
            dataset = "unknown"

        metrics = row.get("per_fold_metrics", {})
        summary.append({
            "task": row["task"],
            "dataset": dataset,
            "cv_strategy": row["cv_strategy"],
            "n_folds": row["n_folds"],
            "n_classes": row["n_classes"],
            "accuracy": _fmt(metrics.get("accuracy", [])),
            "mcc": _fmt(metrics.get("mcc", [])),
        })

    output_path = OUTPUT_DIR / "cv_metrics_summary.yaml"
    output_path.write_text(yaml.dump(summary, sort_keys=False, default_flow_style=False))
    print(f"Wrote {len(summary)} entries to {output_path}")


if __name__ == "__main__":
    main()
