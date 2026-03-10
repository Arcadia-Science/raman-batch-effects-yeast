"""Aggregate per-experiment YAML metrics files into a single summary YAML.

Usage:
    python -m raman_batch_effects.scripts.aggregate_cv_metrics
"""

import csv
import glob
import statistics
import sys
from pathlib import Path

import arcadia_pycolor as apc
import matplotlib.pyplot as plt
import numpy as np
import yaml

from raman_batch_effects import plotting
from raman_batch_effects.scripts.config import get_output_dir

apc.mpl.setup()

INPUT_DIR = get_output_dir("plot_cross_validation")
OUTPUT_DIR = get_output_dir(Path(__file__).stem)


def _sigfig(x, n=2):
    """Round x to n significant figures."""
    if x == 0:
        return 0.0
    return round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1))


def _fmt_val(x):
    """Format a value rounded to 2 significant figures, with at least 2 decimal places."""
    rounded = _sigfig(x)
    # Determine decimal places needed: at least 2, but enough to show 2 sig figs.
    if rounded == 0:
        return "0.00"
    decimal_places = max(2, -int(np.floor(np.log10(abs(rounded)))) + 1)
    return f"{rounded:.{decimal_places}f}"


def _fmt(values):
    """Format a list of per-fold values as 'median (min, max)' with 2 significant figures."""
    if not values:
        return "n/a"
    med = statistics.median(values)
    lo = min(values)
    hi = max(values)
    return f"{_fmt_val(med)} ({_fmt_val(lo)}, {_fmt_val(hi)})"


def _fmt_plot(values):
    """Like _fmt, but with the median in bold for matplotlib rendering."""
    if not values:
        return "n/a"
    med = statistics.median(values)
    lo = min(values)
    hi = max(values)
    return rf"$\mathbf{{{_fmt_val(med)}}}$ ({_fmt_val(lo)}, {_fmt_val(hi)})"


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
        raw_mcc = metrics.get("mcc", [])
        summary.append(
            {
                "task": row["task"],
                "model": model,
                "dataset": dataset,
                "cv_strategy": row["cv_strategy"],
                "n_folds": row["n_folds"],
                "n_classes": row["n_classes"],
                "accuracy": _fmt(metrics.get("accuracy", [])),
                "mcc": _fmt(raw_mcc),
                "_raw_mcc": raw_mcc,
            }
        )

    output_path = OUTPUT_DIR / "cv_metrics_summary.yaml"
    yaml_summary = [{k: v for k, v in e.items() if not k.startswith("_")} for e in summary]
    output_path.write_text(yaml.dump(yaml_summary, sort_keys=False, default_flow_style=False))
    print(f"Wrote {len(yaml_summary)} entries to {output_path}")

    # Build a pivot table: rows = model/dataset variants, columns = tasks.
    # Store both formatted MCC string and raw median for heatmap coloring.
    # Explicit ordering for tasks (columns) and model/dataset (rows).
    TASK_ORDER = [
        "Strain prediction (5-fold cross-validation)",
        "Strain prediction (Leave-one-day-out cross-validation)",
        "Day/plate prediction (Leave-one-strain-out cross-validation)",
        "Species prediction (Leave-one-day-out cross-validation)",
    ]
    TASK_LABELS = {
        "Strain prediction (5-fold cross-validation)": "Strain (5-fold CV)",
        "Strain prediction (Leave-one-day-out cross-validation)": "Strain (LODO CV)",
        "Day/plate prediction (Leave-one-strain-out cross-validation)": "Plate (LOSO CV)",
        "Species prediction (Leave-one-day-out cross-validation)": "Species (LODO CV)",
    }
    all_tasks = {f"{e['task']} ({e['cv_strategy']})": None for e in summary}
    col_keys = [t for t in TASK_ORDER if t in all_tasks]

    VARIANT_ORDER = [
        "rf / uncorrected",
        "svc / uncorrected",
        "rf / corrected-lmm",
        "svc / corrected-lmm",
        "rf / corrected-combat",
        "svc / corrected-combat",
    ]
    VARIANT_LABELS = {
        "rf / uncorrected": "Uncorrected | RF",
        "svc / uncorrected": "Uncorrected | SVC",
        "rf / corrected-lmm": "Corrected (LMM) | RF",
        "svc / corrected-lmm": "Corrected (LMM) | SVC",
        "rf / corrected-combat": "Corrected (ComBat) | RF",
        "svc / corrected-combat": "Corrected (ComBat) | SVC",
    }
    all_variants = {f"{e['model']} / {e['dataset']}" for e in summary}
    row_keys = [v for v in VARIANT_ORDER if v in all_variants]

    pivot_fmt = {}
    pivot_plot = {}
    pivot_median = {}
    for entry in summary:
        task_key = f"{entry['task']} ({entry['cv_strategy']})"
        variant_key = f"{entry['model']} / {entry['dataset']}"
        raw = entry.get("_raw_mcc", [])
        pivot_fmt[(variant_key, task_key)] = entry["mcc"]
        pivot_plot[(variant_key, task_key)] = _fmt_plot(raw)
        pivot_median[(variant_key, task_key)] = statistics.median(raw) if raw else None

    # Write pivot CSV.
    csv_path = OUTPUT_DIR / "cv_mcc_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model / dataset"] + [TASK_LABELS.get(c, c) for c in col_keys])
        for row_key in row_keys:
            writer.writerow(
                [VARIANT_LABELS.get(row_key, row_key)]
                + [pivot_fmt.get((row_key, col), "") for col in col_keys]
            )
    print(f"Wrote pivot CSV to {csv_path}")

    # Plot heatmap.
    n_rows = len(row_keys)
    n_cols = len(col_keys)
    data = np.full((n_rows, n_cols), np.nan)
    for i, rk in enumerate(row_keys):
        for j, ck in enumerate(col_keys):
            val = pivot_median.get((rk, ck))
            if val is not None:
                data[i, j] = val

    cmap = (apc.gradients.purple_green).to_mpl_cmap()

    fig, ax = plt.subplots(figsize=(len(col_keys) * 2.8 + 2, len(row_keys) * 0.8 + 1.5))

    x = np.arange(n_cols + 1) - 0.5
    y = np.arange(n_rows + 1) - 0.5
    X_mesh, Y_mesh = np.meshgrid(x, y)

    im = ax.pcolormesh(
        X_mesh,
        Y_mesh,
        data,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        shading="flat",
        edgecolors="white",
        linewidth=1,
    )

    for i, rk in enumerate(row_keys):
        for j, ck in enumerate(col_keys):
            text = pivot_plot.get((rk, ck), "")
            if text:
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontfamily="Suisse Int'l Mono",
                    color="black",
                )

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(
        [TASK_LABELS.get(ck, ck) for ck in col_keys], rotation=45, ha="right", fontsize=12
    )
    ax.set_yticklabels([VARIANT_LABELS.get(rk, rk) for rk in row_keys], fontsize=12)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.invert_yaxis()

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Median MCC", rotation=270, labelpad=15)

    plt.tight_layout()
    plotting.utils.save_figure(OUTPUT_DIR / "cv_mcc_heatmap.pdf", overwrite=True)
    print(f"Wrote heatmap to {OUTPUT_DIR / 'cv_mcc_heatmap.pdf'}")


if __name__ == "__main__":
    main()
