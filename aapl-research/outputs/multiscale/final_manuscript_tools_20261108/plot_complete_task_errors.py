"""Plot all twenty completed tasks without selecting favorable comparisons."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

MODELS = ("drift", "seasonal", "ridge", "mlp", "gru", "lstm", "xpatch")
LABELS = ("Drift", "Seasonal", "Ridge", "MLP", "GRU", "LSTM", "xPatch")
TASKS = (("annual", 12), ("quarterly", 3), ("monthly", 21),
         ("hourly", 1), ("hourly", 4), ("hourly", 8),
         ("minute", 5), ("minute", 15), ("minute", 30), ("minute", 60))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.metrics.open(encoding="utf8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    lookup = {(r["profile"], int(r["horizon"]), r["target"], r["model"]): r for r in rows}
    if len(rows) != len(lookup) or len(rows) != 160 or any(r["completed_folds"] != "5" or r["missing_folds"] for r in rows):
        raise ValueError("Expected eight models for all twenty complete five-fold tasks")
    panels, cells = [], []
    for target in ("future_close", "future_log_return"):
        matrix = []
        for profile, horizon in TASKS:
            baseline = float(lookup[profile, horizon, target, "last_close"]["endpoint_mse"])
            if not math.isfinite(baseline) or baseline <= 0:
                raise ValueError("Undefined baseline MSE")
            values = []
            for model in MODELS:
                record = lookup[profile, horizon, target, model]
                ratio = float(record["endpoint_mse"]) / baseline
                if not math.isfinite(ratio) or ratio <= 0 or not math.isclose(
                        1 - ratio, float(record["relative_improvement"]), abs_tol=1e-12):
                    raise ValueError("Error ratio differs from source")
                values.append(ratio)
                cells.append({"profile": profile, "horizon": horizon, "target": target,
                              "model": model, "endpoint_mse_ratio": ratio, "origins": int(record["origins"])})
            matrix.append(values)
        panels.append(np.asarray(matrix))
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 6.15), constrained_layout=True)
    lo = min(math.log10(0.5), min(float(np.log10(p).min()) for p in panels))
    hi = max(3.5, max(float(np.log10(p).max()) for p in panels))
    norm = TwoSlopeNorm(vmin=lo, vcenter=0, vmax=hi)
    for ax, ratios, title in zip(axes, panels, ("Nominal close price", "Cumulative log return"), strict=True):
        image = ax.imshow(np.log10(ratios), norm=norm, cmap="RdBu_r", aspect="auto")
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xticks(range(len(MODELS)), LABELS, rotation=30, ha="right")
        ax.set_yticks(range(len(TASKS)), [p.title() + f" H{h}" for p, h in TASKS])
        ax.tick_params(axis="both", length=0)
        ax.set_xticks(np.arange(-0.5, len(MODELS)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(TASKS)), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        for (y, x), value in np.ndenumerate(ratios):
            level = norm(math.log10(value))
            text = f"{value:.3f}" if 0.9 < value < 1.1 else f"{value:.2f}" if value < 100 else f"{value:.0f}"
            ax.text(x, y, text, ha="center", va="center", fontsize=8,
                    color="white" if level < 0.18 or level > 0.82 else "#181818")
        for spine in ax.spines.values():
            spine.set_visible(False)
    ticks = [0.5, 1, 2, 10, 100, 1000]
    bar = fig.colorbar(image, ax=axes, fraction=0.032, pad=0.025, ticks=np.log10(ticks), shrink=0.9)
    bar.ax.set_yticklabels([str(v) for v in ticks])
    bar.set_label("Endpoint MSE / persistence (log color scale)", labelpad=10)
    fig.suptitle("Historical development: twenty complete five-fold tasks", fontsize=13)
    paths = []
    for extension in ("png", "pdf", "svg"):
        path = args.output / ("completed_task_error_ratios." + extension)
        fig.savefig(path, dpi=220, facecolor="white")
        paths.append(path)
    plt.close(fig)
    values = args.output / "completed_task_error_ratios.csv"
    with values.open("x", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    audit = {"status": "all_twenty_complete_task_error_ratios_plotted", "created_at_utc": datetime.now(timezone.utc).isoformat(),
             "source_metrics": str(args.metrics.resolve()), "source_metrics_sha256": digest(args.metrics),
             "script_sha256": digest(__file__), "complete_tasks": 20, "completed_folds": 100, "cells": len(cells),
             "color": "shared logarithmic error-ratio scale centered at one", "log_color_limits": [lo, hi],
             "artifacts": {p.name: digest(p) for p in paths + [values]}, "independent_test": False,
             "Q3_values_read": False, "limitations": ["Exposed historical development; one core training seed.",
             "Annual task uses the shorter-history variant.", "Ratios below one are descriptive, not independent confirmation."]}
    (args.output / "FIGURE_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"status": audit["status"], "cells": len(cells), "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
