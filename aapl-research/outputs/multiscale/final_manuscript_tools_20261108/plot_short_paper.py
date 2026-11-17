"""Plot all 140 compact-paper comparisons at their actual publication width."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from build_short_paper import TASKS, TARGETS, MODELS, MODEL_NAMES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    audit = json.loads((directory / "MAIN_PAPER_SOURCE_AUDIT.json").read_text())
    source = directory / "figures/short_paper_error_ratios.csv"
    if sha(source) != audit["artifacts"]["figures/short_paper_error_ratios.csv"]:
        raise ValueError("Figure source changed")
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    lookup = {(r["profile"], int(r["horizon"]), r["target"], r["model"]): r for r in rows}
    if len(rows) != 140 or len(lookup) != 140:
        raise ValueError("Expected every nonbaseline model for twenty tasks")
    panels = []
    for target in TARGETS:
        panel = np.array([[float(lookup[p, h, target, m]["ratio"]) for m in MODELS[1:]] for p, h in TASKS])
        if not np.all(np.isfinite(panel) & (panel > 0)):
            raise ValueError("Undefined error ratio")
        panels.append(panel)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.4,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.6), sharey=True,
                             gridspec_kw={"wspace": 0.07})
    fig.subplots_adjust(left=0.145, right=0.925, bottom=0.22, top=0.86)
    lo = min(math.log10(0.5), min(float(np.log10(p).min()) for p in panels))
    hi = max(3.5, max(float(np.log10(p).max()) for p in panels))
    norm = TwoSlopeNorm(vmin=lo, vcenter=0, vmax=hi)
    for index, (ax, ratios, title) in enumerate(zip(axes, panels, ("Nominal close price", "Cumulative log return"), strict=True)):
        im = ax.imshow(np.log10(ratios), aspect="auto", cmap="RdBu_r", norm=norm)
        ax.set_title(title, fontsize=9.1, pad=8)
        ax.set_xticks(range(7), MODEL_NAMES[1:], rotation=42, ha="right", fontsize=7.3)
        ax.set_yticks(range(10), [p.title()+f" H{h}"+("*" if p=="annual" else "") for p, h in TASKS], fontsize=7.4)
        ax.tick_params(axis="both", length=0)
        ax.set_xticks(np.arange(-0.5, 7), minor=True)
        ax.set_yticks(np.arange(-0.5, 10), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)
        if index:
            ax.tick_params(labelleft=False)
        for (y, x), value in np.ndenumerate(ratios):
            level = norm(math.log10(value))
            text = (f"{value:.3f}" if 0.9 < value < 1.1 else f"{value:.2f}" if value < 10
                    else f"{value:.1f}" if value < 100 else f"{value:.0f}" if value < 10000
                    else f"{value:.0e}".replace("e+0", "e").replace("e+", "e"))
            ax.text(x, y, text, ha="center", va="center", fontsize=6.7,
                    color="white" if level < 0.18 or level > 0.82 else "#171717")
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax = fig.add_axes([0.941, 0.22, 0.014, 0.64])
    bar = fig.colorbar(im, cax=cax, ticks=np.log10([0.5, 1, 10, 100, 1000]))
    bar.ax.set_yticklabels(["0.5", "1", "10", "100", "1000"], fontsize=6.8)
    bar.ax.tick_params(length=2, width=0.5, pad=2)
    title = ("Historical development: twenty five-fold tasks" if audit["complete_core"] else
             f"Layout preview: {audit['completed_folds']}/100 folds; minute comparisons incomplete")
    fig.suptitle(title, fontsize=9, y=0.97)
    fig.text(0.145, 0.035, "Ratio < 1 favors the model; shared logarithmic colors.  *Annual: short-history variant.", fontsize=7.2)
    paths = []
    for extension in ("png", "pdf", "svg"):
        path = directory / f"figures/short_paper_error_ratios.{extension}"
        if path.exists():
            raise ValueError("Preserve earlier figure releases")
        fig.savefig(path, dpi=320, facecolor="white")
        paths.append(path)
    plt.close(fig)
    result = {"status": "compact_all_twenty_task_figure_generated", "at_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": sha(source), "cells": 140, "completed_folds": audit["completed_folds"],
              "layout_preview": audit["layout_preview"], "figure_width_inches": 7.1,
              "figure_height_inches": 3.6, "minimum_label_points_at_native_width": 6.7,
              "shared_log_color_limits": [lo, hi], "script_sha256": sha(__file__),
              "artifacts": {p.name: sha(p) for p in paths}, "new_training_started": False, "Q3_values_read": False}
    (directory / "figures/FIGURE_AUDIT.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"status": result["status"], "cells": 140, "native_width_inches": 7.1}))


if __name__ == "__main__":
    main()
