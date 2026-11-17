"""Draw the article's four figures from bundled aggregates, without experiments."""
from pathlib import Path
import csv
import json
from datetime import datetime, timedelta

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

BASE = Path(__file__).resolve().parent
FIG = BASE / "figures"
INK = "#243744"
TEAL = "#176A68"
RUST = "#B1643C"
BLUE = "#4576A0"
PALE = "#EDF2F3"
GRAY = "#69757C"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.edgecolor": "#9CA7AC", "axes.linewidth": .6,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "savefig.facecolor": "white",
})


def read_csv(name):
    with (FIG / name).open(newline="") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    for ext in ["pdf", "svg"]:
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", pad_inches=.045)
    fig.savefig(FIG / f"{name}.png", dpi=240, bbox_inches="tight", pad_inches=.045)
    plt.close(fig)


def factorial():
    rows = read_csv("return_factorial.csv")
    fig = plt.figure(figsize=(3.5, 2.75))
    grid = fig.add_gridspec(1, 2, left=.22, right=.99, bottom=.37,
                           top=.83, wspace=.18)
    cmap = LinearSegmentedColormap.from_list("excess", ["#F1F5F4", "#9BBCC4", "#2B566D"])
    norm = Normalize(1, 1.82)
    tasks = [("minute_h60_future_log_return", "a  Minute H60"),
             ("hourly_h8_future_log_return", "b  Hourly H8")]
    for k, (task, title) in enumerate(tasks):
        ax = fig.add_subplot(grid[0, k])
        sub = [r for r in rows if r["task"] == task]
        for i, init in enumerate(["Default", "Zero"]):
            for j, anchor in enumerate(["Observed", "Removed"]):
                r = next(r for r in sub if r["initialization"] == init and r["anchor"] == anchor)
                v = float(r["normalized_mse"])
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor=cmap(norm(v)),
                                       edgecolor="white", linewidth=2))
                ax.text(j + .5, i + .5, f"{v:.4f}", ha="center", va="center",
                        fontsize=9.2, weight="medium", color="white" if v > 1.6 else INK)
        ax.set(xlim=(0, 2), ylim=(2, 0), aspect="equal")
        ax.set_xticks([.5, 1.5], ["Observed", "Removed"], fontsize=7.2)
        ax.xaxis.set_ticks_position("top")
        ax.tick_params(axis="both", length=0, pad=4)
        ax.set_yticks([.5, 1.5], ["Default\nhead", "Zero\nhead"] if k == 0 else ["", ""], fontsize=7.6)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_title(title, loc="left", pad=25, fontsize=8.4, weight="bold")
        shrunk = next(float(r["normalized_mse"]) for r in sub if r["initialization"] == "Zero + shrink")
        ax.text(.5, -.24, f"Zero head + shrink\n{shrunk:.4f}", transform=ax.transAxes,
                ha="center", va="top", fontsize=7.4, color=TEAL)
    cax = fig.add_axes([.24, .075, .73, .043])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_ticks([1, 1.4, 1.8]); cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_visible(False)
    cb.set_label("MSE / zero-return MSE   (lower is better)", labelpad=2, fontsize=7.1)
    save(fig, "return_factorial")


def cap():
    rows = read_csv("cap_values.csv")
    fig, ax = plt.subplots(figsize=(3.5, 2.65))
    fig.subplots_adjust(left=.17, right=.97, top=.77, bottom=.23)
    for y, r in zip([1, 0], rows):
        vals = [float(r[k]) for k in ["init20", "default60", "default20"]]
        ax.plot(vals, [y]*3, color="#C5CDD0", linewidth=1, zorder=1)
        for key, color, marker, dy in [("default20", RUST, "o", 14),
                                        ("default60", BLUE, "D", -18),
                                        ("init20", TEAL, "s", 14)]:
            v = float(r[key])
            ax.scatter(v, y, color=color, s=34, marker=marker, zorder=3,
                       edgecolor="white", linewidth=.6)
            label = f"{v:.4f}" if v < 2 else f"{v:.2f}"
            ax.annotate(label, (v, y), xytext=(0, dy), textcoords="offset points",
                        ha="center", va="center", fontsize=7.4, color=color)
    ax.axvline(1.8985, color=INK, linewidth=.9, linestyle=(0, (3, 2)), zorder=0)
    ax.set_xscale("log"); ax.set_xlim(.9, 18000); ax.set_ylim(-.6, 1.55)
    ax.set_yticks([1, 0], ["LSTM", "xPatch"])
    ax.set_xticks([1, 10, 100, 1000, 10000], ["1", "10", "100", "1,000", "10,000"])
    ax.tick_params(axis="x", labelsize=7, length=3)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Price MSE (USD$^2$; logarithmic axis)", fontsize=7.6)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color="#E9EDEF", linewidth=.5)
    ax.text(.035, -.34, "Dashed line: persistence = 1.8985", transform=ax.transAxes,
            fontsize=7.1, color=GRAY)
    handles = [Line2D([], [], color=c, marker=m, linestyle="none", markersize=5,
                     label=l) for c,m,l in [(RUST,"o","Default, 20 epochs"),
                                            (BLUE,"D","Default, up to 60"),
                                            (TEAL,"s","Initialized, 20")]]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.16, 1.02),
               frameon=False, fontsize=7.5, handletextpad=.6, labelspacing=.45)
    save(fig, "cap_comparison")


def annual():
    rows = read_csv("annual_intervals.csv")
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.05), sharex=True)
    fig.subplots_adjust(left=.34, right=.96, bottom=.21, top=.86, hspace=.48)
    for ax, target, title in zip(axes, ["future_close", "future_log_return"],
                                  ["a  Annual price", "b  Annual return"]):
        ax.axvline(0, color=INK, linewidth=.8, linestyle=(0, (3, 2)))
        for center, ref in [(1, "persistence" if target == "future_close" else "zero"),
                            (0, "fit_mean_shrink")]:
            for block, offset, color, marker in [(12, .1, TEAL, "o"), (24, -.1, RUST, "s")]:
                r = next(r for r in rows if r["task"] == "annual_h12_"+target
                         and r["reference"] == ref and int(r["block"]) == block)
                v = float(r["relative_improvement_pct"])
                lo,hi = json.loads(r["relative_ci95_pct"])
                y = center + offset
                ax.plot([lo, hi], [y, y], color=color, linewidth=1.3)
                ax.plot([lo,lo], [y-.045,y+.045], color=color, linewidth=.8)
                ax.plot([hi,hi], [y-.045,y+.045], color=color, linewidth=.8)
                ax.scatter(v, y, color=color, s=17, marker=marker, zorder=3)
        ax.set_ylim(-.4, 1.5)
        ax.set_yticks([1,0], ["Persistence" if target == "future_close" else "Zero return", "Shrunk mean"], fontsize=7.1)
        ax.tick_params(axis="y", length=0, pad=6)
        ax.set_title(title, loc="left", fontsize=8, weight="bold", pad=4)
        ax.grid(axis="x", linewidth=.5, color="#E8ECEE")
        for sp in ["left","bottom"]: ax.spines[sp].set_visible(False)
    axes[1].set_xlim(-142, 66)
    axes[1].set_xticks([-120,-60,0,60])
    axes[1].set_xlabel("MSE improvement (%)\nover named reference", fontsize=7.4, labelpad=4)
    axes[1].tick_params(axis="x", length=0, labelsize=7)
    fig.legend(handles=[Line2D([],[],color=TEAL,marker="o",label="12-month blocks",markersize=4),
                        Line2D([],[],color=RUST,marker="s",label="24-month blocks",markersize=4)],
               loc="upper center", bbox_to_anchor=(.53, 1), frameon=False, fontsize=7, ncol=2,
               columnspacing=1.2, handlelength=1.5)
    fig.text(.50,-.005,"47 overlapping origins  |  nominal 95% intervals",ha="center",fontsize=6.7,color=GRAY)
    save(fig, "annual_intervals")


def split():
    r = json.loads((FIG / "split_event.json").read_text())
    fig = plt.figure(figsize=(3.5, 2.85))
    ax = fig.add_axes([.06,.49,.91,.47])
    ax.axis("off")
    # Calendar positions are proportional; this is an event timeline, not a price series.
    start = datetime.fromisoformat(r["announcement_date"])
    end = datetime.fromisoformat(r["effective_date"])
    boundary = datetime.fromisoformat(r["eligibility_boundary"]).replace(tzinfo=None)
    x0,x1,xb = [mdates.date2num(d) for d in [start,end,boundary]]
    ax.set_xlim(x0-1.6,x1+1.8);ax.set_ylim(0,1)
    ax.plot([x0,x1],[.44,.44],color="#CED6D9",linewidth=1.2)
    ax.plot([xb,x1],[.44,.44],color=TEAL,linewidth=3,solid_capstyle="butt")
    for x in [x0,xb,x1]: ax.scatter(x,.44,s=22,color=INK,zorder=4)
    ax.plot([x0,x0],[.44,.73],color=INK,linewidth=.7)
    ax.text(x0,.78,"Jul 30\nSplit announced",ha="left",va="bottom",fontsize=7.6)
    ax.plot([xb,xb],[.44,.18],color=INK,linewidth=.7)
    ax.text(xb,.11,"Aug 1, 00:00 NY\nInformation eligible",ha="left",va="top",fontsize=7.2)
    ax.plot([x1,x1],[.44,.73],color=INK,linewidth=.7)
    ax.text(x1,.78,"Aug 31\nSplit takes effect",ha="right",va="bottom",fontsize=7.6)
    ax.text((xb+x1)/2,.53,"Public information available",ha="center",va="bottom",fontsize=7.1,color=TEAL)
    fig.text(.08,.425,"For eligible paths crossing the effective date:",fontsize=7.6)
    fig.text(.13,.335,r"$\hat P=P_t/4$",fontsize=12,color=TEAL)
    fig.text(.56,.335,r"$\hat R=-\log 4$",fontsize=12,color=TEAL)
    fig.text(.08,.235,"Endpoint-MSE reduction vs persistence / zero",fontsize=7.2,color=GRAY)
    fig.text(.13,.14,f"Price  {r['price_reference_mse_reduction_pct']:.2f}%",fontsize=11,color=TEAL)
    fig.text(.56,.14,f"Return  {r['return_reference_mse_reduction_pct']:.2f}%",fontsize=11,color=TEAL)
    fig.text(.5,.035,"21 of 1,970 monthly paths  |  one split event",ha="center",fontsize=7.4)
    save(fig, "split_event")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    factorial(); cap(); annual(); split()
