#!/usr/bin/env python
"""Figure 4: in-context learning across pretraining regimes.

Three panels, one per pretraining distribution: self-play, programs drawn
from the fixed universal prior, and random PCFGs. All three learners use the
24.4M architecture at round 2816 as a four-seed ensemble, so the panels differ
only in the data the learner saw. Each curve is exact-match accuracy under
greedy decoding as a function of the number of in-context examples m.

The y axis is symlog below the smallest nonzero self-play accuracy, so cells
where a model never answered correctly sit on the axis instead of vanishing.

Usage: python fig4.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter, ScalarFormatter

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ARMS = [("selfplay", "Self-play"), ("uniform_prior", "Universal prior"),
        ("pcfg", "PCFG")]

SURFACE, INK, INK2, MUTED = "#ffffff", "#111111", "#3a3a3a", "#7a7a7a"
GRID, BASE = "#e8e8e8", "#b8b8b8"
M0X = 0.5                                   # m = 0 slot on the log-2 axis
LINTHRESH = 0.00341796875                   # smallest nonzero self-play accuracy

# label -> (results file, key template, accuracy field, ms, colour)
CURVES = {
    "reverse string": ("v4", "palin|m={m}|k=8", "acc_mean",
                       (1, 2, 4, 8, 32, 128), "#2b2b2b"),
    "stack":          ("v4", "stack|m={m}|L=4", "acc",
                       (1, 2, 4, 8, 32, 128), "#63586e"),
    "assoc":          ("v3", "assoc|m={m}|V=16", "acc", (128, 512), "#7d9b80"),
    "sum":            ("sweep", "sum|{m}|2", "acc", None, "#a3704a"),
    "max":            ("sweep", "max|{m}|2", "acc", None, "#c9a22b"),
    "min":            ("sweep", "min|{m}|2", "acc", None, "#9b90a4"),
}
M0KEY = {"reverse string": ("palin|m=0|k=8", "acc_mean"),
         "stack": ("stack|m=0|L=4", "acc"),
         "max": ("max|0|2", "acc"),
         "min": ("min|0|2", "acc")}

plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42})


def load(arm):
    """Read one arm's harness output: the m sweep plus the task-specific runs."""
    def results(name):
        return json.load(open(DATA / arm / name))["results"]
    sweep = json.load(open(DATA / arm / "icl_results.json"))
    return {"sections": {"sweep": sweep["results"], "v3": results("icl_v3_results.json"),
                         "v4": results("icl_v4_results.json")},
            "ms": sweep["ms"], "m0": results("icl_m0_results.json"),
            "lowm": results("icl_sum_lowm_results.json"),
            "adict": results("icl_assoc_dict_results.json")}


def draw(ax, arm, title):
    ax.set_facecolor(SURFACE)
    for label, (section, template, field, ms, color) in CURVES.items():
        if ms is None:                       # the sweep tasks use every m
            ms = tuple(m for m in arm["ms"] if m >= 8) if label == "sum" \
                else tuple(arm["ms"])
        xs = [float(m) for m in ms]
        ys = [arm["sections"][section][template.format(m=m)][field] for m in ms]
        if label == "sum":                   # 4096-trial cells for small m
            xs = [M0X, 1, 2, 3, 4] + xs
            ys = [arm["lowm"][f"sum|{m}|2"]["acc"] for m in (0, 1, 2, 3, 4)] + ys
        elif label == "assoc":               # x = examples beyond one printing
            xs = [M0X, 1, 2, 4, 8, 16] + [m - 16 for m in ms]
            ys = [arm["adict"][f"assoc|extra={e}|V=16"]["acc"]
                  for e in (0, 1, 2, 4, 8, 16)] + ys
        elif label in M0KEY:
            key, f0 = M0KEY[label]
            xs, ys = [M0X] + xs, [arm["m0"][key][f0]] + ys
        ax.plot(np.array(xs), np.array(ys, dtype=float), color=color, lw=1.3,
                marker="o", ms=3.2, mfc="white", mew=0.9, label=label, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("symlog", linthresh=LINTHRESH, linscale=0.4)
    ax.set_xlim(left=0.36)
    ax.set_ylim(-0.0012, 1.3)
    ax.set_xticks([M0X, 1, 4, 16, 64, 256])
    ax.set_xticklabels(["0", "1", "4", "16", "64", "256"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([0, 0.005, 0.01, 0.05, 0.2, 1.0])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"$m$  (in-context examples)", fontsize=11, color=INK2)
    ax.set_title(title, fontsize=16, color=INK, pad=8)
    ax.grid(color=GRID, lw=0.7, which="major")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=10, which="both", width=0.8)


fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.3), sharey=True)
for ax, (arm, title) in zip(axes, ARMS):
    draw(ax, load(arm), title)
axes[0].set_ylabel("exact-match accuracy", fontsize=11, color=INK2)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False,
           fontsize=13, bbox_to_anchor=(0.5, 1.0), labelcolor=INK2,
           handletextpad=0.4, markerscale=1.3)
fig.suptitle("In-context learning", fontsize=15, color=INK, y=1.06)
fig.tight_layout(rect=(0, 0, 1, 0.94))
for ext in ("pdf", "png"):
    fig.savefig(HERE / f"fig4.{ext}", facecolor=SURFACE, dpi=200, bbox_inches="tight")
print(HERE / "fig4.pdf")
