"""Figure 1, lower panels: per-modality compute frontiers and in-context learning.

(b) Validation-loss cross-rung Pareto envelopes for images, text, and
    symbolic melody, with asymptotic power-law fits and exponents in the
    legend (the same fit as ../table2_scaling_exponents/).
(c) In-context learning: exact-match accuracy (log) vs number of in-context
    examples m for reverse string, stack, associative recall, sum, max, and
    min, from the ICL harness output in this folder.

Reads frontier_traj_perk.json, icl_all_data.json, icl_m0_results.json,
icl_sum_lowm_results.json and icl_assoc_dict_results.json; uses
scaling_analysis.py from this folder.

    python fig1.py              ->  fig1_bc.pdf, fig1_bc.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter

from scaling_analysis import compute_frontier, fit_curve_power_law_with_floor

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "mathtext.fontset": "dejavusans",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ONECOL = "--onecol" in sys.argv
if ONECOL:                       # sized for a 5.5in single-column text block
    W, H, LW = 5.5, 3.95, 0.72
    F = dict(head=9.8, name=6.4, sub=4.6, alab=5.4, big=9.4, tok=7.7,
             small=5.0, note=5.7, ds=6.4, title=10.0, axis=7.2, tick=6.2,
             leg=5.6)
    MARGINS = dict(left=0.095, right=0.98, top=0.955, bottom=0.10)
    HRATIOS = [3.7, 2.55]
else:
    W, H, LW = 9.8, 8.25, 1.0
    F = dict(head=14.8, name=9.0, sub=6.6, alab=7.4, big=13, tok=10.7,
             small=6.9, note=7.8, ds=9.0, title=16.0, axis=9.8, tick=8.8,
             leg=7.8)
    MARGINS = dict(left=0.068, right=0.985, top=0.965, bottom=0.068)
    HRATIOS = [3.55, 3.45]

LEGEND_FONT = F["leg"] + (1.2 if ONECOL else 3.2)

HERE = Path(__file__).resolve().parent
D = json.load(open(HERE / "frontier_traj_perk.json"))
D.pop("_meta", None)
D.pop("provenance", None)

SURFACE, INK, INK2, MUTED = "#ffffff", "#1a1a19", "#3d3d38", "#6b6b63"
GRID, BASE, RED = "#d7d7d1", "#c3c2b7", "#b3402b"
# Series colors for the scaling panel.
MCOL = {"audio_8bit": "#d28e28", "cifar10_rgb_hwc": "#5b9360",
        "dclm": "#a9577b", "mutopia_melody_16th": "#c28a2c"}
GRID_LINE, AXIS_INK = "#d8d8d8", "#111111"
PANELS = [("cifar10_rgb_hwc", "CIFAR-10 (pixel-interleaved RGB)"),
          ("dclm", "Text (DCLM)"),
          ("mutopia_melody_16th", "Melody (Mutopia)")]
SHORT = {"audio_8bit": "Audio", "cifar10_rgb_hwc": "Images",
         "dclm": "Text", "mutopia_melody_16th": "Melody"}

fig = plt.figure(figsize=(W, H), dpi=160)
fig.patch.set_facecolor(SURFACE)
gs = fig.add_gridspec(2, 2, height_ratios=HRATIOS, width_ratios=[1.12, 1],
                      hspace=0.30, wspace=0.28, **MARGINS)
axa = fig.add_subplot(gs[0, :])
axb = fig.add_subplot(gs[1, 0])
axc = fig.add_subplot(gs[1, 1])
axa.set_visible(False)          # top row reserved for the schematic, drawn separately

# ---------------------------------------------------------------- subfig (b)
axb.set_facecolor(SURFACE)
FIT_INFO, FIT_HANDLES = {}, {}
for key, title in PANELS:
    env = compute_frontier(D, key)
    power_fit = fit_curve_power_law_with_floor(env)
    xs = np.logspace(np.log10(env[0, 0]), np.log10(env[-1, 0]), 80)
    fit = power_fit.predict(xs)
    if np.any(fit <= 0):
        raise ValueError(f"nonpositive fitted validation loss for {key}")
    FIT_INFO[key] = (xs, fit, power_fit.alpha)
    axb.scatter(env[:, 0], env[:, 1], s=12 * LW, c=MCOL[key], alpha=0.48,
                lw=0, zorder=3)
    fit_line, = axb.plot(xs, fit, color=MCOL[key], lw=1.65 * LW,
                         ls=(0, (4.0, 2.2)), alpha=1.0, zorder=4,
                         dash_capstyle="butt")
    FIT_HANDLES[key] = fit_line

axb.set_xscale("log"); axb.set_yscale("log")
axb.set_ylim(1.5, 9.0)
axb.set_yticks([2, 3, 5, 8])
axb.yaxis.set_major_formatter(ScalarFormatter())
axb.yaxis.set_minor_formatter(NullFormatter())
axb.xaxis.set_minor_formatter(NullFormatter())
compute_label = (r"effective compute  $C = K \times$ params $\times$ tokens"
                 + "\n"
                 + r"($K$ = number of seeds)")
axb.set_xlabel(compute_label,
               fontsize=F["axis"], color=AXIS_INK, fontfamily="serif",
               math_fontfamily="cm")
axb.set_ylabel("validation loss (bits/byte)", fontsize=F["axis"], color=AXIS_INK,
               fontfamily="serif")
frontier_title = ("Power-law scaling of\nvalidation loss" if ONECOL
                  else "Power-law scaling of validation loss")
axb.set_title(frontier_title, fontsize=F["title"],
              color=AXIS_INK, pad=(46 if ONECOL else 54) * LW,
              fontfamily="serif")
# Keep the legend inside the plotting area so the panel is self-contained.
legend_order = ("cifar10_rgb_hwc", "dclm", "mutopia_melody_16th")
legend_labels = [
    fr"{SHORT[key]}  $\alpha={f'{FIT_INFO[key][2]:.3f}'.lstrip('0')}$"
    for key in legend_order
]
frontier_legend = axb.legend(
    [FIT_HANDLES[key] for key in legend_order], legend_labels,
    loc="lower left", bbox_to_anchor=(0.0, 1.01, 1.0, 0.0),
    mode="expand", ncol=2,
    frameon=True, facecolor=SURFACE, edgecolor=GRID, framealpha=1.0,
    prop={"family": "serif", "size": LEGEND_FONT - 1.6},
    handlelength=1.45, handletextpad=0.32, labelspacing=0.24,
    columnspacing=1.0,
    borderpad=0.35, borderaxespad=0)
for text in frontier_legend.get_texts():
    text.set_color(INK2)
    text.set_fontweight("normal")
axb.grid(color=GRID_LINE, lw=0.72 * LW, alpha=0.68, which="major")

# ---------------------------------------------------------------- subfig (c)
# ICL harness output for the self-play ensemble (24.4M, round 2816); the same
# cells as the self-play panel of Figure 4.
def _results(name):
    return json.load(open(HERE / name))["results"]


ICLD = json.load(open(HERE / "icl_all_data.json"))["sections"]
SWEEP, SWEEP_MS = ICLD["sweep"]["results"], ICLD["sweep"]["ms"]
V3, V4 = ICLD["v3"]["results"], ICLD["v4"]["results"]
M0 = _results("icl_m0_results.json")
LOWM = _results("icl_sum_lowm_results.json")          # 4096-trial cells, m <= 4
ADICT = _results("icl_assoc_dict_results.json")       # x = examples beyond one printing
LIST_MS = (1, 2, 4, 8, 32, 128)
ICL = {
    "palindrome": ([0, *LIST_MS],
                   [M0["palin|m=0|k=8"]["acc_mean"]]
                   + [V4[f"palin|m={m}|k=8"]["acc_mean"] for m in LIST_MS],
                   "reverse string"),
    "stack":      ([0, *LIST_MS],
                   [M0["stack|m=0|L=4"]["acc"]]
                   + [V4[f"stack|m={m}|L=4"]["acc"] for m in LIST_MS],
                   "stack"),
    "assoc":      ([0, 1, 2, 4, 8, 16, 128 - 16, 512 - 16],
                   [ADICT[f"assoc|extra={e}|V=16"]["acc"] for e in (0, 1, 2, 4, 8, 16)]
                   + [V3[f"assoc|m={m}|V=16"]["acc"] for m in (128, 512)],
                   "assoc"),
    "sum":        ([0, 1, 2, 3, 4] + [m for m in SWEEP_MS if m >= 8],
                   [LOWM[f"sum|{m}|2"]["acc"] for m in (0, 1, 2, 3, 4)]
                   + [SWEEP[f"sum|{m}|2"]["acc"] for m in SWEEP_MS if m >= 8],
                   "sum"),
    "max":        ([0, *SWEEP_MS],
                   [M0["max|0|2"]["acc"]] + [SWEEP[f"max|{m}|2"]["acc"] for m in SWEEP_MS],
                   "max"),
    "min":        ([0, *SWEEP_MS],
                   [M0["min|0|2"]["acc"]] + [SWEEP[f"min|{m}|2"]["acc"] for m in SWEEP_MS],
                   "min"),
}
ICOL = {"palindrome": "#343238", "stack": "#5b4864", "assoc": "#687667",
        "sum": "#9a6952", "max": "#a77b2d", "min": "#88727d"}
axc.set_facecolor(SURFACE)
for fn, (ms, ys, label) in ICL.items():
    # m = 0 sits one step left of m = 1 on the log2 axis.
    xs = np.array([-1.0 if m == 0 else np.log2(m) for m in ms])
    axc.plot(
        xs,
        ys,
        color=ICOL[fn],
        lw=1.95 * LW,
        marker="o",
        markevery=2,
        ms=3.25 * LW,
        markerfacecolor=SURFACE,
        markeredgecolor=ICOL[fn],
        markeredgewidth=0.9 * LW,
        label=label,
        alpha=0.98,
        zorder=3,
        solid_capstyle="round",
        solid_joinstyle="round",
    )
axc.set_yscale("log")
axc.set_xlim(-1.45, 9.55)
axc.set_xticks([-1, 0, 2, 4, 6, 8], ["0", "1", "4", "16", "64", "256"])
axc.xaxis.set_minor_formatter(NullFormatter())
axc.set_yticks([0.005, 0.01, 0.05, 0.2, 1.0])
axc.yaxis.set_major_formatter(ScalarFormatter())
axc.yaxis.set_minor_formatter(NullFormatter())
axc.set_ylim(0.0025, 1.35)
axc.set_xlabel(r"$m$  (in-context examples)", fontsize=F["axis"], color=INK2)
axc.xaxis.label.set_color(AXIS_INK)
axc.xaxis.label.set_fontfamily("serif")
axc.xaxis.label.set_math_fontfamily("cm")
axc.set_ylabel("exact-match accuracy", fontsize=F["axis"], color=AXIS_INK,
               fontfamily="serif")
axc.set_title("In-context learning",
              fontsize=F["title"], color=AXIS_INK,
              pad=(40 if ONECOL else 50) * LW,
              fontfamily="serif")
axc.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
           prop={"family": "serif", "size": LEGEND_FONT},
           frameon=False,
           labelcolor=AXIS_INK, handletextpad=0.4,
           handlelength=2.15, columnspacing=1.0,
           borderpad=0.2, borderaxespad=0, labelspacing=0.32)
axc.grid(axis="y", color=GRID_LINE, lw=0.72 * LW, alpha=0.60, which="major")
axc.grid(axis="x", color=GRID_LINE, lw=0.58 * LW, alpha=0.42, which="major")

for ax in (axb, axc):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_INK)
        ax.spines[side].set_linewidth(1.0 * LW)
    ax.tick_params(colors=AXIS_INK, labelsize=F["tick"], which="both",
                   width=0.9 * LW, length=3.8 * LW)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontfamily("serif")
        t.set_math_fontfamily("cm")

# Render the two quantitative panels: (b) scaling and (c) in-context learning.
import matplotlib.transforms as mtransforms
fig.canvas.draw()
_renderer = fig.canvas.get_renderer()
_bb = mtransforms.Bbox.union(
    [ax.get_tightbbox(_renderer) for ax in (axb, axc)])
_bb = _bb.transformed(fig.dpi_scale_trans.inverted()).expanded(1.02, 1.04)
for ext in ("pdf", "png"):
    fig.savefig(HERE / f"fig1_bc.{ext}", facecolor=SURFACE, bbox_inches=_bb)

print(HERE / "fig1_bc.pdf")
