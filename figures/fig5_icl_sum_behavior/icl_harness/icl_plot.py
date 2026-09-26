"""ICL heatmaps: greedy accuracy per function over (m examples, k input size).

Self-contained: numpy + matplotlib + icl_results.json.
    python icl_plot.py    ->  icl_heatmaps.pdf
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
D = json.load(open(HERE / "icl_results.json"))
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
MS, KS = D["ms"], D["ks"]
FUNCS = ["first", "last", "max", "min", "mean", "sum"]

fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.6), dpi=160)
fig.patch.set_facecolor(SURFACE)
for ax, f in zip(axes.ravel(), FUNCS):
    A = np.array([[D["results"].get(f"{f}|{m}|{k}", {}).get("acc", np.nan)
                   for k in KS] for m in MS])
    im = ax.imshow(np.ma.masked_invalid(A), cmap="viridis", vmin=0, vmax=1,
                   aspect="auto", origin="lower")
    ax.set_facecolor("#e8e7e2")                 # masked cells (over the 4096 ctx)
    for i in range(len(MS)):
        for j in range(len(KS)):
            v = A[i, j]
            if not np.isfinite(v):
                continue
            ax.text(j, i, f"{v:.2f}".lstrip("0") if v < 1 else "1.0",
                    ha="center", va="center", fontsize=7.6,
                    color="white" if v < 0.55 else "black")
    ax.set_xticks(range(len(KS)), KS)
    ax.set_yticks(range(len(MS)), MS)
    ax.set_title(f, fontsize=10.5, color=INK, pad=6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_visible(False)
for ax in axes[:, 0]:
    ax.set_ylabel("m (examples)", fontsize=9, color=INK2)
for ax in axes[1]:
    ax.set_xlabel("k (input length)", fontsize=9, color=INK2)
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cb.set_label("greedy exact-match accuracy", fontsize=8.6, color=INK2)
cb.ax.tick_params(colors=MUTED, labelsize=7.5)
fig.suptitle("In-context learning: 'O', then m x [0, x$_{1..k}$, f(x)], then [0, x$_{1..k}$] $\\to$ predict f(x)\n"
             + D["model"] + f" · {D['trials']} trials/cell · chance $\\approx$ 0.004",
             fontsize=9.6, color=INK, y=0.99)
out = HERE / "icl_heatmaps.pdf"
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print(out)
