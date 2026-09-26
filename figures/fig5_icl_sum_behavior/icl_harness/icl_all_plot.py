"""Consolidated ICL figure: every eval in icl_all_data.json, one page.

Self-contained (numpy + matplotlib). Layout:
  top row     six m x k accuracy heatmaps (first, last, max, min, mean, sum)
  bottom row  (g) index heatmap  (h) assoc heatmap  (i) learning curves
              (j) sentinel ablation  (k) decoupled control

    python icl_all_plot.py    ->  icl_all.pdf
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
D = json.load(open(HERE / "icl_all_data.json"))
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
CAT = {"shift": "#2a78d6", "sum": "#eb6834", "succ": "#1baf7a", "prev": "#eda100",
       "closest": "#e87ba4", "index": "#4a3aa7"}

S = D["sections"]
MS, KS = S["sweep"]["ms"], S["sweep"]["ks"]

fig = plt.figure(figsize=(16.4, 11.0), dpi=160)
fig.patch.set_facecolor(SURFACE)
gs = fig.add_gridspec(3, 30, hspace=0.52, wspace=2.2,
                      left=0.045, right=0.985, top=0.90, bottom=0.065)


def style(ax):
    ax.set_facecolor(SURFACE)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=6.6)


def heat(ax, A, xs, ys, title, xlab, ylab, annot=True):
    im = ax.imshow(np.ma.masked_invalid(A), cmap="viridis", vmin=0, vmax=1,
                   aspect="auto", origin="lower")
    ax.set_facecolor("#e8e7e2")
    if annot:
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                v = A[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}".lstrip("0") if v < 1 else "1.0",
                            ha="center", va="center", fontsize=5.6,
                            color="white" if v < 0.55 else "black")
    ax.set_xticks(range(len(xs)), xs)
    ax.set_yticks(range(len(ys)), ys)
    ax.set_title(title, fontsize=8.6, color=INK, pad=4)
    ax.set_xlabel(xlab, fontsize=7, color=INK2, labelpad=1)
    if ylab:
        ax.set_ylabel(ylab, fontsize=7, color=INK2)
    ax.tick_params(colors=MUTED, labelsize=6.2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im

# ---------------------------------------------------------- top: 6 sweep heatmaps
for t, f in enumerate(["first", "last", "max", "min", "mean", "sum"]):
    ax = fig.add_subplot(gs[0, t * 5:t * 5 + 5])
    A = np.array([[S["sweep"]["results"].get(f"{f}|{m}|{k}", {}).get("acc", np.nan)
                   for k in KS] for m in MS])
    heat(ax, A, KS, MS, f, "k", "m (examples)" if t == 0 else "")

# ------------------------------------------------------------ (g) index heatmap
ax = fig.add_subplot(gs[1, 0:5])
ims, iks = [4, 16, 64, 256], [2, 4, 8, 16]
A = np.array([[S["v2"]["results"].get(f"index|m={m}|k={k}", {}).get("acc", np.nan)
               for k in iks] for m in ims])
heat(ax, A, iks, ims, "index  (x$_{1..k}$, i $\\to$ x$_i$)", "k", "m (examples)")

# ------------------------------------------------------------ (h) assoc heatmap
ax = fig.add_subplot(gs[1, 5:10])
avs, ams = [4, 16, 64], [8, 32, 128, 512]
A = np.array([[S["v3"]["results"].get(f"assoc|m={m}|V={v}", {}).get("acc", np.nan)
               for m in ams] for v in avs])
heat(ax, A, ams, avs, "assoc  (key $\\to$ value recall)", "m (examples)", "V (dict size)")

# ------------------------------------------------------- (i) learning curves
ax = fig.add_subplot(gs[1, 10:17])
style(ax)
curves = {
    "shift": [(m, S["v2"]["results"][f"shift|m={m}|k=1"]["acc"]) for m in (2, 4, 8, 16, 32, 128, 512)],
    "sum":   [(m, S["v2"]["results"][f"sum|m={m}|k=2"]["acc"]) for m in (8, 32, 128, 512)],
    "succ":  [(m, S["v3"]["results"][f"succ|m={m}|k=1"]["acc"]) for m in (8, 32, 128, 512)],
    "prev":  [(m, S["v3"]["results"][f"prev|m={m}|k=1"]["acc"]) for m in (8, 32, 128, 512)],
    "closest": [(m, S["v3"]["results"][f"closest|m={m}|k=4"]["acc"]) for m in (8, 32, 128, 512)],
}
for name, pts in curves.items():
    xs, ys = zip(*pts)
    ax.semilogx(xs, ys, "o-", color=CAT[name], lw=1.6, ms=3.5, label=name)
ax.set_ylim(-0.03, 1.05)
ax.grid(color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.set_xlabel("m (examples)", fontsize=7.6, color=INK2)
ax.set_ylabel("accuracy", fontsize=7.6, color=INK2)
ax.set_title("rule induction & retrieval vs m", fontsize=8.6, color=INK, pad=4)
ax.legend(fontsize=6.4, frameon=False, labelcolor=INK2, ncol=2, loc="center right")

# ------------------------------------------------------ (j) sentinel ablation
ax = fig.add_subplot(gs[1, 17:23])
style(ax)
tasks, fmts = ["first", "last", "max", "sum"], ["zero", "s255", "none"]
FC = {"zero": "#2a78d6", "s255": "#898781", "none": "#eb6834"}
w = 0.26
for fi, fmt in enumerate(fmts):
    vals = [S["sentinel"]["results"][f"{fmt}|{t}|512"]["acc"] for t in tasks]
    ax.bar(np.arange(len(tasks)) + (fi - 1) * w, vals, w, color=FC[fmt], label=fmt)
ax.set_xticks(range(len(tasks)), tasks)
ax.set_ylim(0, 1.05)
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.set_title("sentinel ablation (k=2, m=512)", fontsize=8.6, color=INK, pad=4)
ax.set_ylabel("accuracy", fontsize=7.6, color=INK2)
ax.legend(fontsize=6.4, frameon=False, labelcolor=INK2)

# ----------------------------------------------------- (k) decoupled control
ax = fig.add_subplot(gs[1, 23:29])
style(ax)
tasks = ["max", "min", "mean", "sum"]
norm = [S["control"]["results"][f"{t}|256|2"]["acc"] for t in tasks]
dec = [S["control"]["results"][f"{t}|256|2"]["acc_decoupled"] for t in tasks]
ax.bar(np.arange(len(tasks)) - 0.19, norm, 0.38, color="#2a78d6", label="normal")
ax.bar(np.arange(len(tasks)) + 0.19, dec, 0.38, color="#b3402b", label="decoupled")
ax.set_xticks(range(len(tasks)), tasks)
ax.set_ylim(0, 1.05)
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.set_title("input-blind control (k=2, m=256)", fontsize=8.6, color=INK, pad=4)
ax.legend(fontsize=6.4, frameon=False, labelcolor=INK2)

# --------------------------------------------------- (l) palindrome heatmap
ax = fig.add_subplot(gs[2, 0:5])
pms, pks = [1, 2, 4, 8, 32, 128], [2, 4, 8, 16]
A = np.array([[S["v4"]["results"].get(f"palin|m={m}|k={k}", {}).get("acc_mean", np.nan)
               for k in pks] for m in pms])
heat(ax, A, pks, pms, "palindrome  (reverse x$_{1..k}$)", "k", "m (examples)")

# -------------------------------------------------------- (m) stack heatmap
ax = fig.add_subplot(gs[2, 5:10])
sms, sls = [1, 2, 4, 8, 32, 128], [4, 8, 16]
A = np.array([[S["v4"]["results"].get(f"stack|m={m}|L={L}", {}).get("acc", np.nan)
               for L in sls] for m in sms])
heat(ax, A, sls, sms, "stack  (push/pop trace, final pop)", "L (ops)", "m (examples)")

# ------------------------------------- (n) mechanism: accuracy vs copy lag
ax = fig.add_subplot(gs[2, 10:17])
style(ax)
PK = {4: "#2a78d6", 8: "#eb6834", 16: "#4a3aa7"}
for k in (4, 8, 16):
    best_m = max(m for m in (1, 2, 4, 8, 32, 128)
                 if f"palin|m={m}|k={k}" in S["v4"]["results"])
    acc = S["v4"]["results"][f"palin|m={best_m}|k={k}"]["acc_by_position"]
    ax.plot([2 * j + 1 for j in range(k)], acc, "o-", color=PK[k], lw=1.5, ms=3.2,
            label=f"k={k} (m={best_m})")
ax.set_ylim(-0.03, 1.05)
ax.grid(color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.set_xlabel("copy lag of the output slot (tokens)", fontsize=7.4, color=INK2)
ax.set_ylabel("accuracy", fontsize=7.6, color=INK2)
ax.set_title("palindrome slots: the ~10-15 token retrieval horizon",
             fontsize=8.6, color=INK, pad=4)
ax.legend(fontsize=6.4, frameon=False, labelcolor=INK2)

# --------------------------------- (o) mechanism: stack height at final pop
ax = fig.add_subplot(gs[2, 18:25])
style(ax)
LC = {8: "#eda100", 16: "#e87ba4"}
for L, mm in ((8, 32), (16, 32)):
    bd = S["v4"]["results"][f"stack|m={mm}|L={L}"]["acc_by_burial_depth"]
    ds = sorted(int(d) for d in bd)
    ax.plot(ds, [bd[str(d)][0] for d in ds], "o-", color=LC[L], lw=1.5, ms=3.5,
            label=f"L={L}, m={mm}")
ax.set_ylim(-0.03, 1.05)
ax.grid(color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.set_xlabel("stack height at the final pop", fontsize=7.6, color=INK2)
ax.set_ylabel("accuracy", fontsize=7.6, color=INK2)
ax.set_title("stack: taller = fresher top = easier", fontsize=8.6, color=INK, pad=4)
ax.legend(fontsize=6.4, frameon=False, labelcolor=INK2)

fig.suptitle("In-context learning battery: 'O', then m $\\times$ [0, input, f(input)], "
             "then [0, input] $\\to$ predict f(input)\n" + D["model"]
             + " · greedy exact-match · chance $\\approx$ 0.004 · gray cells exceed the 4096 context",
             fontsize=10, color=INK, y=0.985)
out = HERE / "icl_all.pdf"
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print(out)
