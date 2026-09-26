"""Figure 5: behavior of the learner on the in-context SUM task.

Panel (a) classifies the argmax prediction at the query position for every
trial and context length m into five categories, stacked as shares of the
1,024 trials. Panel (b) is the entropy of the predictive distribution, mean
and interquartile range over trials. Both read `sum_behavior_p.npz`; the
trial prompts are regenerated from the RNG seed stored in that file, exactly
as `make_sum_behavior.py` built them. No smoothing or fitted curves: area
boundaries connect the observed fractions linearly in log2(m).

Categories, in order of precedence:

    correct answer     argmax == (q1 + q2) mod 256
    preferred bytes    argmax in {0, 255, 1, 16}, the bytes the untrained
                       learner favours before any demonstration
    low 4 bits correct argmax agrees with the answer in its low nibble
    context byte       argmax appears somewhere in the prompt
    other              anything else

Writes fig5.pdf, fig5.png and sum_behavior_composition.csv.

    python fig5.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
PREFERRED_BYTES = (0, 255, 1, 16)
LABELS = ("correct_answer", "low_4_bits_correct", "preferred_bytes", "context_byte", "other")

INK, MUTED = "#222222", "#6b7280"
TEAL, TEAL_LIGHT, PREFERRED, CONTEXT, OTHER = "#167d8d", "#aad5d8", "#77838c", "#ce8a64", "#e9edef"
NAVY, GRID, AXIS = "#344f73", "#e6e9ed", "#b1b8bc"
COLORS = (TEAL, TEAL_LIGHT, PREFERRED, CONTEXT, OTHER)


def compute_statistics(path):
    z = np.load(path)
    P = z["P"].astype(np.float64)                      # [len(ms), trials, 256]
    ms = [int(m) for m in z["ms"]]
    trials = P.shape[1]
    rng = np.random.default_rng(int(z["rng_seed"]))
    demos = rng.integers(1, 256, (trials, ms[-1], 2))
    queries = rng.integers(1, 256, (trials, 2))
    dans = demos.sum(-1) % 256
    answer = queries.sum(1) % 256

    composition = np.zeros((len(ms), 5))
    for mi, m in enumerate(ms):
        argmax = P[mi].argmax(1)
        for t in range(trials):
            a = int(argmax[t])
            context = {ord("O"), 0, int(queries[t, 0]), int(queries[t, 1])}
            context |= {int(v) for j in range(m) for v in (demos[t, j, 0], demos[t, j, 1], dans[t, j])}
            if a == answer[t]:
                c = 0
            elif a in PREFERRED_BYTES:
                c = 2
            elif (a & 15) == (answer[t] & 15):
                c = 1
            elif a in context:
                c = 3
            else:
                c = 4
            composition[mi, c] += 1
    composition /= trials

    H = -(P * np.log2(P + 1e-30)).sum(2)               # [len(ms), trials] bits
    return {"ms": ms, "composition": composition, "mean": H.mean(1),
            "q25": np.quantile(H, .25, 1), "q75": np.quantile(H, .75, 1)}


def position(m):
    # m = 0 gets its own slot at the left, separated by an axis break mark
    return 0. if m == 0 else 1. + np.log2(m)


def style_axis(ax, ticks):
    ax.set_xlim(0, 10)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(.7)
    ax.tick_params(length=0, pad=4, labelsize=7)
    ax.set_xticks([position(m) for m in ticks], [str(m) for m in ticks])
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linewidth=.65, color=GRID)
    ax.set_xlabel(r"Demonstrations, $m$", fontsize=8, labelpad=7)
    for x0 in (.46, .54):
        ax.plot([x0, x0 + .03], [-.014, .014], transform=ax.get_xaxis_transform(),
                color=AXIS, linewidth=.8, clip_on=False)


def heading(fig, x, y, letter, title, subtitle):
    fig.text(x, y, letter, fontsize=9.5, fontweight="bold", va="center")
    fig.text(x + .032, y, title, fontsize=9.5, va="center")
    fig.text(x, y - .058, subtitle, fontsize=7, color=MUTED, va="center")


def draw_composition(ax, X, stats):
    values = stats["composition"]
    cumulative = np.column_stack((np.zeros(len(X)), np.cumsum(values, axis=1)))
    for i, color in enumerate(COLORS):
        ax.fill_between(X, cumulative[:, i], cumulative[:, i + 1],
                        facecolor=color, edgecolor="white", linewidth=.65, zorder=2)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_ylabel("Share of trials", fontsize=8, labelpad=6)
    ax.text(1.18, .43, "Preferred\nbytes", color="white", ha="center", va="center",
            fontsize=9, linespacing=1.15)
    ax.text(1.35, .255, "{0, 255, 1, 16}", color="white", fontsize=6.2,
            ha="center", va="center", alpha=.92)
    # the early context band is too narrow for an inside label
    ax.plot([.70, .70, 1.05], [.956, 1.052, 1.052], color="#a15d37",
            lw=.65, clip_on=False, zorder=6)
    ax.text(1.17, 1.052, "Match context byte", color="#875332", fontsize=6.7,
            ha="left", va="center", clip_on=False)
    ax.text(4.37, .735, "Other", color="#758087", fontsize=8.5, ha="center", va="center")
    ax.text(8.35, .865, "Low 4 bits\ncorrect", color="#276d78", fontsize=7,
            ha="center", va="center", linespacing=1.0)
    ax.text(7.45, .32, "Correct\nanswer", color="white", fontsize=9,
            ha="center", va="center")
    ax.plot(X[4:], values[4:, 0], "o", ms=2.5, mec="white", mew=.4, mfc=TEAL,
            clip_on=False, zorder=5)


def draw_entropy(ax, X, stats):
    ax.fill_between(X, stats["q25"], stats["q75"], color=NAVY, alpha=.12, lw=0)
    ax.plot(X, stats["mean"], color=NAVY, linewidth=1.4, marker="o", markersize=3.0,
            markeredgecolor="white", markeredgewidth=.45, clip_on=False, zorder=4)
    ax.axhline(8, color="#a7b0b6", ls=(0, (2.5, 2.5)), linewidth=.75)
    ax.set_ylim(4.55, 8.28)
    ax.set_yticks([5, 6, 7, 8])
    ax.set_ylabel("Entropy (bits)", fontsize=8, labelpad=6)
    ax.text(9.96, 8.12, "Uniform: 8 bits", color=MUTED, fontsize=6.5, ha="right", va="bottom")


def main():
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK})
    stats = compute_statistics(HERE / "sum_behavior_p.npz")
    X = np.array([position(m) for m in stats["ms"]])
    ticks = (0, 1, 2, 4, 16, 64, 512)

    fig = plt.figure(figsize=(7.0, 3.85), facecolor="white")
    ax = fig.add_axes([.095, .150, .405, .640])
    entropy = fig.add_axes([.620, .150, .350, .640])
    heading(fig, .095, .950, "a", "Behavioral composition", "Argmax categories / 1,024 trials")
    heading(fig, .620, .950, "b", "Predictive uncertainty", "Mean and interquartile range")
    style_axis(ax, ticks)
    style_axis(entropy, ticks)
    draw_composition(ax, X, stats)
    draw_entropy(entropy, X, stats)
    fig.savefig(HERE / "fig5.pdf")
    fig.savefig(HERE / "fig5.png", dpi=200)
    print(HERE / "fig5.pdf")

    with open(HERE / "sum_behavior_composition.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["m", *LABELS, "entropy_mean_bits", "entropy_q25_bits", "entropy_q75_bits"])
        for row in zip(stats["ms"], *stats["composition"].T, stats["mean"], stats["q25"], stats["q75"]):
            w.writerow([row[0]] + [f"{v:.6f}" for v in row[1:]])


if __name__ == "__main__":
    main()
