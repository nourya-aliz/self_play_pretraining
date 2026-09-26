"""Figure 6: pre-pretraining training curves, three modalities, 24.4M.

DCLM / CIFAR-10 / ESC-50 at the 24.4M rung, from random initialization and
from the final self-play checkpoint. Reads `fig6_data_cache.json`, which holds
exactly the plotted data, and writes fig6.{png,pdf}.

Presentation conventions:
  * mean over 4 seeds at each token count; a finished seed holds its final
    value in the mean, so curves extend to the longest seed and terminate at
    the mean converged BPB
  * curves smoothed with a 31-tap Hanning window; initial and final values
    are unsmoothed anchors
  * the t=0 (zero-shot) value sits at a pseudo-position labeled "0" on the
    log axis, connected into the curve
  * endpoint glyphs: mean (tokens-at-convergence, converged BPB) with a
    horizontal bar spanning the seed range; vertically offset +-12/-5 pt for
    legibility
  * fonts pre-scaled by S=1.9 so they render at ~10pt body size when the
    13.5in figure is included at ~5.9in \\linewidth

    python fig6.py
"""
import json
import statistics
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms

S = 1.9

C_TEXT, C_GOLD = '#1f77b4', '#ff7f0e'   # from-scratch, warm-start
PANELS = ["DCLM (text)", "CIFAR-10 (images)", "ESC-50 (audio)"]
HERE = Path(__file__).resolve().parent
CACHE = HERE / "fig6_data_cache.json"


def smooth(y, w=31):
    ypad = np.concatenate([np.full(w//2, y[0]), y, np.full(w//2, y[-1])])
    k = np.hanning(w); k /= k.sum()
    return np.convolve(ypad, k, mode="valid")


def main():
    data = json.loads(CACHE.read_text())

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    fig.patch.set_facecolor('white')
    for ax, title in zip(axes, PANELS):
        panel = data[title]
        x0 = None
        for arm, color, ls, dy, zo in [("warm", C_GOLD, "--", -5, 6),
                                       ("scratch", C_TEXT, "-", +12, 7)]:
            d = panel[arm]
            curves = [(np.array(sd["t_M"]), np.array(sd["bpb"]))
                      for sd in d["seeds"].values()]
            tmin = max(c[0][0] for c in curves)
            tmax = max(c[0][-1] for c in curves)
            if x0 is None:
                x0 = tmin / 1.9
            grid = np.geomspace(tmin, tmax, 500)
            vals = np.stack([np.interp(grid, t, b, right=b[-1])
                             for t, b in curves])
            gmean = smooth(vals.mean(0)); gmin = smooth(vals.min(0)); gmax = smooth(vals.max(0))
            gx = np.concatenate([[x0], grid])
            gmean = np.concatenate([[statistics.mean(d["inits"])], gmean])
            gmean[-1] = vals.mean(0)[-1]
            gmin = np.concatenate([[min(d["inits"])], gmin]); gmin[-1] = vals.min(0)[-1]
            gmax = np.concatenate([[max(d["inits"])], gmax]); gmax[-1] = vals.max(0)[-1]
            ax.plot(gx, gmean, color=color, ls=ls, lw=1.6,
                    zorder=4)
            ax.fill_between(gx, gmin, gmax, color=color, alpha=0.15, lw=0, zorder=2)
            ex = [e[0] for e in d["ends"]]; ey = [e[1] for e in d["ends"]]
            mx, my = statistics.mean(ex), statistics.mean(ey)
            offset = mtransforms.offset_copy(ax.transData, fig=fig, x=0, y=dy,
                                             units="points")
            ax.errorbar([mx], [my], xerr=[[mx - min(ex)], [max(ex) - mx]],
                        fmt='o', ms=5.5, mfc='white', mew=1.4, color=color,
                        ecolor=color, elinewidth=1.4, capsize=7, capthick=1.6,
                        zorder=zo, transform=offset)
        ax.set_facecolor('white')
        ax.set_xscale("log")
        ax.set_xlim(left=x0 * 0.92)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xticks([x0, 10, 100])
        ax.set_xticklabels(['0', '10', '100'], fontsize=9 * S)
        ax.set_title(title, fontsize=10 * S, pad=12)
        ax.set_xlabel("Tokens (M)", fontsize=10 * S)
        ax.tick_params(labelsize=9 * S)
        ax.grid(True, color='#e3e3e3', lw=.7); ax.set_axisbelow(True)
    axes[0].set_ylabel("Val BPB", fontsize=10 * S)
    handles = [
        Line2D([], [], color=C_TEXT, ls='-', lw=1.6, label='From scratch'),
        Line2D([], [], color=C_GOLD, ls='--', lw=1.6, label='Self-play warm start'),
        Line2D([], [], color='#777777', marker='o', ls='none', ms=5.5,
               mfc='white', mew=1.4, label='Convergence (mean ± range)'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=9 * S,
               frameon=False, bbox_to_anchor=(0.5, 1.04), columnspacing=1.0,
               handlelength=1.8, handletextpad=0.5)
    fig.tight_layout(rect=[0, 0, 1, 0.85])
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"fig6.{ext}",
                    dpi=180 if ext == "png" else None, facecolor='white')
    print(HERE / "fig6.pdf")


if __name__ == "__main__":
    main()
