#!/usr/bin/env python
"""Figure 2: self-play learns predictive structure that transfers across
modalities.

Eight panels, three pretraining arms: self-play, programs drawn from a fixed
universal prior, and probabilistic context-free grammars. For each arm the
plotted points are the cross-rung Pareto envelope over (model size, training
round, ensemble size K) and the line is a log-log power-law fit to that
envelope.

Effective compute is ``C = K * N * programs_per_round * 4096 * (round + 1)``
with 1536 programs per round for self-play and the per-rung counts in
``data/*_programs_per_round.json`` for the two fixed arms. Round 0 is excluded.

Usage: python fig2.py
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, ScalarFormatter

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CTX, SELFPLAY_POOL = 4096, 1536
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e6e3"

# (arm, colour, line style, filled markers)
ARMS = [("selfplay", "#2a8a8a", "-", True),
        ("uniform", "#e8702a", "--", False),
        ("pcfg", "#7b5ea7", "-.", False)]
LABELS = ["Self-Play Pretraining\nwith Zero Data",
          "Pretraining on programs sampled\nfrom a universal prior",
          "Pretrain on PCFG"]
# (corpus, panel title, dataset subtitle)
PANELS = [("cifar10_rgb_hwc", "Images", "CIFAR-10"),
          ("mutopia_melody_16th", "Melody", "Mutopia"),
          ("musicnet_pcm8", "Audio", "MusicNet 44.1 kHz"),
          ("audio_8bit", "Speech", "LibriSpeech PCM8"),
          ("dclm", "Text", "DCLM"),
          ("llm_compression_python", "Python", "GitHub Python"),
          ("aitdcc_b_c_source", "C code", "C source"),
          ("llm_compression_cc", "Common Crawl", "Web text")]

plt.rcParams.update({"font.size": 10, "font.family": "serif",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.5, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})


def load(arm, pool_of):
    pts = defaultdict(list)
    for row in csv.DictReader(open(DATA / f"{arm}_frontier_perk.csv")):
        rnd = int(row["round"])
        if rnd == 0:
            continue
        C = int(row["K"]) * int(row["N"]) * pool_of(row["rung"]) * CTX * (rnd + 1.0)
        pts[row["corpus"]].append((C, float(row["bpb"])))
    return pts


def envelope(points):
    """Running minimum of loss over points sorted by compute."""
    best, env = np.inf, []
    for C, bpb in sorted(points):
        if bpb < best:
            best = bpb
            env.append((C, bpb))
    return np.asarray(env)


def pool_from(name):
    counts = json.load(open(DATA / f"{name}_programs_per_round.json"))["programs_per_round"]
    return lambda rung: counts[rung]


arms = {"selfplay": load("selfplay", lambda rung: SELFPLAY_POOL),
        "uniform": load("uniform", pool_from("prior")),
        "pcfg": load("pcfg", pool_from("pcfg"))}

fig, axes = plt.subplots(2, 4, figsize=(16.8, 8.15), squeeze=False)
for index, (ax, (corpus, title, dataset)) in enumerate(zip(axes.ravel(), PANELS)):
    envs = []
    for arm, color, ls, filled in ARMS:
        env = envelope(arms[arm][corpus])
        envs.append(env)
        ax.scatter(env[:, 0], env[:, 1], s=15, facecolors=color if filled else "white",
                   edgecolors=color, linewidths=1, zorder=3)
        slope, intercept = np.polyfit(np.log(env[:, 0]), np.log(env[:, 1]), 1)
        xs = np.geomspace(env[0, 0], env[-1, 0], 120)
        ax.plot(xs, np.exp(intercept + slope * np.log(xs)), color=color, lw=2,
                ls=ls, zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    lo = min(env[:, 1].min() for env in envs)
    hi = max(env[:, 1].max() for env in envs)
    ticks = (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8) if hi / lo > 20 else \
            (1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9)
    ax.set_yticks([t for t in ticks if lo * 0.95 <= t <= hi * 1.05])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_title(title, fontsize=15, color=INK, pad=25)
    ax.text(0.5, 1.035, dataset, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10, color=INK2)
    if index // 4 == 1:
        ax.set_xlabel(r"effective compute  $C = K \times$ params $\times$ tokens",
                      fontsize=9)
    if index % 4 == 0:
        ax.set_ylabel("loss (bits/byte)")

handles = [Line2D([], [], color=color, lw=3, ls=ls, marker="o", ms=8,
                  markerfacecolor=color if filled else "white", markeredgewidth=1)
           for _, color, ls, filled in ARMS]
fig.legend(handles, LABELS, loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1), fontsize=20, columnspacing=1.6,
           handlelength=2, handletextpad=0.6)
fig.tight_layout(rect=(0, 0, 1, 1 - 0.9 / 8.15), w_pad=2, h_pad=2.6)
for ext in ("pdf", "png"):
    fig.savefig(HERE / f"fig2.{ext}", dpi=250, bbox_inches="tight")
print(HERE / "fig2.pdf")
