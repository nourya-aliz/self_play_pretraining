#!/usr/bin/env python
"""Figure 7: scaling laws across all evaluation datasets.

The same three pretraining arms and the same compute-optimal construction as
Figure 2, on every corpus in the evaluation suite, grouped by modality. Points
are the cross-rung Pareto envelope over (model size, training round, ensemble
size K); lines are log-log power-law fits to those envelopes.

The frontier CSVs live next to Figure 2 and are read from there rather than
duplicated.

Usage: python fig7.py
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
DATA = HERE.parent / "fig2_transfer_across_modalities" / "data"
CTX, SELFPLAY_POOL = 4096, 1536
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e6e3"

ARMS = [("selfplay", "#2a8a8a", "-", True),
        ("uniform", "#e8702a", "--", False),
        ("pcfg", "#7b5ea7", "-.", False)]
LABELS = ["Self-Play Pretraining\nwith Zero Data",
          "Pretraining on programs sampled\nfrom a universal prior",
          "Pretrain on PCFG"]
GROUPS = [
    ("Text", [("dclm", "DCLM"),
              ("llm_compression_cc", "Common Crawl"),
              ("kolmogorov_text", "Kolmogorov text")]),
    ("Code & formal", [("llm_compression_python", "GitHub Python"),
                       ("aitdcc_b_c_source", "C source"),
                       ("metamath", "Metamath"),
                       ("llm_compression_arxiv_math", "arXiv math")]),
    ("Speech audio", [("speech_commands_pcm8", "Speech Commands 16 kHz"),
                      ("speech_commands_pcm8_8khz", "Speech Commands 8 kHz"),
                      ("audio_8bit", "LibriSpeech PCM8"),
                      ("audio_16bit", "LibriSpeech PCM16")]),
    ("Sound & music audio", [("esc50_pcm8", "ESC-50 44.1 kHz"),
                             ("esc50_pcm8_11khz", "ESC-50 11 kHz"),
                             ("musicnet_pcm8", "MusicNet 44.1 kHz"),
                             ("musicnet_pcm8_11khz", "MusicNet 11 kHz")]),
    ("Vision, melody & DNA", [("cifar10_rgb_hwc", "CIFAR-10 HWC"),
                              ("cifar10_rgb_planar", "CIFAR-10 planar"),
                              ("mutopia_melody_16th", "Mutopia melody"),
                              ("kolmogorov_dna", "Kolmogorov DNA")]),
    ("Binary, numeric & synthetic", [("aitdcc_d_glibc_rand", "glibc rand"),
                                     ("arithmetic", "Arithmetic")]),
]

plt.rcParams.update({"font.size": 9, "font.family": "serif",
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
    best, env = np.inf, []
    for C, bpb in sorted(points):
        if bpb < best:
            best = bpb
            env.append((C, bpb))
    return np.asarray(env)


def pool_from(name):
    counts = json.load(open(DATA / f"{name}_programs_per_round.json"))["programs_per_round"]
    return lambda rung: counts[rung]


def yticks_for(lo, hi):
    ticks = (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8) if hi / lo > 20 else \
            (1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9)
    return [t for t in ticks if lo * 0.95 <= t <= hi * 1.05]


arms = {"selfplay": load("selfplay", lambda rung: SELFPLAY_POOL),
        "uniform": load("uniform", pool_from("prior")),
        "pcfg": load("pcfg", pool_from("pcfg"))}

fig, axes = plt.subplots(len(GROUPS), 4, figsize=(17.6, 3.4 * len(GROUPS)))
for r, (group, panels) in enumerate(GROUPS):
    for c in range(4):
        ax = axes[r, c]
        if c >= len(panels):
            ax.axis("off")
            continue
        corpus, title = panels[c]
        envs = []
        for arm, color, ls, filled in ARMS:
            env = envelope(arms[arm][corpus])
            envs.append(env)
            ax.scatter(env[:, 0], env[:, 1], s=13,
                       facecolors=color if filled else "white",
                       edgecolors=color, linewidths=1.0, zorder=3)
            slope, intercept = np.polyfit(np.log(env[:, 0]), np.log(env[:, 1]), 1)
            xs = np.geomspace(env[0, 0], env[-1, 0], 120)
            ax.plot(xs, np.exp(intercept + slope * np.log(xs)), color=color,
                    lw=1.8, ls=ls, zorder=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_yticks(yticks_for(min(e[:, 1].min() for e in envs),
                                 max(e[:, 1].max() for e in envs)))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(title, fontsize=12, color=INK, pad=6)
        if c == 0:
            ax.set_ylabel("loss (bits/byte)")
        if r == len(GROUPS) - 1:
            ax.set_xlabel(r"$C = K \times$ params $\times$ tokens")

fig.tight_layout(rect=(0.025, 0, 1, 0.95))
for r, (group, _) in enumerate(GROUPS):      # rotated modality label per row
    box = axes[r, 0].get_position()
    fig.text(0.008, (box.y0 + box.y1) / 2, group, rotation=90, ha="center",
             va="center", fontsize=13, color=INK)

handles = [Line2D([], [], color=color, lw=3, ls=ls, marker="o", ms=8,
                  markerfacecolor=color if filled else "white", markeredgewidth=1.0)
           for _, color, ls, filled in ARMS]
fig.legend(handles, LABELS, loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 0.995), fontsize=20, columnspacing=1.6,
           handlelength=2, handletextpad=0.6)
for ext in ("pdf", "png"):
    fig.savefig(HERE / f"fig7.{ext}", dpi=150, bbox_inches="tight")
print(HERE / "fig7.pdf")
