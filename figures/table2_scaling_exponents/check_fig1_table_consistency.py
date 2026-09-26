#!/usr/bin/env python
"""Assert that Figure 1(b)'s legend exponents equal the LaTeX table's "Ours b" column.

Replays fig1.py's scaling-panel fit (same data file, same
``compute_frontier`` + ``fit_curve_power_law_with_floor`` calls, same ``:.3f``
formatting of ``power_fit.alpha``) for every corpus in the table, and compares
against the numbers written in exponents_table.tex by make_latex_table.py.
Exit status is non-zero on any mismatch.  Run after regenerating either side:

    python make_latex_table.py && python check_fig1_table_consistency.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

from scaling_analysis import compute_frontier, fit_curve_power_law_with_floor

HERE = Path(__file__).resolve().parent

# ---- load the data as fig1.py does ----
D = json.load(open(HERE / "frontier_traj_perk.json"))
D.pop("_meta", None)
D.pop("provenance", None)

# corpus key -> row label, taken from make_latex_table.GROUPS so the mapping is not retyped
src = (HERE / "make_latex_table.py").read_text()
# Only the GROUPS literal is needed; exec the module up to (not including) the fit/IO part.
ns = {"__file__": str(HERE / "make_latex_table.py")}
exec(src[:src.index("def envelope_fit")], ns)
LABELS = {key: label for _, rows in ns["GROUPS"] for key, label, *_ in rows
          if key in ns["INCLUDE"]}                       # only rendered rows

tex = (HERE / "exponents_table.tex").read_text()
table_alpha = {}
for key, label in LABELS.items():
    m = re.search(re.escape(label) + r"\s*&\s*\$(-?\d+\.\d+)\$", tex)
    if not m:
        sys.exit(f"row for {key!r} ({label}) not found in exponents_table.tex")
    table_alpha[key] = m.group(1)

FIG1_PANELS = ("cifar10_rgb_hwc", "dclm", "mutopia_melody_16th")   # fig1.PANELS keys
bad = []
print(f"{'corpus':<28} {'fig1 alpha':>10} {'table b':>8}  note")
for key in LABELS:
    env = compute_frontier(D, key)                       # as in fig1.py
    power_fit = fit_curve_power_law_with_floor(env)
    xs = np.logspace(np.log10(env[0, 0]), np.log10(env[-1, 0]), 80)
    fit = power_fit.predict(xs)
    if np.any(fit <= 0):                                 # fig1's plottability guard
        bad.append(f"{key}: fig1 would raise (nonpositive fitted validation loss)")
    legend = f"{power_fit.alpha:.3f}"                    # legend formatting before lstrip('0')
    ok = legend == table_alpha[key]
    if not ok:
        bad.append(f"{key}: fig1 legend {legend} != table {table_alpha[key]}")
    print(f"{key:<28} {legend:>10} {table_alpha[key]:>8}  "
          f"{'OK' if ok else 'MISMATCH'}{'  [in fig1 legend]' if key in FIG1_PANELS else ''}")

if bad:
    print("\nINCONSISTENT:\n  " + "\n  ".join(bad))
    sys.exit(1)
print(f"\nall {len(LABELS)} table rows match fig1's fit; "
      f"fig1 legend rows: " + ", ".join(f"{k}={table_alpha[k]}" for k in FIG1_PANELS))
