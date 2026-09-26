#!/usr/bin/env python3
"""Derive the numeric columns of Table 2 from `hits.jsonl` and `prior_counts.json`.

For each family: the smallest rung that produced a member, the earliest
self-play round at which one appeared and the rung where that happened, and
the expected first round of discovery under uniform program sampling at 1,024
programs per round (or a 95% lower bound when the sampler found no member).

    python prior_analysis.py
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAMILIES = ["arithmetic", "fibonacci", "geometric", "quadratic", "cubic"]

# rung name -> (paper size label, parameter count)
RUNGS = {"d64h1L1": ("99k", 65_728), "d128h2L2": ("495k", 492_160),
         "d128h2L4": ("984k", 984_192), "d256h4L4": ("3.1M", 3_016_960),
         "d256h4L8": ("6.2M", 6_033_664), "d512h8L8": ("24M", 24_253_184)}


def poisson_ci(k, n):
    """Exact 95% Poisson interval for the hit probability p from k hits in n draws."""
    from scipy.stats import chi2
    lo = chi2.ppf(0.025, 2 * k) / 2 / n if k else 0.0
    hi = chi2.ppf(0.975, 2 * k + 2) / 2 / n
    return lo, hi


def main():
    prior = json.load(open(HERE / "prior_counts.json"))
    n, per_round = prior["samples"], prior["programs_per_round"]
    print(f"uniform-prior samples: {n:,} ({per_round} programs per round)\n")

    first = {}   # (family, rung) -> earliest round
    for line in open(HERE / "hits.jsonl"):
        h = json.loads(line)
        key = (h["family"], h["rung"])
        first[key] = min(first.get(key, h["round"]), h["round"])

    for fam in FAMILIES:
        obs = {rung: r for (f, rung), r in first.items() if f == fam}
        smallest = min(obs, key=lambda g: RUNGS[g][1])
        earliest_round = min(obs.values())
        at = min((g for g in obs if obs[g] == earliest_round), key=lambda g: RUNGS[g][1])
        print(f"== {fam}")
        print(f"   smallest size: {RUNGS[smallest][0]}   earliest round: {earliest_round} "
              f"at {RUNGS[at][0]}")
        print("   first hit per rung: " + ", ".join(
            f"{g}@r{obs[g]}" for g in sorted(obs, key=lambda g: RUNGS[g][1])))

        k = prior["hit_counts"].get(fam, 0)
        lo, hi = poisson_ci(k, n)
        if k:
            p = k / n
            er = 1 / (per_round * p)
            print(f"   uniform prior: {k} hits, p = {p:.3g} (95% CI {lo:.3g} to {hi:.3g}); "
                  f"expected first round {er:,.0f}, median {er * math.log(2):,.0f}, "
                  f"CI [{1 / (per_round * hi):,.0f}, {1 / (per_round * lo):,.0f}]")
        else:
            hi = 3 / n   # rule of three: one-sided 95% bound after zero hits
            print(f"   uniform prior: 0 hits, p <= {hi:.2g} (rule of three); "
                  f"expected first round > {1 / (per_round * hi):,.0f}")
        print()


if __name__ == "__main__":
    main()
