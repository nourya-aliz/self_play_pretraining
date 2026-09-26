#!/usr/bin/env python3
"""Flatten a scored trajectory JSON into the per-K CSV that Figure 2 reads.

    python traj_to_csv.py --traj frontier_traj.json \
        --out ../figures/fig2_transfer_across_modalities/data/selfplay_frontier_perk.csv

Columns: rung, N, round, corpus, K, bpb, n_seeds: one row per
(rung, round, corpus, ensemble size).
"""
import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--traj", type=Path, required=True,
                    help="trajectory JSON from build_traj_json.py")
    ap.add_argument("--out", type=Path, required=True, help="CSV to write")
    args = ap.parse_args()

    src = json.load(open(args.traj))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rung", "N", "round", "corpus", "K", "bpb", "n_seeds"])
        for rung, entry in sorted(src.items()):
            if rung == "provenance":
                continue
            for rnd, row in sorted(entry["rounds"].items(), key=lambda kv: int(kv[0])):
                for corpus, per_k in row.items():
                    if corpus == "n_seeds":
                        continue
                    for K, bpb in per_k.items():
                        w.writerow([rung, entry["N"], rnd, corpus, K, bpb,
                                    row["n_seeds"]])
                        rows += 1
    print(f"{args.out}: {rows} rows")


if __name__ == "__main__":
    main()
