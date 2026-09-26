#!/usr/bin/env python3
"""Assemble an ensemble directory from the released checkpoints.

The in-context learning harness (``figures/fig5_icl_sum_behavior/icl_harness``)
expects a directory holding a model ``config.json`` and one subdirectory per
seed with that seed's weights:

    <out>/config.json
    <out>/seed-<seed>/learner_<round>.pth

This script downloads that layout from Hugging Face. The paper's ICL results
use the 24M rung at self-play round 2816 with four seeds, which is the default.

    python fetch_ensemble.py --out ../figures/fig5_icl_sum_behavior/ensemble
"""
import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "nourya-cohen/solomonoff-paper"
PAPER_SEEDS = [40354564, 40354565, 40354566, 40354567]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--size", default="24M", help="ladder rung (default: 24M)")
    ap.add_argument("--round", type=int, default=2816,
                    help="self-play round (default: 2816)")
    ap.add_argument("--seeds", nargs="*", type=int, default=PAPER_SEEDS)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    cfg_src = hf_hub_download(repo_id=REPO,
                              filename=f"{args.size}/seed-{args.seeds[0]}/config.json")
    shutil.copyfile(cfg_src, args.out / "config.json")

    for seed in args.seeds:
        remote = f"{args.size}/seed-{seed}/learner_{args.round}.pth"
        dst = args.out / f"seed-{seed}" / f"learner_{args.round}.pth"
        if dst.exists():
            print(f"[skip] {dst}")
            continue
        cached = hf_hub_download(repo_id=REPO, filename=remote)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, dst)
        print(f"[done] {dst}")

    print(f"\nensemble ready: {args.out}")
    print(f"  export ICL_ENSEMBLE={args.out.resolve()}")


if __name__ == "__main__":
    main()
