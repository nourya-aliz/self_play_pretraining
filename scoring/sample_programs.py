#!/usr/bin/env python3
"""Sample programs from generator checkpoints and run them on the machine.

Downloads `<size>/seed-<seed>/generator_<round>.pth` from
nourya-cohen/solomonoff-paper, samples programs the way training did (the
`S` tag, then 255 tokens at temperature 1 with no top-k), and runs each one
with that model size's training limits (`src/framework/machine.py`). With
several rounds, the programs are split equally across them.

Output is one `.npz` file with
  programs : uint8 [n, 256]   `S` tag + program, padded with `F`
  rounds   : int64 [n]        generator round each program came from
  outputs  : uint8 [n, 4096]  the learner's training rows: `O` + output buffer

Usage:
    python sample_programs.py --size 1M --seed 40354564 \
        --rounds 256:4096:256 --n 65536 --out programs.npz
`--rounds` takes a comma-separated list or `start:stop:step` (inclusive).
`--no-execute` keeps only the programs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from src.framework.constants import program_prefix_ids       # noqa: E402
from src.framework.machine import machine_limits, run_programs  # noqa: E402
from src.framework.model import ProgramLanguageModel          # noqa: E402
from huggingface_hub import hf_hub_download                   # noqa: E402

REPO = "nourya-cohen/solomonoff-paper"
PROGRAM_LENGTH = 256


def parse_rounds(spec: str) -> list[int]:
    if ":" in spec:
        start, stop, step = (int(x) for x in spec.split(":"))
        return list(range(start, stop + 1, step))
    return [int(x) for x in spec.split(",")]


def load_generator(size: str, seed: int, rnd: int, device: str):
    folder = f"{size}/seed-{seed}"
    cfg = json.load(open(hf_hub_download(REPO, f"{folder}/config.json")))
    model = ProgramLanguageModel(**cfg["generator"])
    blob = torch.load(hf_hub_download(REPO, f"{folder}/generator_{rnd:04d}.pth"),
                      map_location="cpu", weights_only=False)
    model.load_state_dict(blob["generator_state_dict"])
    return model.eval().to(device), cfg["generator"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--size", required=True,
                    help="ladder folder: 100k, 500k, 1M, 3M, 6M or 24M")
    ap.add_argument("--seed", type=int, default=40354564)
    ap.add_argument("--rounds", required=True)
    ap.add_argument("--n", type=int, required=True, help="total programs")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--no-execute", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rounds = parse_rounds(args.rounds)
    counts = [args.n // len(rounds) + (i < args.n % len(rounds))
              for i in range(len(rounds))]
    torch.manual_seed(args.sample_seed)

    programs, round_ids = [], []
    for rnd, count in zip(rounds, counts):
        gen, arch = load_generator(args.size, args.seed, rnd, args.device)
        for start in range(0, count, args.batch):
            b = min(args.batch, count - start)
            out = gen.generate(program_prefix_ids(b, args.device),
                               max_new_tokens=PROGRAM_LENGTH - 1)
            programs.append(out.cpu().numpy().astype(np.uint8))
        round_ids += [rnd] * count
        print(f"round {rnd}: {count} programs", flush=True)
    programs = np.concatenate(programs)
    result = {"programs": programs, "rounds": np.array(round_ids, dtype=np.int64)}

    if not args.no_execute:
        limits = machine_limits(arch["d_model"], arch["n_layers"])
        result["outputs"] = run_programs(
            programs, np.random.default_rng(args.sample_seed), **limits)
        print(f"executed with {limits}", flush=True)

    np.savez_compressed(args.out, **result)
    print(f"wrote {args.out}: {len(programs)} programs", flush=True)


if __name__ == "__main__":
    main()
