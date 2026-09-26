"""Prepare the arithmetic-progression benchmark JSONL at an exact eval window.

Generates a static held-out twin of the in-training arithmetic distribution:
each sequence is ``(start + step * j) mod 256`` for ``j in [0, window)`` with
``start, step ~ Uniform[0, 256)``. Baked at exactly
``context_length - PREFIX_LEN`` data bytes into the context-stamped directory
(``<data_dir>/c{context_length}/arithmetic.jsonl``), validated on write
Fully synthetic; no source fetch, no raw cache.

The output prefix (``OUTPUT_PREFIX``, ``b'O'``) is added at load time by
``EvalCorpus``; never baked in.

Usage (from project root):
    python -m src.scripts.prepare_arithmetic_benchmark --context-length 161
"""

import argparse
from pathlib import Path

import numpy as np

from src.scripts.bake_utils import (bake_output_path, make_record,
                                    window_from_context, write_bake)


def generate_sequences(num_samples: int, window: int, seed: int):
    """(starts, steps, seqs) for the arithmetic validation corpus."""
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, 256, size=num_samples, dtype=np.uint8)
    steps = rng.integers(0, 256, size=num_samples, dtype=np.uint8)
    idx = np.arange(window, dtype=np.int64)
    seqs = ((starts.astype(np.int64)[:, None]
             + steps.astype(np.int64)[:, None] * idx[None, :]) % 256
            ).astype(np.uint8)
    return starts, steps, seqs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bake the arithmetic-progression benchmark at an exact eval window.")
    parser.add_argument("--context-length", type=int, required=True,
                        help="Target context (prefix + data bytes); the baked "
                             "window is context_length - 1")
    parser.add_argument("--output", default=None,
                        help="Output path (default: "
                             "<data_dir>/c{context_length}/arithmetic.jsonl)")
    parser.add_argument("--num-samples", type=int, default=8192,
                        help="Number of sequences (default: 8192)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for reproducibility (default: 0)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite an existing output file")
    return parser.parse_args()


def main():
    args = parse_args()
    window = window_from_context(args.context_length)
    out_path = (Path(args.output) if args.output
                else bake_output_path('arithmetic', args.context_length))

    print(f"Generating {args.num_samples} arithmetic sequences "
          f"(window {window}, seed {args.seed})...", flush=True)
    starts, steps, seqs = generate_sequences(args.num_samples, window, args.seed)

    records = (make_record('arithmetic', seq.tolist(), i, args.context_length,
                           {'source': 'synthetic_arithmetic_progression',
                            'start': int(start), 'step': int(step),
                            'seed': args.seed})
               for i, (start, step, seq) in enumerate(zip(starts, steps, seqs)))
    write_bake(out_path, records, args.context_length, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
