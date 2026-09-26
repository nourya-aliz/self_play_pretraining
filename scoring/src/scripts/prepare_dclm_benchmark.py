"""Prepare a DCLM-Baseline benchmark JSONL from CommonCrawl's public HTTPS endpoint.

Bakes a representative sample from all 10 global shards of DCLM-Baseline-1.0
at an EXACT eval window: every sequence carries exactly
``context_length - PREFIX_LEN`` data bytes and the output is context-stamped
(``<data_dir>/c{context_length}/dclm.jsonl``) and validated on write.

Raw shard bytes are cached once under raw_sources/ (per shard), so baking a
NEW context re-chunks locally without touching the network as long as the
cache holds enough bytes; a larger request re-streams the shard.

No AWS credentials are required. The output prefix (``OUTPUT_PREFIX``,
``b'O'``) is added at load time by ``EvalCorpus``; never baked in.

Usage (from project root):
    python -m src.scripts.prepare_dclm_benchmark --context-length 161
    python -m src.scripts.prepare_dclm_benchmark --context-length 257 \
        --num-sequences 16384 --overwrite

Dependencies:
    pip install requests zstandard
"""

import argparse
import gzip
import io
import json
from pathlib import Path

import requests
import zstandard as zstd

from src.scripts.bake_utils import (RAW_SOURCES_DIR, bake_output_path,
                                    chunk_bytes, ensure_raw_cache, make_record,
                                    window_from_context, write_bake)

COMMONCRAWL_BASE = "https://data.commoncrawl.org"
MANIFEST_URL = f"{COMMONCRAWL_BASE}/contrib/datacomp/DCLM-baseline/DCLM-baseline.paths.gz"
NUM_GLOBAL_SHARDS = 10


def fetch_manifest() -> list[str]:
    """Download and decompress the DCLM-Baseline path manifest."""
    print("Fetching manifest...", end=" ", flush=True)
    resp = requests.get(MANIFEST_URL, timeout=60)
    resp.raise_for_status()
    with gzip.open(io.BytesIO(resp.content)) as f:
        paths = [line.decode().strip() for line in f if line.strip()]
    print(f"{len(paths)} paths found.")
    return paths


def pick_one_path_per_shard(paths: list[str]) -> dict[int, str]:
    """Return one path per global shard index (1-indexed, 1..10)."""
    selected: dict[int, str] = {}
    for path in paths:
        for shard_idx in range(1, NUM_GLOBAL_SHARDS + 1):
            tag = f"global-shard_{shard_idx:02d}_of_10"
            if tag in path and shard_idx not in selected:
                selected[shard_idx] = path
        if len(selected) == NUM_GLOBAL_SHARDS:
            break
    if len(selected) != NUM_GLOBAL_SHARDS:
        raise RuntimeError(
            f"Expected paths for {NUM_GLOBAL_SHARDS} global shards, found {len(selected)}. "
            "The manifest may be incomplete or the shard naming has changed.")
    return selected


def stream_shard_text_bytes(path: str):
    """Yield the concatenated document text bytes of one shard file."""
    url = f"{COMMONCRAWL_BASE}/{path}"
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(resp.raw) as reader:
            for line in io.TextIOWrapper(reader, encoding="utf-8"):
                text = json.loads(line).get("text", "")
                yield text.encode("utf-8", errors="replace")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bake a representative DCLM-Baseline sample at an exact eval window.")
    parser.add_argument("--context-length", type=int, required=True,
                        help="Target context (prefix + data bytes); the baked "
                             "window is context_length - 1")
    parser.add_argument("--output", default=None,
                        help="Output path (default: "
                             "<data_dir>/c{context_length}/dclm.jsonl)")
    parser.add_argument("--num-sequences", type=int, default=8192,
                        help="Total sequences, spread evenly across 10 global "
                             "shards (default: 8192)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite output file if it already exists")
    return parser.parse_args()


def main():
    args = parse_args()
    window = window_from_context(args.context_length)
    out_path = (Path(args.output) if args.output
                else bake_output_path('dclm', args.context_length))

    base_quota = args.num_sequences // NUM_GLOBAL_SHARDS
    remainder = args.num_sequences % NUM_GLOBAL_SHARDS
    quotas = {idx: base_quota + (1 if (idx - 1) < remainder else 0)
              for idx in range(1, NUM_GLOBAL_SHARDS + 1)}

    # Manifest + streaming happen only for shards whose raw cache is short.
    shard_paths: dict[int, str] = {}

    def shard_bytes(shard_idx: int, needed: int) -> bytes:
        cache_file = RAW_SOURCES_DIR / f"dclm_shard_{shard_idx:02d}.raw"

        def fetch():
            if not shard_paths:
                shard_paths.update(pick_one_path_per_shard(fetch_manifest()))
            return stream_shard_text_bytes(shard_paths[shard_idx])

        return ensure_raw_cache(cache_file, needed, fetch)

    def records():
        index = 0
        for shard_idx in range(1, NUM_GLOBAL_SHARDS + 1):
            quota = quotas[shard_idx]
            if quota == 0:
                continue
            raw = shard_bytes(shard_idx, quota * window)
            for seq in chunk_bytes(raw, window):
                yield make_record('dclm', seq, index, args.context_length,
                                  {'source': 'dclm-baseline-1.0',
                                   'global_shard': shard_idx,
                                   'chunk_index': index})
                index += 1

    write_bake(out_path, records(), args.context_length,
               overwrite=args.overwrite)


if __name__ == "__main__":
    main()
