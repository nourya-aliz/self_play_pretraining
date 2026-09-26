"""Prepare the pinned Metamath ``set.mm`` benchmark at an exact eval window.

The original file and a deterministic stripped byte stream are cached under
``raw_sources/``.  The stripped stream removes ``$( ... $)`` comments, drops
empty lines, and normalizes ASCII whitespace within each remaining line.  It is
then divided into consecutive full windows; only the final incomplete tail is
dropped.

Usage (from project root):
    python -m src.scripts.prepare_metamath_benchmark --context-length 256
"""

import argparse
import hashlib
from pathlib import Path

import requests

from src.scripts.bake_utils import (RAW_SOURCES_DIR, bake_output_path,
                                    chunk_bytes, make_record,
                                    window_from_context, write_bake)

STEM = "metamath"
REVISION = "bcfef9892b6103ba9046bf683b4903d2ad081a41"
METAMATH_URL = (
    "https://raw.githubusercontent.com/metamath/set.mm/"
    f"{REVISION}/set.mm"
)

ORIGINAL_SIZE = 50_939_171
ORIGINAL_SHA256 = "6ec915f96ba5803de2207ca9b22e9601c96cc8bfb469afba7b5fec483e13405b"
STRIPPED_SIZE = 36_507_506
STRIPPED_SHA256 = "fd65ec9f822b714aa1b379ebe930822c7aaacb784b8c8bbb9477517cb4d4cc06"

ORIGINAL_CACHE = RAW_SOURCES_DIR / "metamath_set.mm"
STRIPPED_CACHE = RAW_SOURCES_DIR / "metamath_set_stripped.mm"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_matches(path: Path, expected_size: int, expected_sha256: str) -> bool:
    return (path.is_file()
            and path.stat().st_size == expected_size
            and _file_sha256(path) == expected_sha256)


def strip_comments_and_normalize(source: bytes) -> bytes:
    """Return the canonical benchmark stream derived from ``set.mm`` bytes.

    Metamath comments are non-nested spans beginning with ``$(`` and ending
    with the next ``$)``; an unterminated span is rejected.  After removing
    them, each physical line is normalized with ``bytes.split()``, which uses
    ASCII whitespace, and non-empty lines are joined with LF plus a final LF.
    """
    without_comments = bytearray()
    position = 0
    while True:
        start = source.find(b"$(", position)
        if start < 0:
            without_comments.extend(source[position:])
            break
        without_comments.extend(source[position:start])
        end = source.find(b"$)", start + 2)
        if end < 0:
            raise ValueError(
                f"unterminated Metamath comment starting at byte {start}")
        position = end + 2

    normalized = bytearray()
    for line in without_comments.splitlines():
        fields = line.split()
        if not fields:
            continue
        normalized.extend(b" ".join(fields))
        normalized.append(0x0A)
    return bytes(normalized)


def ensure_original_cache() -> Path:
    """Return a verified cache of the exact revision-pinned ``set.mm``."""
    if _cache_matches(ORIGINAL_CACHE, ORIGINAL_SIZE, ORIGINAL_SHA256):
        print(f"  raw cache hit: {ORIGINAL_CACHE}")
        return ORIGINAL_CACHE

    if ORIGINAL_CACHE.exists():
        print(f"  invalid raw cache: refreshing {ORIGINAL_CACHE}")
    ORIGINAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    temporary = ORIGINAL_CACHE.with_name(ORIGINAL_CACHE.name + ".tmp")
    digest = hashlib.sha256()
    size = 0
    try:
        print(f"  downloading {METAMATH_URL} -> {ORIGINAL_CACHE}")
        with requests.get(METAMATH_URL, stream=True, timeout=120) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for block in response.iter_content(1 << 20):
                    if not block:
                        continue
                    output.write(block)
                    digest.update(block)
                    size += len(block)

        actual_sha256 = digest.hexdigest()
        if size != ORIGINAL_SIZE or actual_sha256 != ORIGINAL_SHA256:
            raise SystemExit(
                "Metamath download failed integrity check: "
                f"got {size} bytes, sha256 {actual_sha256}; expected "
                f"{ORIGINAL_SIZE} bytes, sha256 {ORIGINAL_SHA256}")
        temporary.replace(ORIGINAL_CACHE)
    finally:
        if temporary.exists():
            temporary.unlink()
    return ORIGINAL_CACHE


def ensure_stripped_cache(original: Path) -> Path:
    """Return the verified comment-free, whitespace-normalized stream."""
    if _cache_matches(STRIPPED_CACHE, STRIPPED_SIZE, STRIPPED_SHA256):
        print(f"  stripped cache hit: {STRIPPED_CACHE}")
        return STRIPPED_CACHE

    if STRIPPED_CACHE.exists():
        print(f"  invalid stripped cache: regenerating {STRIPPED_CACHE}")
    print(f"  deriving stripped stream -> {STRIPPED_CACHE}")
    stripped = strip_comments_and_normalize(original.read_bytes())
    actual_sha256 = hashlib.sha256(stripped).hexdigest()
    if len(stripped) != STRIPPED_SIZE or actual_sha256 != STRIPPED_SHA256:
        raise SystemExit(
            "Metamath transform failed integrity check: "
            f"got {len(stripped)} bytes, sha256 {actual_sha256}; expected "
            f"{STRIPPED_SIZE} bytes, sha256 {STRIPPED_SHA256}")

    STRIPPED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STRIPPED_CACHE.with_name(STRIPPED_CACHE.name + ".tmp")
    try:
        temporary.write_bytes(stripped)
        temporary.replace(STRIPPED_CACHE)
    finally:
        if temporary.exists():
            temporary.unlink()
    return STRIPPED_CACHE


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bake pinned Metamath set.mm at an exact eval window.")
    parser.add_argument(
        "--context-length", type=int, required=True,
        help="Target context (prefix + data bytes); the baked window is "
             "context_length - 1")
    parser.add_argument(
        "--output", default=None,
        help="Output path (default: "
             "<data_dir>/c{context_length}/metamath.jsonl)")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite an existing output file")
    return parser.parse_args()


def main():
    args = parse_args()
    window = window_from_context(args.context_length)
    out_path = (Path(args.output) if args.output
                else bake_output_path(STEM, args.context_length))

    original = ensure_original_cache()
    stripped_path = ensure_stripped_cache(original)
    raw = stripped_path.read_bytes()
    num_sequences, dropped_tail = divmod(len(raw), window)
    if not num_sequences:
        raise SystemExit(
            f"{STEM}: source has {len(raw)} bytes, fewer than the "
            f"{window}-byte eval window")
    print(f"{STEM}: {len(raw)} stripped bytes, {num_sequences} full "
          f"{window}-byte windows; dropping final {dropped_tail}-byte tail")

    def records():
        for index, sequence in enumerate(chunk_bytes(raw, window)):
            yield make_record(
                STEM, sequence, index, args.context_length,
                {"source": "metamath/set.mm",
                 "revision": REVISION,
                 "transform": "comments_removed_whitespace_normalized",
                 "chunk_index": index})

    write_bake(out_path, records(), args.context_length,
               overwrite=args.overwrite)


if __name__ == "__main__":
    main()
