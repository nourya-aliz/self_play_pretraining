"""Prepare contiguous KoLMogorov text and DNA benchmark streams.

The large upstream byte streams are sampled by retaining a verified contiguous
prefix (32 MiB by default) under ``raw_sources/``; that cache can then be
re-chunked locally for any eval context.  The complete downloaded ZIP and its
complete selected member are validated before the prefix is accepted.

Usage (from project root):
    python -m src.scripts.prepare_kolmogorov_benchmark --context-length 256
    python -m src.scripts.prepare_kolmogorov_benchmark \
        --datasets kolmogorov_dna --context-length 256
"""

import argparse
import hashlib
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests

from src.scripts.bake_utils import (RAW_SOURCES_DIR, bake_output_path,
                                    chunk_bytes, ensure_raw_cache, make_record,
                                    window_from_context, write_bake)

DEFAULT_SAMPLE_BYTES = 32 * 1024 * 1024
KOLMOGOROV_REVISION = "690236b2192e5bc8d591f0f475367c92434a60a1"


@dataclass(frozen=True)
class SourceSpec:
    stem: str
    url: str
    member: str
    archive_size: int
    member_size: int
    source: str
    archive_sha256: str | None = None
    member_md5: str | None = None
    member_sha1: str | None = None


SOURCES = {
    "kolmogorov_text": SourceSpec(
        stem="kolmogorov_text",
        url="https://mattmahoney.net/dc/enwik9.zip",
        member="enwik9",
        archive_size=322_592_222,
        member_size=1_000_000_000,
        member_md5="e206c3450ac99950df65bf70ef61a12d",
        member_sha1="2996e86fb978f93cca8f566cc56998923e7fe581",
        source="enwik9",
    ),
    "kolmogorov_dna": SourceSpec(
        stem="kolmogorov_dna",
        url=(
            "https://media.githubusercontent.com/media/facebookresearch/"
            f"KoLMogorov/{KOLMOGOROV_REVISION}/src/data/"
            "data_to_compress_1gb/dna.bin.zip"
        ),
        member="dna.bin",
        archive_size=782_286_693,
        archive_sha256=(
            "0ca8b8bca22d501d04f2a44c5979ec383696305fa6a31abb75900462d9fb3b8f"
        ),
        member_size=2_666_666_666,
        source=f"facebookresearch/KoLMogorov@{KOLMOGOROV_REVISION}:dna.bin",
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_verified_prefix(archive_path: Path, spec: SourceSpec,
                            sample_bytes: int) -> bytes:
    """Validate ``archive_path`` and return a contiguous member prefix.

    Reading the selected member to EOF validates the ZIP CRC even though only
    its prefix is retained.  This function is deliberately independent of the
    network so tests and future callers can use a local fixture archive.
    """
    archive_path = Path(archive_path)
    if sample_bytes <= 0:
        raise ValueError(f"sample_bytes must be positive, got {sample_bytes}")
    if sample_bytes > spec.member_size:
        raise ValueError(
            f"{spec.stem}: requested {sample_bytes} bytes from a "
            f"{spec.member_size}-byte source")

    actual_archive_size = archive_path.stat().st_size
    if actual_archive_size != spec.archive_size:
        raise ValueError(
            f"{spec.stem}: archive size {actual_archive_size} != expected "
            f"{spec.archive_size}")
    if spec.archive_sha256:
        actual_sha256 = _sha256_file(archive_path)
        if actual_sha256 != spec.archive_sha256:
            raise ValueError(
                f"{spec.stem}: archive SHA-256 {actual_sha256} != expected "
                f"{spec.archive_sha256}")

    prefix = bytearray()
    md5 = hashlib.md5(usedforsecurity=False) if spec.member_md5 else None
    sha1 = hashlib.sha1(usedforsecurity=False) if spec.member_sha1 else None
    member_bytes = 0
    with zipfile.ZipFile(archive_path) as archive:
        try:
            info = archive.getinfo(spec.member)
        except KeyError as exc:
            raise ValueError(
                f"{spec.stem}: ZIP has no member {spec.member!r}") from exc
        if info.file_size != spec.member_size:
            raise ValueError(
                f"{spec.stem}: member size {info.file_size} != expected "
                f"{spec.member_size}")
        with archive.open(info) as member:
            while block := member.read(1 << 20):
                member_bytes += len(block)
                if len(prefix) < sample_bytes:
                    remaining = sample_bytes - len(prefix)
                    prefix.extend(block[:remaining])
                if md5:
                    md5.update(block)
                if sha1:
                    sha1.update(block)

    if member_bytes != spec.member_size:
        raise ValueError(
            f"{spec.stem}: extracted {member_bytes} member bytes != expected "
            f"{spec.member_size}")
    if md5 and md5.hexdigest() != spec.member_md5:
        raise ValueError(
            f"{spec.stem}: member MD5 {md5.hexdigest()} != expected "
            f"{spec.member_md5}")
    if sha1 and sha1.hexdigest() != spec.member_sha1:
        raise ValueError(
            f"{spec.stem}: member SHA-1 {sha1.hexdigest()} != expected "
            f"{spec.member_sha1}")
    return bytes(prefix)


def download_archive(url: str, destination: Path) -> None:
    """Stream one complete upstream ZIP to a temporary local file."""
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with open(destination, "wb") as output:
            for block in response.iter_content(1 << 20):
                if block:
                    output.write(block)


def fetch_verified_prefix(
        spec: SourceSpec, sample_bytes: int,
        downloader: Callable[[str, Path], None] | None = None,
        temp_parent: Path | None = None) -> bytes:
    """Download a source to temporary storage, validate it, and return a prefix."""
    downloader = downloader or download_archive
    temp_parent = Path(temp_parent) if temp_parent is not None else RAW_SOURCES_DIR
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix=f".{spec.stem}_", dir=temp_parent) as temp_dir:
        archive_path = Path(temp_dir) / f"{spec.member}.zip"
        print(f"  downloading {spec.url} ({spec.archive_size} bytes)")
        downloader(spec.url, archive_path)
        print(f"  validating complete archive and {spec.member} member")
        return extract_verified_prefix(archive_path, spec, sample_bytes)


def ensure_source_prefix(
        spec: SourceSpec, sample_bytes: int, cache_file: Path | None = None,
        downloader: Callable[[str, Path], None] | None = None) -> bytes:
    """Return a cached source prefix, fetching and verifying it when absent."""
    cache_file = (cache_file if cache_file is not None
                  else RAW_SOURCES_DIR / f"{spec.stem}.raw")
    cache_file = Path(cache_file)
    return ensure_raw_cache(
        cache_file, sample_bytes,
        lambda: [fetch_verified_prefix(
            spec, sample_bytes, downloader, cache_file.parent)])


def prepare_dataset(spec: SourceSpec, sample_bytes: int, context_length: int,
                    out_path: Path, overwrite: bool) -> None:
    window = window_from_context(context_length)
    raw = ensure_source_prefix(spec, sample_bytes)
    dropped_tail = len(raw) % window
    print(f"{spec.stem}: {len(raw)} contiguous bytes; "
          f"dropping {dropped_tail} final tail bytes")

    def records():
        for index, sequence in enumerate(chunk_bytes(raw, window)):
            yield make_record(
                spec.stem, sequence, index, context_length,
                {"source": spec.source, "chunk_index": index})

    write_bake(out_path, records(), context_length, overwrite=overwrite)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bake contiguous KoLMogorov text/DNA source prefixes.")
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(SOURCES), default=list(SOURCES),
        help="Dataset stem(s) to bake (default: both)")
    parser.add_argument(
        "--context-length", type=int, required=True,
        help="Target context (prefix + data bytes); the baked window is "
             "context_length - 1")
    parser.add_argument(
        "--sample-bytes", type=int, default=DEFAULT_SAMPLE_BYTES,
        help=f"Contiguous source-prefix bytes per dataset "
             f"(default: {DEFAULT_SAMPLE_BYTES})")
    parser.add_argument(
        "--output", default=None,
        help="Output path (one dataset only; default: "
             "<data_dir>/c{context_length}/<dataset>.jsonl)")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output files")
    return parser.parse_args()


def main():
    args = parse_args()
    window = window_from_context(args.context_length)
    if args.sample_bytes <= 0:
        raise SystemExit(
            f"--sample-bytes must be positive, got {args.sample_bytes}")
    if args.sample_bytes < window:
        raise SystemExit(
            f"--sample-bytes ({args.sample_bytes}) must be at least the eval "
            f"window ({window})")
    if args.output and len(args.datasets) != 1:
        raise SystemExit(
            "--output is only valid with exactly one --datasets entry")

    for stem in args.datasets:
        spec = SOURCES[stem]
        out_path = (Path(args.output) if args.output
                    else bake_output_path(stem, args.context_length))
        prepare_dataset(
            spec, args.sample_bytes, args.context_length, out_path,
            args.overwrite)


if __name__ == "__main__":
    main()
