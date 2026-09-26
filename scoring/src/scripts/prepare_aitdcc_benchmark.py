"""Prepare the five AITDCC byte-stream benchmarks at an exact eval window.

Each selected source is extracted verbatim from the commit-pinned official
archive and cached as one contiguous ``.raw`` file. Baking then emits every
full, contiguous, non-overlapping window and drops only the incomplete tail.
The source bytes are never decoded or interpreted.

Usage (from project root):
    python -m src.scripts.prepare_aitdcc_benchmark --context-length 256
    python -m src.scripts.prepare_aitdcc_benchmark \
        --datasets aitdcc_a_protein --context-length 256 \
        --output src/data/learner_test_data/c256/aitdcc_a_protein.jsonl
"""

import argparse
import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

from src.scripts.bake_utils import (RAW_SOURCES_DIR, bake_output_path,
                                    chunk_bytes, make_record,
                                    window_from_context, write_bake)

ARCHIVE_REVISION = "16887f9d68f273503eddb69e0c29aa7b4f39ba7a"
ARCHIVE_URL = (
    "https://raw.githubusercontent.com/AITDCC/aitdcc.github.io/"
    f"{ARCHIVE_REVISION}/data/data.zip"
)
ARCHIVE_SIZE = 7_816_155
ARCHIVE_SHA256 = (
    "1b7128ca7ee6dd391cc09fb4dbea54e817b85a5cfe11dbc9c00c9794bb36feef"
)


@dataclass(frozen=True)
class DatasetSpec:
    stem: str
    member: str
    size: int
    sha256: str


DATASETS = {
    "aitdcc_a_protein": DatasetSpec(
        stem="aitdcc_a_protein",
        member="data/A",
        size=1_308_765,
        sha256="4847f6507db6987d1c31e0a692ccf40d1c8953d0c15feee8139615a6a1f7f8e7",
    ),
    "aitdcc_b_c_source": DatasetSpec(
        stem="aitdcc_b_c_source",
        member="data/B",
        size=1_168_767,
        sha256="e14f65d4ba595d0c7c84e21a43f07de7d1cac693220d7366517f2428f52051ab",
    ),
    "aitdcc_d_glibc_rand": DatasetSpec(
        stem="aitdcc_d_glibc_rand",
        member="data/D",
        size=2_000_000,
        sha256="dbb62ea0572cac060ea94f70b4562d7f1177143ce74fdce9eb25005df0044d43",
    ),
    "aitdcc_e_atlas_float32": DatasetSpec(
        stem="aitdcc_e_atlas_float32",
        member="data/E",
        size=1_012_112,
        sha256="ff07707d8705721e2ef9f0dc40fadf4af073cf17e072d26c7bab4188b1080e13",
    ),
    "aitdcc_g_astronomy": DatasetSpec(
        stem="aitdcc_g_astronomy",
        member="data/G",
        size=2_526_752,
        sha256="87ab967bcc7cba7192352915ecd147e9a96114923ea69bbc7fa32d7e7599ce16",
    ),
}


def _verify_bytes(data: bytes, expected_size: int, expected_sha256: str,
                  label: str) -> None:
    actual_sha256 = hashlib.sha256(data).hexdigest()
    if len(data) != expected_size or actual_sha256 != expected_sha256:
        raise SystemExit(
            f"{label} failed verification: got {len(data)} bytes, "
            f"sha256 {actual_sha256}; expected {expected_size} bytes, "
            f"sha256 {expected_sha256}")


def cache_zip_members(archive: bytes, specs: Iterable[DatasetSpec],
                      cache_dir: Path, *, archive_size: int,
                      archive_sha256: str) -> dict[str, Path]:
    """Verify an archive and cache selected members exactly as stored.

    The explicit archive bytes, specs, and expected hashes make this helper
    independently testable with an in-memory fixture ZIP.
    """
    _verify_bytes(archive, archive_size, archive_sha256, "AITDCC archive")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as source_zip:
            for spec in specs:
                try:
                    raw = source_zip.read(spec.member)
                except KeyError as exc:
                    raise SystemExit(
                        f"AITDCC archive is missing member {spec.member}") from exc
                _verify_bytes(raw, spec.size, spec.sha256,
                              f"AITDCC member {spec.member}")
                cache_file = cache_dir / f"{spec.stem}.raw"
                tmp = cache_file.with_suffix(cache_file.suffix + ".tmp")
                tmp.write_bytes(raw)
                tmp.replace(cache_file)
                paths[spec.stem] = cache_file
                print(f"  cached {spec.member}: {cache_file} ({len(raw)} bytes)")
    except zipfile.BadZipFile as exc:
        raise SystemExit("AITDCC archive is not a valid ZIP file") from exc
    return paths


def _cache_matches(cache_file: Path, spec: DatasetSpec) -> bool:
    if not cache_file.is_file() or cache_file.stat().st_size != spec.size:
        return False
    return hashlib.sha256(cache_file.read_bytes()).hexdigest() == spec.sha256


def ensure_raw_sources(stems: Iterable[str],
                       cache_dir: Path = RAW_SOURCES_DIR) -> dict[str, Path]:
    """Return verified raw caches, downloading the archive at most once."""
    specs = [DATASETS[stem] for stem in stems]
    paths = {}
    missing = []
    for spec in specs:
        cache_file = Path(cache_dir) / f"{spec.stem}.raw"
        if _cache_matches(cache_file, spec):
            print(f"  raw cache hit: {cache_file} ({spec.size} bytes, verified)")
            paths[spec.stem] = cache_file
        else:
            print(f"  raw cache miss or invalid: {cache_file}")
            missing.append(spec)

    if missing:
        print(f"  downloading pinned AITDCC archive: {ARCHIVE_URL}")
        response = requests.get(ARCHIVE_URL, timeout=120)
        response.raise_for_status()
        paths.update(cache_zip_members(
            response.content, missing, Path(cache_dir),
            archive_size=ARCHIVE_SIZE, archive_sha256=ARCHIVE_SHA256))
    return paths


def prepare_dataset(spec: DatasetSpec, source: Path, context_length: int,
                    out_path: Path, overwrite: bool) -> None:
    window = window_from_context(context_length)
    raw = source.read_bytes()
    num_sequences, dropped_tail = divmod(len(raw), window)
    if not num_sequences:
        raise SystemExit(
            f"{spec.stem}: source has {len(raw)} bytes, fewer than the "
            f"{window}-byte eval window")
    print(f"  {spec.stem}: {len(raw)} source bytes -> {num_sequences} full "
          f"{window}-byte windows; dropping {dropped_tail} tail bytes")

    def records():
        for index, seq in enumerate(chunk_bytes(raw, window)):
            yield make_record(
                spec.stem, seq, index, context_length,
                {"source": "AITDCC",
                 "source_revision": ARCHIVE_REVISION,
                 "archive_member": spec.member,
                 "source_sha256": spec.sha256,
                 "chunk_index": index})

    write_bake(out_path, records(), context_length, overwrite=overwrite)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bake AITDCC byte-stream benchmark(s) at an exact eval window.")
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS),
        metavar="STEM",
        help="Dataset stem(s) to bake (default: all five)")
    parser.add_argument(
        "--context-length", type=int, required=True,
        help="Target context (prefix + data bytes); the baked window is "
             "context_length - 1")
    parser.add_argument(
        "--output", default=None,
        help="Output path (valid only with exactly one --datasets entry)")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output files")
    return parser.parse_args()


def main():
    args = parse_args()
    window_from_context(args.context_length)
    if args.output and len(args.datasets) != 1:
        raise SystemExit(
            "--output is only valid with exactly one --datasets entry")

    sources = ensure_raw_sources(args.datasets)
    for stem in args.datasets:
        spec = DATASETS[stem]
        print(f"\nDataset: {stem}")
        out_path = (Path(args.output) if args.output
                    else bake_output_path(stem, args.context_length))
        prepare_dataset(spec, sources[stem], args.context_length, out_path,
                        overwrite=args.overwrite)

    print("\nAll done.")


if __name__ == "__main__":
    main()
