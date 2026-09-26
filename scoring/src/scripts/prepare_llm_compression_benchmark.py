"""Bake the ``hkust-nlp/llm-compression`` corpora as UTF-8 byte streams.

Huang, Zhang, Shan and He, "Compression Represents Intelligence Linearly"
(COLM 2024, arXiv:2404.09937) released three corpora built specifically for
bits-per-character compression evaluation of language models:

* ``python``      - GitHub Python files (collected Jun-Sep 2023, CodeParrot
                    quality filter, segment-level sampling) -> coding ability.
* ``cc``          - Common Crawl text (Sep-Oct 2023) -> knowledge/commonsense.
* ``arxiv_math``  - arXiv math papers (Sep-Oct 2023) -> mathematical reasoning.

Each corpus is one JSONL file (``data/<subset>.jsonl``, fields ``content`` /
``subset`` / ``meta``) fetched at a commit-pinned revision of the Hugging
Face repository (not gated; plain HTTPS) and cached verbatim under
``raw_sources/`` with size + sha256 verification, so baking a new context is a
network-free re-parse that preserves document boundaries.

The scored bytes are the UTF-8 encoding of ``content``. Documents are never
concatenated: each document is divided into exact eval windows on its own and
its short tail is dropped (at window 4095 many Python files hold no full
window; the bake reports how many). A deterministic document-balanced cap
(shared with the audio bakes) keeps the default bake small while mixing
documents and positions.

License: CC-BY-NC-SA-4.0 (research use).

Usage (from project root):
    python -m src.scripts.prepare_llm_compression_benchmark --context-length 256
    python -m src.scripts.prepare_llm_compression_benchmark \
        --datasets llm_compression_arxiv_math --context-length 256 \
        --output src/data/learner_test_data/c256/llm_compression_arxiv_math.jsonl
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

from src.scripts.audio_bake_utils import (clip_sort_key, iter_capped_windows,
                                          plan_clip_windows, sha256_file)
from src.scripts.bake_utils import (RAW_SOURCES_DIR, bake_output_path,
                                    make_record, window_from_context,
                                    write_bake)

HF_BASE = "https://huggingface.co"
DATASET_ID = "hkust-nlp/llm-compression"
SOURCE = "hkust-nlp/llm-compression"
SOURCE_REVISION = "3dcd667026290f7f3e228af087a6cd2b48ce1e5a"
SOURCE_LICENSE = "CC-BY-NC-SA-4.0"
SOURCE_CITATION = ("Huang, Zhang, Shan, He. Compression Represents "
                   "Intelligence Linearly. COLM 2024 (arXiv:2404.09937)")
STEM = "llm_compression_python"
DEFAULT_NUM_SEQUENCES = 8_192


@dataclass(frozen=True)
class DatasetSpec:
    stem: str
    subset: str
    hf_path: str
    size: int
    sha256: str


# Sizes and sha256 digests are the Hugging Face LFS object ids at
# SOURCE_REVISION (the LFS oid IS the file's sha256).
DATASETS = {
    STEM: DatasetSpec(
        stem=STEM,
        subset="python",
        hf_path="data/python.jsonl",
        size=104_703_460,
        sha256="41de68abeb7e6062a204992e041bb55f1435a3edced873327caec6036ca9b657",
    ),
    "llm_compression_cc": DatasetSpec(
        stem="llm_compression_cc",
        subset="cc",
        hf_path="data/cc.jsonl",
        size=141_379_142,
        sha256="bb43e50f5b06ca322d9adf7f3add493d15ce3d472e0ade636c28ca43b64ee898",
    ),
    "llm_compression_arxiv_math": DatasetSpec(
        stem="llm_compression_arxiv_math",
        subset="arxiv_math",
        hf_path="data/arxiv_math.jsonl",
        size=107_766_552,
        sha256="7eacfc29f7df0b8f8a128beb504459fb8531834e9f2f26b8eec4b58df8d0b5a0",
    ),
}


def source_url(spec: DatasetSpec) -> str:
    return (f"{HF_BASE}/datasets/{DATASET_ID}/resolve/{SOURCE_REVISION}/"
            f"{spec.hf_path}")


def cache_path(spec: DatasetSpec, cache_dir: Path = RAW_SOURCES_DIR) -> Path:
    return Path(cache_dir) / f"llm_compression_{spec.subset}.jsonl"


def _cache_matches(cache_file: Path, spec: DatasetSpec) -> bool:
    cache_file = Path(cache_file)
    if not cache_file.is_file() or cache_file.stat().st_size != spec.size:
        return False
    return sha256_file(cache_file) == spec.sha256


def download_file(url: str, destination: Path, *, expected_size: int,
                  expected_sha256: str, label: str) -> Path:
    """Stream ``url`` to ``destination`` (tmp + atomic replace), hashing as it
    is written; a size/sha256 mismatch leaves no cache behind."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    digest = hashlib.sha256()
    size = 0
    try:
        print(f"  downloading {url} -> {destination}")
        with requests.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for block in response.iter_content(1 << 20):
                    if not block:
                        continue
                    output.write(block)
                    digest.update(block)
                    size += len(block)
        actual_sha256 = digest.hexdigest()
        if size != expected_size or actual_sha256 != expected_sha256:
            raise SystemExit(
                f"{label} download failed integrity check: got {size} bytes, "
                f"sha256 {actual_sha256}; expected {expected_size} bytes, "
                f"sha256 {expected_sha256}")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def ensure_raw_sources(stems: Iterable[str],
                       cache_dir: Path = RAW_SOURCES_DIR) -> dict[str, Path]:
    """Return verified raw JSONL caches, downloading each subset at most once."""
    paths = {}
    for stem in stems:
        spec = DATASETS[stem]
        cache_file = cache_path(spec, cache_dir)
        if _cache_matches(cache_file, spec):
            print(f"  raw cache hit: {cache_file} ({spec.size} bytes, verified)")
        else:
            if cache_file.exists():
                print(f"  invalid raw cache: refreshing {cache_file}")
            download_file(source_url(spec), cache_file,
                          expected_size=spec.size, expected_sha256=spec.sha256,
                          label=f"{DATASET_ID} {spec.hf_path}")
        paths[stem] = cache_file
    return paths


def document_key(spec: DatasetSpec, line_index: int) -> str:
    return f"{spec.subset}:{line_index}"


def load_documents(jsonl_path: Path, spec: DatasetSpec
                   ) -> list[tuple[str, bytes]]:
    """Return ``(key, utf8_bytes)`` per non-empty document, in the shared
    deterministic clip order. ``key`` is ``<subset>:<line_index>``."""
    docs = []
    empty = 0
    unencodable = 0
    with Path(jsonl_path).open("rb") as source:
        for line_index, line in enumerate(source):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{jsonl_path}:{line_index}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"{jsonl_path}:{line_index}: record is not an object")
            if record.get("subset") != spec.subset:
                raise ValueError(
                    f"{jsonl_path}:{line_index}: subset {record.get('subset')!r} "
                    f"!= expected {spec.subset!r}")
            content = record.get("content")
            if not isinstance(content, str):
                raise ValueError(
                    f"{jsonl_path}:{line_index}: content is not a string")
            try:
                data = content.encode("utf-8")
            except UnicodeEncodeError:
                unencodable += 1
                continue
            if not data:
                empty += 1
                continue
            docs.append((document_key(spec, line_index), data))
    if not docs:
        raise ValueError(f"{jsonl_path}: no non-empty {spec.subset} documents")
    docs.sort(key=clip_sort_key)
    print(f"  {spec.stem}: {len(docs)} documents "
          f"({sum(len(d) for _, d in docs)} UTF-8 bytes); skipped {empty} "
          f"empty and {unencodable} unencodable documents")
    return docs


def prepare_dataset(spec: DatasetSpec, source: Path, context_length: int,
                    out_path: Path,
                    num_sequences: int = DEFAULT_NUM_SEQUENCES,
                    overwrite: bool = False) -> int:
    """Bake exact per-document windows from a verified raw JSONL cache."""
    window = window_from_context(context_length)
    if num_sequences < 0:
        raise SystemExit(
            f"--num-sequences must be >= 0, got {num_sequences}")

    docs = load_documents(source, spec)
    plan = plan_clip_windows(docs, window, num_sequences, label=spec.stem)

    def records():
        for index, (doc, chunk_index, window_bytes) in enumerate(
                iter_capped_windows(plan, window)):
            metadata = {
                "source": SOURCE,
                "subset": spec.subset,
                "source_revision": SOURCE_REVISION,
                "source_file": spec.hf_path,
                "source_sha256": spec.sha256,
                "license": SOURCE_LICENSE,
                "encoding": "utf-8",
                "doc_index": int(doc.name.rsplit(":", 1)[1]),
                "doc_bytes": len(doc.pcm8),
                "num_windows": doc.num_windows,
                "chunk_index": chunk_index,
            }
            yield make_record(
                spec.stem, list(window_bytes), index, context_length, metadata)

    return write_bake(
        out_path, records(), context_length, overwrite=overwrite)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Bake hkust-nlp/llm-compression corpora as UTF-8 bytes at "
                    "an exact eval window.")
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(DATASETS), default=[STEM],
        metavar="STEM",
        help=f"Dataset stem(s) to bake (default: {STEM})")
    parser.add_argument(
        "--context-length", type=int, required=True,
        help="Target context (prefix + data bytes); the baked window is "
             "context_length - 1")
    parser.add_argument(
        "--output", default=None,
        help="Output path (valid only with exactly one --datasets entry)")
    parser.add_argument(
        "--num-sequences", type=int, default=DEFAULT_NUM_SEQUENCES,
        help=f"Deterministic document-balanced cap (default: "
             f"{DEFAULT_NUM_SEQUENCES}; 0 = all full windows)")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite an existing output file")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    window_from_context(args.context_length)
    if args.num_sequences < 0:
        raise SystemExit(
            f"--num-sequences must be >= 0, got {args.num_sequences}")
    if len(set(args.datasets)) != len(args.datasets):
        raise SystemExit("--datasets entries must be unique")
    if args.output and len(args.datasets) != 1:
        raise SystemExit(
            "--output is only valid with exactly one --datasets entry")

    jobs = []
    for stem in args.datasets:
        out_path = (Path(args.output) if args.output
                    else bake_output_path(stem, args.context_length))
        jobs.append((DATASETS[stem], out_path))

    targets = [path.resolve() for _, path in jobs]
    if len(set(targets)) != len(targets):
        raise SystemExit(
            f"selected datasets resolve to duplicate outputs: {targets}")
    if not args.overwrite:
        existing = [path for _, path in jobs if path.exists()]
        if existing:
            raise SystemExit(
                f"{existing[0]} already exists; pass --overwrite to replace it")

    sources = ensure_raw_sources(args.datasets)
    for spec, out_path in jobs:
        print(f"\nDataset: {spec.stem}")
        prepare_dataset(spec, sources[spec.stem], args.context_length,
                        out_path, num_sequences=args.num_sequences,
                        overwrite=args.overwrite)

    print("\nAll done.")


if __name__ == "__main__":
    main()
