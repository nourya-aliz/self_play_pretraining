"""Prepare two raw-pixel CIFAR-10 byte benchmarks.

The tracked source is the official binary test batch: 10,000 records containing
one label byte followed by 3,072 row-major pixel bytes (1,024 red, then green,
then blue). Labels are excluded. Images stay in official test order and are
concatenated without headers or separators.

Usage (from project root):
    python -m src.scripts.prepare_cifar10_benchmark --context-length 256
    python -m src.scripts.prepare_cifar10_benchmark \
        --datasets cifar10_rgb_hwc --context-length 256
"""

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

from src.scripts.bake_utils import (RAW_SOURCES_DIR, bake_output_path,
                                    chunk_bytes, make_record,
                                    window_from_context, write_bake)

# Acquisition provenance; normal baking is deliberately offline.
CIFAR10_ARCHIVE_URL = (
    "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
)
CIFAR10_ARCHIVE_MD5 = "c32a1d4ab5d03f1284b67883e8d87530"
CIFAR10_ARCHIVE_MEMBER = "cifar-10-batches-bin/test_batch.bin"
CIFAR10_SOURCE_PATH = RAW_SOURCES_DIR / "cifar10_test_batch.bin"
IMAGE_PIXELS = 32 * 32
IMAGE_BYTES = 3 * IMAGE_PIXELS
RECORD_BYTES = 1 + IMAGE_BYTES
NUM_TEST_IMAGES = 10_000
CIFAR10_SOURCE_SIZE = NUM_TEST_IMAGES * RECORD_BYTES
CIFAR10_SOURCE_SHA256 = (
    "8e2eb146ae340b09e24670f29cabc6326dba54da8789dab6768acf480273f65b"
)
DEFAULT_NUM_SEQUENCES = 8192


@dataclass(frozen=True)
class DatasetSpec:
    stem: str
    layout: str


DATASETS = {
    "cifar10_rgb_planar": DatasetSpec(
        stem="cifar10_rgb_planar",
        layout="rgb_planar_chw",
    ),
    "cifar10_rgb_hwc": DatasetSpec(
        stem="cifar10_rgb_hwc",
        layout="rgb_interleaved_hwc",
    ),
}


def verify_source_bytes(raw: bytes, *, expected_size: int,
                        expected_sha256: str) -> None:
    """Reject any source other than the expected byte-for-byte test batch."""
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if len(raw) != expected_size or actual_sha256 != expected_sha256:
        raise SystemExit(
            "CIFAR-10 test batch failed verification: "
            f"got {len(raw)} bytes, sha256 {actual_sha256}; "
            f"expected {expected_size} bytes, sha256 {expected_sha256}")


def read_verified_source(source_path: Path = CIFAR10_SOURCE_PATH) -> bytes:
    """Read and verify the repository's exact official test-batch member."""
    source_path = Path(source_path)
    if not source_path.is_file():
        raise SystemExit(
            f"missing CIFAR-10 source: {source_path}. Restore the tracked "
            f"{CIFAR10_ARCHIVE_MEMBER} file")
    raw = source_path.read_bytes()
    verify_source_bytes(
        raw,
        expected_size=CIFAR10_SOURCE_SIZE,
        expected_sha256=CIFAR10_SOURCE_SHA256,
    )
    return raw


def serialize_pixels(records: bytes, layout: str) -> bytes:
    """Strip labels and serialize complete CIFAR records in ``layout`` order."""
    if not records or len(records) % RECORD_BYTES:
        raise ValueError(
            "CIFAR-10 records must be a non-empty multiple of "
            f"{RECORD_BYTES} bytes, got {len(records)}")

    num_images = len(records) // RECORD_BYTES
    output = bytearray(num_images * IMAGE_BYTES)
    source = memoryview(records)
    destination = memoryview(output)

    for image_index in range(num_images):
        record_start = image_index * RECORD_BYTES
        pixels = source[record_start + 1:record_start + RECORD_BYTES]
        output_start = image_index * IMAGE_BYTES
        image = destination[output_start:output_start + IMAGE_BYTES]

        if layout == "rgb_planar_chw":
            image[:] = pixels
        elif layout == "rgb_interleaved_hwc":
            image[0::3] = pixels[:IMAGE_PIXELS]
            image[1::3] = pixels[IMAGE_PIXELS:2 * IMAGE_PIXELS]
            image[2::3] = pixels[2 * IMAGE_PIXELS:]
        else:
            raise ValueError(f"unknown CIFAR-10 layout: {layout}")

    return bytes(output)


def prepare_dataset(spec: DatasetSpec, source: bytes, source_sha256: str,
                    context_length: int, num_sequences: int, out_path: Path,
                    overwrite: bool) -> None:
    """Bake one deterministic prefix of the verified CIFAR-10 test split."""
    window = window_from_context(context_length)
    if num_sequences <= 0:
        raise SystemExit(
            f"--num-sequences must be positive, got {num_sequences}")

    needed_bytes = num_sequences * window
    available_bytes = (len(source) // RECORD_BYTES) * IMAGE_BYTES
    if needed_bytes > available_bytes:
        max_sequences = available_bytes // window
        raise SystemExit(
            f"{spec.stem}: requested {num_sequences} sequences "
            f"({needed_bytes} pixel bytes), but the test split has "
            f"{available_bytes}; maximum at this context is {max_sequences}")

    images_needed = (needed_bytes + IMAGE_BYTES - 1) // IMAGE_BYTES
    source_prefix = source[:images_needed * RECORD_BYTES]
    pixel_stream = serialize_pixels(source_prefix, spec.layout)[:needed_bytes]
    complete_images, partial_image_bytes = divmod(needed_bytes, IMAGE_BYTES)
    print(
        f"  {spec.stem}: {needed_bytes} pixel bytes from "
        f"{complete_images} complete images"
        + (f" plus {partial_image_bytes} bytes" if partial_image_bytes else "")
        + f" -> {num_sequences} x {window}-byte windows")

    def records():
        for index, sequence in enumerate(chunk_bytes(pixel_stream, window)):
            yield make_record(
                spec.stem, sequence, index, context_length,
                {
                    "source": "CIFAR-10 binary test split",
                    "archive_member": CIFAR10_ARCHIVE_MEMBER,
                    "source_sha256": source_sha256,
                    "split": "test",
                    "layout": spec.layout,
                    "chunk_index": index,
                })

    written = write_bake(
        out_path, records(), context_length, overwrite=overwrite)
    assert written == num_sequences


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bake raw-pixel CIFAR-10 benchmark(s) at an exact eval window.")
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(DATASETS),
        default=list(DATASETS), metavar="STEM",
        help="Dataset stem(s) to bake (default: both layouts)")
    parser.add_argument(
        "--context-length", type=int, required=True,
        help="Target context (prefix + data bytes); the baked window is "
             "context_length - 1")
    parser.add_argument(
        "--num-sequences", type=int, default=DEFAULT_NUM_SEQUENCES,
        help=f"Number of sequences per layout (default: "
             f"{DEFAULT_NUM_SEQUENCES})")
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
    if args.num_sequences <= 0:
        raise SystemExit(
            f"--num-sequences must be positive, got {args.num_sequences}")
    if args.output and len(args.datasets) != 1:
        raise SystemExit(
            "--output is only valid with exactly one --datasets entry")

    source = read_verified_source()
    for stem in args.datasets:
        spec = DATASETS[stem]
        print(f"\nDataset: {stem}")
        out_path = (Path(args.output) if args.output
                    else bake_output_path(stem, args.context_length))
        prepare_dataset(
            spec, source, CIFAR10_SOURCE_SHA256, args.context_length,
            args.num_sequences, out_path, overwrite=args.overwrite)

    print("\nAll done.")


if __name__ == "__main__":
    main()
