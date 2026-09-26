"""Bake Speech Commands v0.02 test audio as headerless linear PCM8 bytes.

The official test archive contains one-second, mono, 16 kHz PCM16 WAV files.
The original benchmark converts each sample independently to conventional
unsigned PCM8 by retaining the signed PCM16 high byte and flipping its sign
bit::

    pcm8 = pcm16le_high_byte ^ 0x80

The ``speech_commands_pcm8_8khz`` and ``speech_commands_pcm8_4khz`` siblings
first apply fixed, anti-aliasing 2:1 and 4:1 polyphase FIR resamples to each
signed PCM16 recording independently, then quantize the filtered samples to
the same unsigned PCM8 alphabet.

WAV headers and labels are not part of the scored bytes. Recordings are never
concatenated: each is divided into exact eval windows and its short tail is
dropped. A deterministic clip-balanced cap keeps the default bake small while
mixing recordings, labels, and temporal positions.

The PCM8 map, the frozen FIR recipes, the WAV reader and the clip-balanced
cap live in ``src/scripts/audio_bake_utils.py`` (shared with the ESC-50 and
MusicNet bakes); this module's output is byte-identical to the pre-extraction
bake.

Source: Speech Commands v0.02 test set, created by Pete Warden.
License: Creative Commons Attribution 4.0 (CC BY 4.0).

Usage (from project root):
    python -m src.scripts.prepare_speech_commands_benchmark \
        --context-length 256
    python -m src.scripts.prepare_speech_commands_benchmark \
        --datasets speech_commands_pcm8_8khz --context-length 256
    python -m src.scripts.prepare_speech_commands_benchmark \
        --datasets speech_commands_pcm8_4khz --context-length 256
"""

import argparse
import hashlib
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

from src.scripts.audio_bake_utils import (  # noqa: F401 (re-exports)
    PCM8_4KHZ_FIR_TAPS, PCM8_4KHZ_FIR_TAPS_SHA256, PCM8_8KHZ_FIR_TAPS,
    PCM8_8KHZ_FIR_TAPS_SHA256, RESAMPLE_RECIPES, ResampleRecipe,
    clip_sort_key, iter_capped_windows, linear_pcm_to_pcm8,
    normalized_member_name, pcm16le_to_pcm8, pcm8_from_pcm16_frames,
    plan_clip_windows, read_pcm16_wav, resample_pcm16le, sha256_file,
    stable_digest)
from src.scripts.bake_utils import (RAW_SOURCES_DIR, bake_output_path,
                                    make_record, window_from_context,
                                    write_bake)

# Pre-extraction names, kept importable.
_ResampleRecipe = ResampleRecipe
_RESAMPLE_RECIPES = RESAMPLE_RECIPES
_sha256_file = sha256_file
_resample_pcm16le = resample_pcm16le
_normalized_member_name = normalized_member_name
_stable_digest = stable_digest

STEM = "speech_commands_pcm8"
STEM_4KHZ = "speech_commands_pcm8_4khz"
STEM_8KHZ = "speech_commands_pcm8_8khz"
SOURCE = "speech_commands_v0.02_test"
SOURCE_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/"
    "speech_commands_test_set_v0.02.tar.gz"
)
SOURCE_LICENSE = "CC-BY-4.0"

ARCHIVE_CACHE = RAW_SOURCES_DIR / "speech_commands_test_set_v0.02.tar.gz"
ARCHIVE_SIZE = 112_563_277
ARCHIVE_SHA256 = (
    "cc2a00c1147c2254e9be3fa0f779d8c17421dc349b86366567a8edfa9acd51df"
)
EXPECTED_CLIPS = 4_890
EXPECTED_FRAMES = 16_000
SAMPLE_RATE_HZ = 16_000
DEFAULT_NUM_SEQUENCES = 8_192


@dataclass(frozen=True)
class DatasetSpec:
    stem: str
    sample_rate_hz: int
    expected_output_frames: int
    transform: str
    resample_down: int = 1


DATASETS = {
    STEM: DatasetSpec(
        stem=STEM,
        sample_rate_hz=SAMPLE_RATE_HZ,
        expected_output_frames=EXPECTED_FRAMES,
        transform="pcm16le_high_byte_xor_0x80",
    ),
    STEM_4KHZ: DatasetSpec(
        stem=STEM_4KHZ,
        sample_rate_hz=4_000,
        expected_output_frames=4_000,
        transform="pcm16le_fir_resample_poly_1_4_then_unsigned_pcm8",
        resample_down=4,
    ),
    STEM_8KHZ: DatasetSpec(
        stem=STEM_8KHZ,
        sample_rate_hz=8_000,
        expected_output_frames=8_000,
        transform="pcm16le_fir_resample_poly_1_2_then_unsigned_pcm8",
        resample_down=2,
    ),
}


def _archive_matches(path: Path) -> bool:
    return (path.is_file()
            and path.stat().st_size == ARCHIVE_SIZE
            and sha256_file(path) == ARCHIVE_SHA256)


def ensure_archive(cache_file: Path = ARCHIVE_CACHE) -> Path:
    """Return the checksum-pinned official archive, downloading it if needed."""
    cache_file = Path(cache_file)
    if _archive_matches(cache_file):
        print(f"  raw cache hit: {cache_file} ({ARCHIVE_SIZE} bytes, verified)")
        return cache_file

    if cache_file.exists():
        print(f"  invalid raw cache: refreshing {cache_file}")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_file.with_name(cache_file.name + ".tmp")
    digest = hashlib.sha256()
    size = 0
    try:
        print(f"  downloading {SOURCE_URL} -> {cache_file}")
        with requests.get(SOURCE_URL, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for block in response.iter_content(1 << 20):
                    if not block:
                        continue
                    output.write(block)
                    digest.update(block)
                    size += len(block)

        actual_sha256 = digest.hexdigest()
        if size != ARCHIVE_SIZE or actual_sha256 != ARCHIVE_SHA256:
            raise SystemExit(
                "Speech Commands download failed integrity check: "
                f"got {size} bytes, sha256 {actual_sha256}; expected "
                f"{ARCHIVE_SIZE} bytes, sha256 {ARCHIVE_SHA256}")
        temporary.replace(cache_file)
    finally:
        if temporary.exists():
            temporary.unlink()
    return cache_file


def resample_pcm16le_4khz(frames: bytes) -> np.ndarray:
    """Return float64 4 kHz samples from signed 16 kHz PCM16-LE frames."""
    return resample_pcm16le(frames, RESAMPLE_RECIPES[4])


def resample_pcm16le_8khz(frames: bytes) -> np.ndarray:
    """Return float64 8 kHz samples from signed 16 kHz PCM16-LE frames."""
    return resample_pcm16le(frames, RESAMPLE_RECIPES[2])


def pcm16le_to_pcm8_4khz(frames: bytes) -> bytes:
    """Anti-alias, decimate 16 kHz PCM16 4:1, then quantize to unsigned PCM8.

    Filtering happens in signed PCM16 amplitude space. Constant padding is
    signed zero, the centered polyphase result is clipped to the PCM16 range,
    and the existing high-byte quantizer is extended to float samples as
    ``floor((sample + 32768) / 256)``.
    """
    resampled = resample_pcm16le_4khz(frames)
    return linear_pcm_to_pcm8(resampled)


def pcm16le_to_pcm8_8khz(frames: bytes) -> bytes:
    """Anti-alias, decimate 16 kHz PCM16 2:1, then quantize to unsigned PCM8."""
    resampled = resample_pcm16le_8khz(frames)
    return linear_pcm_to_pcm8(resampled)


def _pcm8_from_wav(wav_bytes: bytes, name: str,
                   expected_frames: int | None,
                   spec: DatasetSpec = DATASETS[STEM]) -> bytes:
    frames = read_pcm16_wav(wav_bytes, name, sample_rate_hz=SAMPLE_RATE_HZ,
                            expected_frames=expected_frames)
    return pcm8_from_pcm16_frames(frames, name, spec.stem, spec.resample_down)


def load_pcm8_clips(archive_path: Path, *, expected_clips: int | None = None,
                    expected_frames: int | None = None,
                    spec: DatasetSpec = DATASETS[STEM]
                    ) -> list[tuple[str, bytes]]:
    """Read and validate WAV members without extracting them to the filesystem."""
    archive_path = Path(archive_path)
    clips = []
    seen = set()
    try:
        # Stream the gzip member once. Random access in a compressed tar would
        # repeatedly decompress from the beginning when clips are later mixed.
        with tarfile.open(archive_path, "r|gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                name = normalized_member_name(member.name)
                if not name.lower().endswith(".wav"):
                    continue
                if name in seen:
                    raise ValueError(f"duplicate WAV archive member: {name}")
                seen.add(name)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"could not read WAV archive member: {name}")
                clips.append((
                    name,
                    _pcm8_from_wav(source.read(), name, expected_frames, spec),
                ))
    except tarfile.TarError as exc:
        raise ValueError(f"{archive_path}: invalid tar archive: {exc}") from exc

    if expected_clips is not None and len(clips) != expected_clips:
        raise ValueError(
            f"archive has {len(clips)} WAV clips != expected {expected_clips}")
    if not clips:
        raise ValueError("archive contains no WAV clips")
    clips.sort(key=clip_sort_key)
    return clips


def prepare_archive(archive_path: Path, context_length: int, out_path: Path,
                    num_sequences: int = DEFAULT_NUM_SEQUENCES,
                    overwrite: bool = False, *,
                    expected_clips: int | None = None,
                    expected_frames: int | None = None,
                    spec: DatasetSpec = DATASETS[STEM]) -> int:
    """Bake exact windows from a local Speech Commands-style WAV archive."""
    window = window_from_context(context_length)
    if num_sequences < 0:
        raise SystemExit(
            f"--num-sequences must be >= 0, got {num_sequences}")

    clips = load_pcm8_clips(
        archive_path,
        expected_clips=expected_clips,
        expected_frames=expected_frames,
        spec=spec,
    )
    source_archive_sha256 = (
        sha256_file(archive_path) if spec.resample_down != 1 else None
    )
    recipe = (
        RESAMPLE_RECIPES.get(spec.resample_down)
        if spec.resample_down != 1 else None
    )
    if spec.resample_down != 1 and recipe is None:
        raise AssertionError(
            f"{spec.stem}: unsupported resample factor {spec.resample_down}")
    plan = plan_clip_windows(clips, window, num_sequences, label=spec.stem)

    def records():
        for index, (clip, chunk_index, window_bytes) in enumerate(
                iter_capped_windows(plan, window)):
            sequence = list(window_bytes)
            metadata = {
                "source": SOURCE,
                "archive_member": clip.name,
                "license": SOURCE_LICENSE,
                "encoding": "unsigned_linear_pcm8",
                "sample_rate_hz": spec.sample_rate_hz,
                "transform": spec.transform,
                "chunk_index": chunk_index,
            }
            if spec.resample_down != 1:
                assert recipe is not None
                metadata.update({
                    "source_sample_rate_hz": SAMPLE_RATE_HZ,
                    "resample_up": 1,
                    "resample_down": spec.resample_down,
                    "resample_filter": recipe.filter_name,
                    "resample_filter_sha256": recipe.filter_sha256,
                    "resample_padtype": "constant_zero",
                    "source_archive_sha256": source_archive_sha256,
                    "quantization":
                        "clip_pcm16_floor_offset_div256",
                })
            yield make_record(
                spec.stem, sequence, index, context_length, metadata)

    return write_bake(
        out_path, records(), context_length, overwrite=overwrite)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Bake Speech Commands v0.02 test audio as linear PCM8.")
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(DATASETS), default=[STEM],
        metavar="STEM",
        help="Dataset stem(s) to bake (default: speech_commands_pcm8)")
    parser.add_argument(
        "--context-length", type=int, required=True,
        help="Target context (prefix + PCM8 samples); the baked window is "
             "context_length - 1")
    parser.add_argument(
        "--output", default=None,
        help="Output path (valid only with exactly one --datasets entry)")
    parser.add_argument(
        "--num-sequences", type=int, default=DEFAULT_NUM_SEQUENCES,
        help=f"Deterministic clip-balanced cap (default: "
             f"{DEFAULT_NUM_SEQUENCES}; 0 = all full windows)")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite an existing output file")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    window = window_from_context(args.context_length)
    if args.num_sequences < 0:
        raise SystemExit(
            f"--num-sequences must be >= 0, got {args.num_sequences}")
    if len(set(args.datasets)) != len(args.datasets):
        raise SystemExit("--datasets entries must be unique")
    if args.output and len(args.datasets) != 1:
        raise SystemExit(
            "--output is only valid with exactly one --datasets entry")

    specs = [DATASETS[stem] for stem in args.datasets]
    jobs = []
    for spec in specs:
        stem = spec.stem
        if window > spec.expected_output_frames:
            raise SystemExit(
                f"{stem}: eval window {window} exceeds each transformed "
                f"clip's {spec.expected_output_frames} PCM8 samples")
        out_path = (Path(args.output) if args.output
                    else bake_output_path(stem, args.context_length))
        jobs.append((spec, out_path))

    targets = [path.resolve() for _, path in jobs]
    if len(set(targets)) != len(targets):
        raise SystemExit(
            f"selected datasets resolve to duplicate outputs: {targets}")
    if not args.overwrite:
        existing = [path for _, path in jobs if path.exists()]
        if existing:
            raise SystemExit(
                f"{existing[0]} already exists; pass --overwrite to replace it")

    archive_path = ensure_archive()
    for spec, out_path in jobs:
        print(f"\nDataset: {spec.stem}")
        prepare_archive(
            archive_path,
            args.context_length,
            out_path,
            num_sequences=args.num_sequences,
            overwrite=args.overwrite,
            expected_clips=EXPECTED_CLIPS,
            expected_frames=EXPECTED_FRAMES,
            spec=spec,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
