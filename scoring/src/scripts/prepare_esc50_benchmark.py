"""Bake ESC-50 environmental sound clips as headerless linear PCM8 bytes.

ESC-50 (Piczak, "ESC: Dataset for Environmental Sound Classification", ACM
Multimedia 2015, doi:10.1145/2733373.2806390) is 2,000 five-second clips of
environmental sound in 50 classes across five categories (animals, natural
soundscapes, human non-speech, interior/domestic, exterior/urban), stored as
mono 44.1 kHz PCM16 WAV in the ``audio/`` folder of the GitHub repository
``karolpiczak/ESC-50``; ``meta/esc50.csv`` labels each clip. The bake fetches
the repository archive at a pinned commit, caches the zip verbatim under
``raw_sources/`` (size + sha256), and additionally pins a content fingerprint
``CLIPS_SHA256`` over the decoded PCM of every clip, so the benchmark survives
GitHub regenerating the archive container.

Samples become unsigned PCM8 exactly as the Speech Commands bake does::

    pcm8 = pcm16le_high_byte ^ 0x80

The ``esc50_pcm8_11khz`` sibling first applies the same frozen 4:1
anti-aliasing polyphase FIR (its cutoff is relative to the source Nyquist, so
44.1 kHz -> 11.025 kHz with a 5,512 Hz cutoff) and then quantizes with the
float extension of the same map.

WAV headers and labels are not part of the scored bytes. Clips are never
concatenated: each is divided into exact eval windows and its short tail is
dropped. A deterministic clip-balanced cap (``audio_bake_utils``) keeps the
default bake small while mixing clips, classes and temporal positions.

License: Creative Commons Attribution Non-Commercial 3.0 (CC BY-NC 3.0).

Usage (from project root):
    python -m src.scripts.prepare_esc50_benchmark --context-length 256
    python -m src.scripts.prepare_esc50_benchmark \
        --datasets esc50_pcm8_11khz --context-length 256
"""

import argparse
import csv
import hashlib
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests

from src.scripts.audio_bake_utils import (RESAMPLE_RECIPES, clip_sort_key,
                                          filter_name_for, iter_capped_windows,
                                          normalized_member_name,
                                          pcm8_from_pcm16_frames,
                                          plan_clip_windows, read_pcm16_wav,
                                          sha256_file)
from src.scripts.bake_utils import (RAW_SOURCES_DIR, bake_output_path,
                                    make_record, window_from_context,
                                    write_bake)

STEM = "esc50_pcm8"
STEM_11KHZ = "esc50_pcm8_11khz"
SOURCE = "esc50"
SOURCE_LICENSE = "CC-BY-NC-3.0"
SOURCE_CITATION = ("Piczak. ESC: Dataset for Environmental Sound "
                   "Classification. ACM Multimedia 2015")

ARCHIVE_REVISION = "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"
ARCHIVE_URL = (
    f"https://github.com/karolpiczak/ESC-50/archive/{ARCHIVE_REVISION}.zip"
)
ARCHIVE_CACHE = RAW_SOURCES_DIR / f"esc50_{ARCHIVE_REVISION[:12]}.zip"
ARCHIVE_SIZE = 645_832_161
ARCHIVE_SHA256 = (
    "661183a6f53ef04f12c9bd618fed0ddc1713280d6c94a5a5431e844ba6f6a21f"
)
# sha256 over ``name + b"\0" + pcm16_frames`` for every clip in sorted member
# order; the semantic pin of the benchmark, independent of the zip container.
CLIPS_SHA256 = (
    "f6ec8a2916b6eaf6c3a24cc7427eee203bc7d4972d6217c95f44c7d8810ecb34"
)
AUDIO_DIR = "audio/"
META_MEMBER = "meta/esc50.csv"
EXPECTED_CLIPS = 2_000
EXPECTED_FRAMES = 220_500
SAMPLE_RATE_HZ = 44_100
DEFAULT_NUM_SEQUENCES = 8_192
_CLIP_NAME = re.compile(r"^(\d)-(\d+)-([A-Z])-(\d{1,2})\.wav$")


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
    STEM_11KHZ: DatasetSpec(
        stem=STEM_11KHZ,
        sample_rate_hz=11_025,
        expected_output_frames=55_125,
        transform="pcm16le_fir_resample_poly_1_4_then_unsigned_pcm8",
        resample_down=4,
    ),
}


@dataclass(frozen=True)
class ClipInfo:
    fold: int
    clip_id: int
    take: str
    target: int
    category: str
    esc10: bool


def _archive_matches(path: Path) -> bool:
    return (path.is_file()
            and path.stat().st_size == ARCHIVE_SIZE
            and sha256_file(path) == ARCHIVE_SHA256)


def ensure_archive(cache_file: Path = ARCHIVE_CACHE) -> Path:
    """Return the checksum-pinned repository archive, downloading it if needed."""
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
        print(f"  downloading {ARCHIVE_URL} -> {cache_file}")
        with requests.get(ARCHIVE_URL, stream=True, timeout=(30, 300)) as response:
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
                "ESC-50 download failed integrity check: "
                f"got {size} bytes, sha256 {actual_sha256}; expected "
                f"{ARCHIVE_SIZE} bytes, sha256 {ARCHIVE_SHA256}. GitHub may "
                "have regenerated the archive container; if CLIPS_SHA256 still "
                "matches, update ARCHIVE_SIZE/ARCHIVE_SHA256.")
        temporary.replace(cache_file)
    finally:
        if temporary.exists():
            temporary.unlink()
    return cache_file


def parse_clip_name(name: str) -> tuple[int, int, str, int]:
    """``{FOLD}-{CLIP_ID}-{TAKE}-{TARGET}.wav`` -> (fold, clip_id, take, target)."""
    match = _CLIP_NAME.match(name)
    if match is None:
        raise ValueError(f"{name}: not an ESC-50 clip filename")
    fold, clip_id, take, target = match.groups()
    return int(fold), int(clip_id), take, int(target)


def _split_top_level(name: str) -> tuple[str, str]:
    top, _, rest = name.partition("/")
    if not rest:
        raise ValueError(f"archive member {name!r} is not under a top-level "
                         "directory")
    return top, rest


def load_meta(csv_bytes: bytes) -> dict[str, ClipInfo]:
    """Parse ``meta/esc50.csv`` into ``filename -> ClipInfo``."""
    rows = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8")))
    required = {"filename", "fold", "target", "category", "esc10", "take"}
    if rows.fieldnames is None or not required <= set(rows.fieldnames):
        raise ValueError(f"{META_MEMBER}: missing columns "
                         f"{sorted(required - set(rows.fieldnames or []))}")
    meta = {}
    for row in rows:
        filename = row["filename"]
        if filename in meta:
            raise ValueError(f"{META_MEMBER}: duplicate row for {filename}")
        fold, clip_id, take, target = parse_clip_name(filename)
        if (int(row["fold"]), row["take"], int(row["target"])) != (
                fold, take, target):
            raise ValueError(
                f"{META_MEMBER}: {filename} row (fold {row['fold']}, take "
                f"{row['take']}, target {row['target']}) disagrees with the "
                "filename")
        meta[filename] = ClipInfo(
            fold=fold, clip_id=clip_id, take=take, target=target,
            category=row["category"],
            esc10=row["esc10"].strip().lower() == "true",
        )
    if not meta:
        raise ValueError(f"{META_MEMBER}: no rows")
    return meta


def load_pcm8_clips(archive_path: Path, *,
                    expected_clips: int | None = None,
                    expected_frames: int | None = None,
                    expected_clips_sha256: str | None = None,
                    spec: DatasetSpec = DATASETS[STEM]
                    ) -> tuple[list[tuple[str, bytes]], dict[str, ClipInfo], str]:
    """Read every ``audio/*.wav`` member (no extraction) and the label csv.

    Returns ``(clips, meta, clips_sha256)``: clips are ``(name, pcm8)`` with
    ``name`` the member path relative to the archive's top-level directory
    (``audio/1-100032-A-0.wav``), in the shared deterministic clip order.
    """
    archive_path = Path(archive_path)
    frames_by_name: dict[str, bytes] = {}
    meta_bytes = None
    top_levels = set()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                full_name = normalized_member_name(info.filename)
                top, rest = _split_top_level(full_name)
                top_levels.add(top)
                if rest == META_MEMBER:
                    meta_bytes = archive.read(info)
                    continue
                if not (rest.startswith(AUDIO_DIR)
                        and rest.lower().endswith(".wav")):
                    continue
                if rest in frames_by_name:
                    raise ValueError(f"duplicate WAV archive member: {rest}")
                frames_by_name[rest] = read_pcm16_wav(
                    archive.read(info), rest, sample_rate_hz=SAMPLE_RATE_HZ,
                    expected_frames=expected_frames)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{archive_path}: invalid zip archive: {exc}") from exc

    if len(top_levels) != 1:
        raise ValueError(
            f"{archive_path}: expected one top-level directory, found "
            f"{sorted(top_levels)}")
    if meta_bytes is None:
        raise ValueError(f"{archive_path}: missing {META_MEMBER}")
    if not frames_by_name:
        raise ValueError("archive contains no WAV clips")
    if expected_clips is not None and len(frames_by_name) != expected_clips:
        raise ValueError(
            f"archive has {len(frames_by_name)} WAV clips != expected "
            f"{expected_clips}")

    meta = load_meta(meta_bytes)
    fingerprint = hashlib.sha256()
    for name in sorted(frames_by_name):
        filename = name[len(AUDIO_DIR):]
        if filename not in meta:
            raise ValueError(f"{name}: no {META_MEMBER} row")
        parse_clip_name(filename)
        fingerprint.update(name.encode("utf-8"))
        fingerprint.update(b"\0")
        fingerprint.update(frames_by_name[name])
    clips_sha256 = fingerprint.hexdigest()
    if (expected_clips_sha256 is not None
            and clips_sha256 != expected_clips_sha256):
        raise ValueError(
            f"{archive_path}: clip content fingerprint {clips_sha256} != "
            f"pinned CLIPS_SHA256 {expected_clips_sha256}")

    clips = [
        (name, pcm8_from_pcm16_frames(frames, name, spec.stem,
                                      spec.resample_down))
        for name, frames in frames_by_name.items()
    ]
    clips.sort(key=clip_sort_key)
    return clips, meta, clips_sha256


def prepare_archive(archive_path: Path, context_length: int, out_path: Path,
                    num_sequences: int = DEFAULT_NUM_SEQUENCES,
                    overwrite: bool = False, *,
                    expected_clips: int | None = None,
                    expected_frames: int | None = None,
                    expected_clips_sha256: str | None = None,
                    spec: DatasetSpec = DATASETS[STEM]) -> int:
    """Bake exact windows from a local ESC-50 repository archive."""
    window = window_from_context(context_length)
    if num_sequences < 0:
        raise SystemExit(
            f"--num-sequences must be >= 0, got {num_sequences}")

    clips, meta, clips_sha256 = load_pcm8_clips(
        archive_path,
        expected_clips=expected_clips,
        expected_frames=expected_frames,
        expected_clips_sha256=expected_clips_sha256,
        spec=spec,
    )
    print(f"  {spec.stem}: clip content fingerprint {clips_sha256}")
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
            info = meta[clip.name[len(AUDIO_DIR):]]
            metadata = {
                "source": SOURCE,
                "source_revision": ARCHIVE_REVISION,
                "archive_member": clip.name,
                "license": SOURCE_LICENSE,
                "encoding": "unsigned_linear_pcm8",
                "sample_rate_hz": spec.sample_rate_hz,
                "transform": spec.transform,
                "fold": info.fold,
                "clip_id": info.clip_id,
                "take": info.take,
                "target": info.target,
                "category": info.category,
                "esc10": info.esc10,
                "chunk_index": chunk_index,
            }
            if spec.resample_down != 1:
                assert recipe is not None
                metadata.update({
                    "source_sample_rate_hz": SAMPLE_RATE_HZ,
                    "resample_up": 1,
                    "resample_down": spec.resample_down,
                    "resample_filter": filter_name_for(recipe, SAMPLE_RATE_HZ),
                    "resample_filter_sha256": recipe.filter_sha256,
                    "resample_padtype": "constant_zero",
                    "source_clips_sha256": clips_sha256,
                    "quantization":
                        "clip_pcm16_floor_offset_div256",
                })
            yield make_record(
                spec.stem, list(window_bytes), index, context_length, metadata)

    return write_bake(
        out_path, records(), context_length, overwrite=overwrite)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Bake ESC-50 environmental sound clips as linear PCM8.")
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(DATASETS), default=[STEM],
        metavar="STEM",
        help=f"Dataset stem(s) to bake (default: {STEM})")
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

    jobs = []
    for stem in args.datasets:
        spec = DATASETS[stem]
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
            expected_clips_sha256=CLIPS_SHA256,
            spec=spec,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
