"""Bake the MusicNet test recordings as headerless linear PCM8 bytes.

MusicNet (Thickstun, Harchaoui, Kakade, "Learning Features of Music from
Scratch", ICLR 2017; Zenodo record 5120004, CC BY 4.0) is 330 freely licensed
recordings of polyphonic classical music at 44.1 kHz. Only the ten recordings
of the canonical test split (``musicnet/test_data/<id>.wav``) are scored here.

The only archive Zenodo offers is the full 11.1 GB ``musicnet.tar.gz``. It is
STREAMED (``tarfile`` ``r|gz`` over the HTTP body) and never written to disk:
each test WAV is verified against the pinned per-recording size + sha256 and
cached under ``raw_sources/musicnet_test_<id>.wav`` (~350 MB total); the
stream stops as soon as every test recording is cached. If it ends without
every test recording, the whole body's size + md5 are compared with the
pinned archive so the error tells a truncated stream from a wrong manifest.
Per the bake_utils convention the fetch is NOT resumable; a missing or
invalid cache file re-streams the archive from scratch. Re-baking at a new
context is network-free.

``musicnet_metadata.csv`` (composer / composition / ensemble per recording)
is pinned by size + md5, cached at ``raw_sources/musicnet_metadata.csv`` and
REQUIRED: every baked record carries those three keys, so a bake that cannot
obtain the verified file fails instead of silently dropping them.

The recordings are mono 44.1 kHz **IEEE float32** WAV (format tag 3, samples
in [-1, 1]; the stdlib ``wave`` module cannot read them, so
``audio_bake_utils.read_float32_wav`` walks the RIFF chunks). Samples become
unsigned PCM8 with the same map the resampled Speech Commands stems use, after
scaling to the PCM16 amplitude range::

    pcm8 = floor((clip(x * 32768, -32768, 32767) + 32768) / 256)

which lands a full-scale float file on the PCM16 high byte ^ 0x80 of its
integer rendering. Levels are NOT normalized (most test recordings peak near
0.2, i.e. ~50 of the 256 levels). ``musicnet_pcm8_11khz`` first applies the
frozen 4:1 anti-aliasing FIR (44.1 kHz -> 11.025 kHz) on the scaled samples.

Recordings are never concatenated: each is divided into exact eval windows
and its tail is dropped. With only ten multi-minute recordings, the plain
clip-balanced cap would emit each recording's ~800 selected windows as one
contiguous run of a few seconds, so this bake uses the ``spread`` cap: each
recording gets exactly the share the round-robin draws from it, spaced evenly
over its whole duration from its hashed start chunk (a fractional stride of
``num_windows / share``), so the default 8,192 windows sample every recording
end to end at any context length.

Usage (from project root):
    python -m src.scripts.prepare_musicnet_benchmark --context-length 256
    python -m src.scripts.prepare_musicnet_benchmark \
        --datasets musicnet_pcm8_11khz --context-length 256
"""

import argparse
import contextlib
import csv
import hashlib
import io
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, ContextManager, Iterable

import requests

from src.scripts.audio_bake_utils import (RESAMPLE_RECIPES, clip_sort_key,
                                          filter_name_for, iter_capped_windows,
                                          normalized_member_name,
                                          pcm8_from_float_samples,
                                          plan_clip_windows, read_float32_wav,
                                          sha256_file)
from src.scripts.bake_utils import (RAW_SOURCES_DIR, bake_output_path,
                                    make_record, window_from_context,
                                    write_bake)

STEM = "musicnet_pcm8"
STEM_11KHZ = "musicnet_pcm8_11khz"
SOURCE = "musicnet_test"
SOURCE_LICENSE = "CC-BY-4.0"
SOURCE_CITATION = ("Thickstun, Harchaoui, Kakade. Learning Features of Music "
                   "from Scratch. ICLR 2017")

ZENODO_RECORD = "5120004"
ARCHIVE_URL = (
    f"https://zenodo.org/records/{ZENODO_RECORD}/files/musicnet.tar.gz"
    "?download=1"
)
ARCHIVE_SIZE = 11_097_394_998
ARCHIVE_MD5 = "844764911fa0d5b97c97da944a057590"
METADATA_URL = (
    f"https://zenodo.org/records/{ZENODO_RECORD}/files/musicnet_metadata.csv"
    "?download=1"
)
METADATA_SIZE = 43_775
METADATA_MD5 = "1caef62cee9c875235e62aac368b49d8"
METADATA_CACHE = RAW_SOURCES_DIR / "musicnet_metadata.csv"
TEST_DIR = "musicnet/test_data/"
SAMPLE_RATE_HZ = 44_100
DEFAULT_NUM_SEQUENCES = 8_192
PROGRESS_BYTES = 256 << 20


@dataclass(frozen=True)
class RecordingSpec:
    recording_id: str
    size: int | None
    sha256: str | None

    @property
    def pinned(self) -> bool:
        return self.size is not None and self.sha256 is not None


# The ten canonical MusicNet test recordings. Sizes/digests are the verified
# ``musicnet/test_data/<id>.wav`` members of the pinned archive; a ``None``
# pin makes the first stream print the actual values
# and refuse to bake.
TEST_RECORDINGS: dict[str, RecordingSpec] = {
    "1759": RecordingSpec("1759", 34_338_874, "3571d0dafb1ab7635f35cd7b84965dc35f2821d831518bf321711d52e9a17ec2"),
    "1819": RecordingSpec("1819", 31_306_810, "15f460aede5f09a96126ae2948033a88594a5c33abd18fa999b76c932974e0d8"),
    "2106": RecordingSpec("2106", 39_982_138, "0889597dbd1016023d12897167db039f28528cd69626ef728c00ef8c4526197f"),
    "2191": RecordingSpec("2191", 18_141_754, "34a3c1df0e377007a21b639a59090fbdf8b694b003afc2a6cbc704868a3f19ca"),
    "2298": RecordingSpec("2298", 27_081_274, "f681f82fd4a5e4365d450a3def9affc2d46c2ebbe10b58e6f72f85426ad8b3aa"),
    "2303": RecordingSpec("2303", 16_344_634, "6c233b3641a84a1755c683a6205e720805a609beec0d70c9b5e33c5c823fda10"),
    "2382": RecordingSpec("2382", 20_814_394, "b8bbe4b8f9f0c2946b4693d347886dca426c6cbcf70463061513506c690865b2"),
    "2416": RecordingSpec("2416", 24_542_266, "24b416eabea3416cd9e2dfc95aa023a998fae64ef7930c55130e169f0f363d6b"),
    "2556": RecordingSpec("2556", 26_717_242, "0881669b1db9b688526ef41bef35e1895209058be3d719b30d1fb5665d3f8328"),
    "2628": RecordingSpec("2628", 21_994_042, "28f2d0f8a0807b1a2ed4f3cd0b3f5072ff482f843d8b7c3d7236d380cea20caa"),
}


@dataclass(frozen=True)
class DatasetSpec:
    stem: str
    sample_rate_hz: int
    transform: str
    resample_down: int = 1


DATASETS = {
    STEM: DatasetSpec(
        stem=STEM,
        sample_rate_hz=SAMPLE_RATE_HZ,
        transform="float32_times_32768_then_unsigned_pcm8",
    ),
    STEM_11KHZ: DatasetSpec(
        stem=STEM_11KHZ,
        sample_rate_hz=11_025,
        transform="float32_times_32768_fir_resample_poly_1_4_then_unsigned_pcm8",
        resample_down=4,
    ),
}
SOURCE_FORMAT = "wav_ieee_float32_mono_44100hz"


def cache_path(recording_id: str, cache_dir: Path = RAW_SOURCES_DIR) -> Path:
    return Path(cache_dir) / f"musicnet_test_{recording_id}.wav"


def _cache_matches(cache_file: Path, spec: RecordingSpec) -> bool:
    cache_file = Path(cache_file)
    if not spec.pinned:
        return False
    if not cache_file.is_file() or cache_file.stat().st_size != spec.size:
        return False
    return sha256_file(cache_file) == spec.sha256


class _HashingStream:
    """Wrap a raw HTTP body so tarfile's reads are md5-hashed and counted."""

    def __init__(self, raw):
        self._raw = raw
        self.md5 = hashlib.md5()
        self.size = 0
        self._next_report = PROGRESS_BYTES

    def read(self, n: int = -1) -> bytes:
        data = self._raw.read(n) if n is not None and n >= 0 else self._raw.read()
        self.md5.update(data)
        self.size += len(data)
        if self.size >= self._next_report:
            print(f"    streamed {self.size >> 20} MiB")
            self._next_report += PROGRESS_BYTES
        return data


@contextlib.contextmanager
def open_archive_stream():
    with requests.get(ARCHIVE_URL, stream=True, timeout=(30, 600)) as response:
        response.raise_for_status()
        response.raw.decode_content = False
        yield response.raw


def _recording_id(name: str) -> str:
    return PurePosixPath(name).stem


def stream_test_wavs(cache_dir: Path, wanted: Iterable[str], *,
                     opener: Callable[[], ContextManager] = open_archive_stream,
                     recordings: dict[str, RecordingSpec] | None = None,
                     archive_pin: tuple[int, str] = (ARCHIVE_SIZE, ARCHIVE_MD5),
                     ) -> dict[str, Path]:
    """Stream the archive once, caching every ``test_data/*.wav`` member.

    Stops early once all ``wanted`` ids are cached. Members whose spec is
    pinned are verified (size + sha256) before their cache file is committed;
    unpinned members are cached and their actual size/sha256 reported so the
    manifest can be pinned. Returns ``id -> cache path`` for the cached ids.
    If the archive ends without every wanted id, the rest of the body is
    drained and its size + md5 compared with ``archive_pin`` so the error
    distinguishes a truncated/altered stream from a manifest that names a
    recording the pinned archive does not hold.
    """
    recordings = TEST_RECORDINGS if recordings is None else recordings
    wanted = set(wanted)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached: dict[str, Path] = {}
    unpinned: dict[str, tuple[int, str]] = {}
    print(f"  streaming {ARCHIVE_URL} (only {TEST_DIR}*.wav is kept)")
    with opener() as raw:
        stream = _HashingStream(raw)
        try:
            with tarfile.open(fileobj=stream, mode="r|gz") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    name = normalized_member_name(member.name)
                    if not (name.startswith(TEST_DIR)
                            and name.lower().endswith(".wav")):
                        continue
                    rid = _recording_id(name)
                    if rid not in recordings:
                        raise SystemExit(
                            f"unexpected test recording {name}; the manifest "
                            f"lists {sorted(recordings)}")
                    source = archive.extractfile(member)
                    if source is None:
                        raise ValueError(f"could not read archive member {name}")
                    data = source.read()
                    spec = recordings[rid]
                    digest = hashlib.sha256(data).hexdigest()
                    if spec.pinned and (len(data), digest) != (
                            spec.size, spec.sha256):
                        raise SystemExit(
                            f"{name} failed verification: got {len(data)} "
                            f"bytes, sha256 {digest}; expected {spec.size} "
                            f"bytes, sha256 {spec.sha256}")
                    if not spec.pinned:
                        unpinned[rid] = (len(data), digest)
                    target = cache_path(rid, cache_dir)
                    temporary = target.with_name(target.name + ".tmp")
                    temporary.write_bytes(data)
                    temporary.replace(target)
                    cached[rid] = target
                    print(f"  cached {name}: {target} ({len(data)} bytes)")
                    if wanted <= set(cached):
                        break
        except tarfile.TarError as exc:
            raise ValueError(f"invalid tar stream: {exc}") from exc
        missing = wanted - set(cached)
        if missing:
            # tarfile stops at the end-of-archive marker; read the rest of
            # the body so size/md5 describe the whole file Zenodo pins.
            while stream.read(1 << 20):
                pass

    if missing:
        raise SystemExit(
            f"archive stream ended without test recordings {sorted(missing)}: "
            + _describe_stream_end(stream.size, stream.md5.hexdigest(),
                                   archive_pin))
    if unpinned:
        listing = "\n".join(
            f'    "{rid}": RecordingSpec("{rid}", {size}, "{digest}"),'
            for rid, (size, digest) in sorted(unpinned.items()))
        raise SystemExit(
            "MusicNet test recordings are cached but unpinned; paste these "
            "into TEST_RECORDINGS and re-run (the cache will then verify "
            f"without a second stream):\n{listing}")
    return cached


def _describe_stream_end(size: int, md5: str, pin: tuple[int, str]) -> str:
    expected_size, expected_md5 = pin
    if (size, md5) == pin:
        return (f"the full stream matched the pinned musicnet.tar.gz ({size} "
                f"bytes, md5 {md5}), so the pinned archive does not contain "
                "them; fix TEST_RECORDINGS (or the archive pin)")
    if size < expected_size:
        return (f"the stream was truncated (got {size} bytes, expected "
                f"{expected_size}); re-run to stream the archive again")
    return (f"the stream does not match the pinned archive (got {size} bytes, "
            f"md5 {md5}; expected {expected_size} bytes, md5 {expected_md5}); "
            "the archive was altered or the pin is stale")


def ensure_raw_sources(cache_dir: Path = RAW_SOURCES_DIR, *,
                       opener: Callable[[], ContextManager] = open_archive_stream,
                       archive_pin: tuple[int, str] = (ARCHIVE_SIZE, ARCHIVE_MD5),
                       ) -> dict[str, Path]:
    """Return verified per-recording caches, streaming the archive at most once."""
    paths = {}
    missing = []
    for rid, spec in TEST_RECORDINGS.items():
        cache_file = cache_path(rid, cache_dir)
        if _cache_matches(cache_file, spec):
            print(f"  raw cache hit: {cache_file} ({spec.size} bytes, verified)")
            paths[rid] = cache_file
        else:
            print(f"  raw cache miss or invalid: {cache_file}")
            missing.append(rid)
    if missing:
        paths.update(stream_test_wavs(cache_dir, missing, opener=opener,
                                      archive_pin=archive_pin))
    return paths


METADATA_COLUMNS = ("id", "composer", "composition", "ensemble")


def fetch_metadata() -> bytes:
    """Download ``musicnet_metadata.csv`` from Zenodo."""
    response = requests.get(METADATA_URL, timeout=60)
    response.raise_for_status()
    return response.content


def _metadata_matches(cache_file: Path, pin: tuple[int, str]) -> bool:
    return (cache_file.is_file()
            and cache_file.stat().st_size == pin[0]
            and hashlib.md5(cache_file.read_bytes()).hexdigest() == pin[1])


def ensure_metadata(cache_file: Path = METADATA_CACHE, *,
                    fetch: Callable[[], bytes] = fetch_metadata,
                    pin: tuple[int, str] = (METADATA_SIZE, METADATA_MD5),
                    ) -> dict[str, dict[str, str]]:
    """Return the pinned ``musicnet_metadata.csv`` rows keyed by recording id.

    The file is REQUIRED (every baked record carries composer / composition /
    ensemble). A valid cache (size + md5) is a hit; an absent or invalid cache
    is re-downloaded, verified against ``pin`` and only then written. Any
    failure to obtain the verified file raises ``SystemExit``.
    """
    cache_file = Path(cache_file)
    expected_size, expected_md5 = pin
    if _metadata_matches(cache_file, pin):
        print(f"  raw cache hit: {cache_file} ({expected_size} bytes, verified)")
    else:
        if cache_file.exists():
            print(f"  invalid metadata cache: refreshing {cache_file}")
        print(f"  downloading {METADATA_URL} -> {cache_file}")
        try:
            data = fetch()
        except requests.RequestException as exc:
            raise SystemExit(
                "musicnet_metadata.csv is required for a reproducible bake, "
                f"but {cache_file} is missing or invalid and the download "
                f"from {METADATA_URL} failed ({exc}); place a verified copy "
                f"({expected_size} bytes, md5 {expected_md5}) at {cache_file} "
                "and re-run") from exc
        digest = hashlib.md5(data).hexdigest()
        if (len(data), digest) != pin:
            raise SystemExit(
                f"musicnet_metadata.csv failed integrity check: got "
                f"{len(data)} bytes, md5 {digest}; expected {expected_size} "
                f"bytes, md5 {expected_md5}; {cache_file} was not updated")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_file.with_name(cache_file.name + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(cache_file)
    reader = csv.DictReader(io.StringIO(cache_file.read_text("utf-8")))
    columns = reader.fieldnames or []
    if not set(METADATA_COLUMNS) <= set(columns):
        raise SystemExit(
            f"{cache_file}: expected columns {', '.join(METADATA_COLUMNS)}; "
            f"got {columns}")
    return {row["id"]: row for row in reader}


def load_pcm8_recordings(sources: dict[str, Path], *,
                         spec: DatasetSpec = DATASETS[STEM]
                         ) -> list[tuple[str, bytes]]:
    """Decode the cached float32 test WAVs (loud format check) into PCM8."""
    recordings = []
    for rid, path in sources.items():
        name = f"{TEST_DIR}{rid}.wav"
        samples = read_float32_wav(Path(path).read_bytes(), name,
                                   sample_rate_hz=SAMPLE_RATE_HZ)
        recordings.append((rid, pcm8_from_float_samples(
            samples, name, spec.stem, spec.resample_down)))
    if not recordings:
        raise ValueError("no MusicNet test recordings to bake")
    recordings.sort(key=clip_sort_key)
    return recordings


def prepare_recordings(sources: dict[str, Path], context_length: int,
                       out_path: Path,
                       num_sequences: int = DEFAULT_NUM_SEQUENCES,
                       overwrite: bool = False, *,
                       spec: DatasetSpec = DATASETS[STEM],
                       metadata: dict[str, dict],
                       recordings: dict[str, RecordingSpec] | None = None
                       ) -> int:
    """Bake exact, evenly spread windows from the cached test recordings.

    ``metadata`` (from :func:`ensure_metadata`) must hold a row for every
    recording in ``sources``; each baked record carries its composer /
    composition / ensemble.
    """
    window = window_from_context(context_length)
    if num_sequences < 0:
        raise SystemExit(
            f"--num-sequences must be >= 0, got {num_sequences}")
    recordings = TEST_RECORDINGS if recordings is None else recordings
    without_rows = sorted(rid for rid in sources if rid not in metadata)
    if without_rows:
        raise SystemExit(
            f"{spec.stem}: musicnet_metadata.csv has no rows for test "
            f"recordings {without_rows}; every baked record must carry "
            "composer/composition/ensemble")

    clips = load_pcm8_recordings(sources, spec=spec)
    recipe = (
        RESAMPLE_RECIPES.get(spec.resample_down)
        if spec.resample_down != 1 else None
    )
    if spec.resample_down != 1 and recipe is None:
        raise AssertionError(
            f"{spec.stem}: unsupported resample factor {spec.resample_down}")
    plan = plan_clip_windows(clips, window, num_sequences, label=spec.stem,
                             spread=True)

    def records():
        for index, (clip, chunk_index, window_bytes) in enumerate(
                iter_capped_windows(plan, window)):
            rid = clip.name
            record_metadata = {
                "source": SOURCE,
                "zenodo_record": ZENODO_RECORD,
                "recording_id": rid,
                "source_sha256": recordings[rid].sha256,
                "source_format": SOURCE_FORMAT,
                "license": SOURCE_LICENSE,
                "encoding": "unsigned_linear_pcm8",
                "sample_rate_hz": spec.sample_rate_hz,
                "transform": spec.transform,
                "quantization": "clip_pcm16_floor_offset_div256",
                "num_windows": clip.num_windows,
                "chunk_index": chunk_index,
                "cap_strategy": "hashed_start_even_spread",
            }
            info = metadata[rid]
            record_metadata.update({
                "composer": info["composer"],
                "composition": info["composition"],
                "ensemble": info["ensemble"],
            })
            if spec.resample_down != 1:
                assert recipe is not None
                record_metadata.update({
                    "source_sample_rate_hz": SAMPLE_RATE_HZ,
                    "resample_up": 1,
                    "resample_down": spec.resample_down,
                    "resample_filter": filter_name_for(recipe, SAMPLE_RATE_HZ),
                    "resample_filter_sha256": recipe.filter_sha256,
                    "resample_padtype": "constant_zero",
                })
            yield make_record(
                spec.stem, list(window_bytes), index, context_length,
                record_metadata)

    return write_bake(
        out_path, records(), context_length, overwrite=overwrite)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Bake the MusicNet test recordings as linear PCM8.")
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
        help=f"Deterministic evenly-spread cap (default: "
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

    # Metadata first: a bad metadata state must not surface after an 11 GB
    # archive stream.
    metadata = ensure_metadata()
    sources = ensure_raw_sources()
    for spec, out_path in jobs:
        print(f"\nDataset: {spec.stem}")
        prepare_recordings(
            sources,
            args.context_length,
            out_path,
            num_sequences=args.num_sequences,
            overwrite=args.overwrite,
            spec=spec,
            metadata=metadata,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
