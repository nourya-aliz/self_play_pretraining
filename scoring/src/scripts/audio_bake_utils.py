"""Shared helpers for the raw-audio benchmark bakes.

Used by ``prepare_speech_commands_benchmark.py`` (16 kHz speech),
``prepare_esc50_benchmark.py`` (44.1 kHz environmental sound) and
``prepare_musicnet_benchmark.py`` (44.1 kHz classical music). Everything here
was extracted VERBATIM from the Speech Commands bake, so that script's output
is byte-identical to the pre-extraction bake (its frozen tests + the committed
c256 bakes are the acceptance gate for any change in this module).

Conventions shared by every audio stem:

* Scored bytes are headerless **unsigned linear PCM8**, one byte per sample:
  ``pcm8 = pcm16le_high_byte ^ 0x80`` (:func:`pcm16le_to_pcm8`). Resampled
  siblings first apply a FROZEN anti-aliasing polyphase FIR (coefficients are
  stored here and sha256-pinned, never re-designed at bake time) and then
  quantize with the float extension of the same map
  (:func:`linear_pcm_to_pcm8`). The FIR cutoffs are RELATIVE to the source
  Nyquist, so the same taps decimate 16 kHz -> 4 kHz and 44.1 kHz -> 11.025 kHz.
* Recordings are never concatenated: each clip is divided into exact eval
  windows on its own and its short tail is dropped
  (:func:`plan_clip_windows`).
* The default bake is a deterministic clip-balanced cap: clips are ordered by
  ``sha256(name)``, each clip's first window is chosen by its digest, and the
  emitter round-robins across clips (:func:`iter_capped_windows`). The
  ``spread`` option gives each clip exactly the share the round-robin will
  draw from it (:func:`round_robin_counts`) and spaces those picks evenly
  over the clip's whole duration, so a small cap still samples very long
  recordings end to end (MusicNet).
"""

import bisect
import hashlib
import io
import struct
import wave
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterator

import numpy as np
from scipy.signal import resample_poly


@dataclass(frozen=True)
class ResampleRecipe:
    down: int
    filter_name: str
    filter_sha256: str
    taps: np.ndarray
    cutoff_fraction: float  # cutoff relative to the SOURCE Nyquist


# Fixed 81-tap, symmetric low-pass FIR for 16 kHz -> 4 kHz decimation.
# These are the float64 coefficients from:
#   scipy.signal.firwin(
#       81, 0.25, window=("kaiser", 5.0), scale=True)
# where 0.25 is the 2 kHz cutoff relative to the 8 kHz source Nyquist.
# Storing the coefficients, rather than designing them at bake time, makes the
# benchmark independent of future scipy.signal.firwin implementation changes.
PCM8_4KHZ_FIR_TAPS_SHA256 = (
    "5c9034741b5710b11047b48653f92b70cafb67a23c68e16f0f6619e368890669"
)
PCM8_4KHZ_FIR_TAPS = np.asarray((
    -3.5799115519692387e-19, -0.00028264939479611534,
    -0.00052579196542946896, -0.00047541698372319279,
    9.2726027350597403e-19, 0.00073162852724047178,
    0.001254632730867955, 0.0010630797667137762,
    -1.7472805760903997e-18, -0.0014830129609459289,
    -0.0024477084927997232, -0.0020065313653163733,
    2.8035592572493553e-18, 0.002651272519201363,
    0.0042787887572440283, 0.0034384897676579375,
    -4.0459880955907567e-18, -0.0043947913127893607,
    -0.0069958192527395276, -0.0055549713919249392,
    5.3912076618757543e-18, 0.0069667290898802218,
    0.011012871024959065, 0.0086988136875902795,
    -6.7306397836470466e-18, -0.010855771610528709,
    -0.017172133746420341, -0.013606372767195627,
    7.9432457364957951e-18, 0.017243084699543741,
    0.02764743251823136, 0.022320837133011535,
    -8.9111641394119533e-18, -0.030033587538636726,
    -0.050471105705763818, -0.043494435742885106,
    9.5357833560552958e-18, 0.07413570490109378,
    0.15836902171996831, 0.22490814275255208,
    0.25015914127227795,
    0.22490814275255208, 0.15836902171996831,
    0.07413570490109378, 9.5357833560552958e-18,
    -0.043494435742885106, -0.050471105705763818,
    -0.030033587538636726, -8.9111641394119533e-18,
    0.022320837133011535, 0.02764743251823136,
    0.017243084699543741, 7.9432457364957951e-18,
    -0.013606372767195627, -0.017172133746420341,
    -0.010855771610528709, -6.7306397836470466e-18,
    0.0086988136875902795, 0.011012871024959065,
    0.0069667290898802218, 5.3912076618757543e-18,
    -0.0055549713919249392, -0.0069958192527395276,
    -0.0043947913127893607, -4.0459880955907567e-18,
    0.0034384897676579375, 0.0042787887572440283,
    0.002651272519201363, 2.8035592572493553e-18,
    -0.0020065313653163733, -0.0024477084927997232,
    -0.0014830129609459289, -1.7472805760903997e-18,
    0.0010630797667137762, 0.001254632730867955,
    0.00073162852724047178, 9.2726027350597403e-19,
    -0.00047541698372319279, -0.00052579196542946896,
    -0.00028264939479611534, -3.5799115519692387e-19,
), dtype="<f8")
PCM8_4KHZ_FIR_TAPS.setflags(write=False)
_actual_fir_sha256 = hashlib.sha256(
    PCM8_4KHZ_FIR_TAPS.tobytes(order="C")
).hexdigest()
if _actual_fir_sha256 != PCM8_4KHZ_FIR_TAPS_SHA256:
    raise RuntimeError(
        "speech_commands_pcm8_4khz FIR coefficients do not match their "
        f"declared SHA-256: {_actual_fir_sha256}")
del _actual_fir_sha256

# Fixed 41-tap counterpart for 16 kHz -> 8 kHz decimation. These are the
# float64 coefficients from:
#   scipy.signal.firwin(
#       41, 0.5, window=("kaiser", 5.0), scale=True)
# where 0.5 is the 4 kHz cutoff relative to the 8 kHz source Nyquist. The
# factor-specific length follows scipy.signal.resample_poly's design rule and
# gives the same ten-output-sample half-support as the 81-tap 4 kHz filter.
PCM8_8KHZ_FIR_TAPS_SHA256 = (
    "1fa077f5a1e604434e8faafdd981270903e87d640ee6a51f5f7489170f32dff1"
)
PCM8_8KHZ_FIR_TAPS = np.asarray((
    -7.1589709508864583e-19, -0.0010514587726751215,
    1.8542998243319002e-18, 0.0025089668121477146,
    -3.4941452339509673e-18, -0.0048948343392875709,
    5.6064511623748102e-18, 0.008556559002481404,
    -8.0910130944529949e-18, -0.013989973238435148,
    1.0781132014374602e-17, 0.022023120574074611,
    -1.345967742289739e-17, -0.034340179881744919,
    1.5884600682954171e-17, 0.055288283911857014,
    -1.78202070879486e-17, -0.10093019739773382,
    1.9069296838463634e-17, 0.3167003456803007,
    0.50025873529803011,
    0.3167003456803007, 1.9069296838463634e-17,
    -0.10093019739773382, -1.78202070879486e-17,
    0.055288283911857014, 1.5884600682954171e-17,
    -0.034340179881744919, -1.345967742289739e-17,
    0.022023120574074611, 1.0781132014374602e-17,
    -0.013989973238435148, -8.0910130944529949e-18,
    0.008556559002481404, 5.6064511623748102e-18,
    -0.0048948343392875709, -3.4941452339509673e-18,
    0.0025089668121477146, 1.8542998243319002e-18,
    -0.0010514587726751215, -7.1589709508864583e-19,
), dtype="<f8")
PCM8_8KHZ_FIR_TAPS.setflags(write=False)
_actual_fir_sha256 = hashlib.sha256(
    PCM8_8KHZ_FIR_TAPS.tobytes(order="C")
).hexdigest()
if _actual_fir_sha256 != PCM8_8KHZ_FIR_TAPS_SHA256:
    raise RuntimeError(
        "speech_commands_pcm8_8khz FIR coefficients do not match their "
        f"declared SHA-256: {_actual_fir_sha256}")
del _actual_fir_sha256

# ``filter_name`` is the 16 kHz-source label the Speech Commands bake emits
# verbatim; other source rates label the same taps via ``filter_name_for``.
RESAMPLE_RECIPES = {
    2: ResampleRecipe(
        down=2,
        filter_name="fir_41tap_kaiser_beta5_cutoff_4000hz",
        filter_sha256=PCM8_8KHZ_FIR_TAPS_SHA256,
        taps=PCM8_8KHZ_FIR_TAPS,
        cutoff_fraction=0.5,
    ),
    4: ResampleRecipe(
        down=4,
        filter_name="fir_81tap_kaiser_beta5_cutoff_2000hz",
        filter_sha256=PCM8_4KHZ_FIR_TAPS_SHA256,
        taps=PCM8_4KHZ_FIR_TAPS,
        cutoff_fraction=0.25,
    ),
}


def filter_name_for(recipe: ResampleRecipe, source_rate_hz: int) -> str:
    """The recipe's label at a given source rate (16 kHz reproduces
    ``recipe.filter_name`` exactly)."""
    cutoff_hz = int(source_rate_hz / 2 * recipe.cutoff_fraction)
    return f"fir_{len(recipe.taps)}tap_kaiser_beta5_cutoff_{cutoff_hz}hz"


for _recipe in RESAMPLE_RECIPES.values():
    if filter_name_for(_recipe, 16_000) != _recipe.filter_name:
        raise RuntimeError(
            f"resample recipe {_recipe.down}: filter_name "
            f"{_recipe.filter_name!r} != {filter_name_for(_recipe, 16_000)!r}")
del _recipe


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pcm16le_to_pcm8(frames: bytes) -> bytes:
    """Convert signed little-endian PCM16 frames to unsigned linear PCM8."""
    if len(frames) % 2:
        raise ValueError(
            f"PCM16 frame data must have even length, got {len(frames)} bytes")
    return bytes(high_byte ^ 0x80 for high_byte in frames[1::2])


def resample_pcm16le(
    frames: bytes,
    recipe: ResampleRecipe,
) -> np.ndarray:
    """Return float64 decimated samples from signed PCM16-LE frames."""
    if len(frames) % 2:
        raise ValueError(
            f"PCM16 frame data must have even length, got {len(frames)} bytes")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float64)
    return resample_linear_pcm(samples, recipe)


def resample_linear_pcm(samples: np.ndarray, recipe: ResampleRecipe
                        ) -> np.ndarray:
    """Decimate float64 samples (PCM16 amplitude scale) with the frozen
    recipe: centered polyphase FIR, signed-zero constant padding."""
    samples = np.asarray(samples, dtype=np.float64)
    resampled = resample_poly(
        samples,
        up=1,
        down=recipe.down,
        window=recipe.taps,
        padtype="constant",
        cval=0.0,
    )
    expected = (len(samples) + recipe.down - 1) // recipe.down
    if len(resampled) != expected:
        raise AssertionError(
            f"resampler returned {len(resampled)} frames, expected {expected}")
    return resampled


def linear_pcm_to_pcm8(samples: np.ndarray) -> bytes:
    """Quantize signed linear PCM amplitudes to conventional unsigned PCM8."""
    samples = np.asarray(samples, dtype=np.float64)
    if not np.isfinite(samples).all():
        raise ValueError("linear PCM samples must all be finite")
    clipped = np.clip(samples, -32768.0, 32767.0)
    return np.floor((clipped + 32768.0) / 256.0).astype(np.uint8).tobytes()


def _rate_label(sample_rate_hz: int) -> str:
    return f"{sample_rate_hz / 1000:g} kHz"


def read_pcm16_wav(wav_bytes: bytes, name: str, *, sample_rate_hz: int,
                   expected_frames: int | None = None) -> bytes:
    """Return the raw PCM16-LE frames of a mono, uncompressed WAV.

    Any deviation (channels, sample width, rate, compression, frame count,
    truncated data) raises ``ValueError`` naming ``name``.
    """
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as source:
            actual_format = (
                source.getnchannels(),
                source.getsampwidth(),
                source.getframerate(),
                source.getcomptype(),
            )
            expected_format = (1, 2, sample_rate_hz, "NONE")
            if actual_format != expected_format:
                raise ValueError(
                    f"{name}: WAV format {actual_format} != expected "
                    f"{expected_format} (mono, PCM16, "
                    f"{_rate_label(sample_rate_hz)}, uncompressed)")
            frame_count = source.getnframes()
            if expected_frames is not None and frame_count != expected_frames:
                raise ValueError(
                    f"{name}: {frame_count} frames != expected {expected_frames}")
            frames = source.readframes(frame_count)
    except wave.Error as exc:
        raise ValueError(f"{name}: invalid WAV file: {exc}") from exc

    if len(frames) != frame_count * 2:
        raise ValueError(
            f"{name}: read {len(frames)} PCM bytes for {frame_count} frames")
    return frames


def read_float32_wav(wav_bytes: bytes, name: str, *, sample_rate_hz: int
                     ) -> np.ndarray:
    """Return the samples of a mono IEEE-float32 WAV (format tag 3) as float64.

    The stdlib ``wave`` module only reads integer PCM, so the RIFF chunks are
    walked directly (``fmt ``, optional ``fact``/``LIST``, ``data``). Any
    deviation (format tag, channels, rate, bit depth, truncated chunk,
    non-finite sample) raises ``ValueError`` naming ``name``.
    """
    if (len(wav_bytes) < 12 or wav_bytes[:4] != b"RIFF"
            or wav_bytes[8:12] != b"WAVE"):
        raise ValueError(f"{name}: not a RIFF/WAVE file")
    fmt = None
    data = None
    pos = 12
    while pos + 8 <= len(wav_bytes):
        chunk_id = wav_bytes[pos:pos + 4]
        size = int.from_bytes(wav_bytes[pos + 4:pos + 8], "little")
        body = wav_bytes[pos + 8:pos + 8 + size]
        if len(body) != size:
            raise ValueError(f"{name}: truncated {chunk_id!r} chunk")
        if chunk_id == b"fmt ":
            if size < 16:
                raise ValueError(f"{name}: fmt chunk too short ({size} bytes)")
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif chunk_id == b"data" and data is None:
            data = body
        pos += 8 + size + (size & 1)
    if fmt is None or data is None:
        raise ValueError(f"{name}: missing fmt or data chunk")
    tag, channels, rate, _byte_rate, _block_align, bits = fmt
    actual_format = (tag, channels, rate, bits)
    expected_format = (3, 1, sample_rate_hz, 32)
    if actual_format != expected_format:
        raise ValueError(
            f"{name}: WAV format (tag, channels, rate, bits) {actual_format} "
            f"!= expected {expected_format} (mono, IEEE float32, "
            f"{_rate_label(sample_rate_hz)}, uncompressed)")
    if len(data) % 4:
        raise ValueError(
            f"{name}: float32 data chunk has {len(data)} bytes, not a "
            "multiple of 4")
    samples = np.frombuffer(data, dtype="<f4").astype(np.float64)
    if not np.isfinite(samples).all():
        raise ValueError(f"{name}: float32 samples must all be finite")
    return samples


FLOAT_TO_PCM16_SCALE = 32768.0


def pcm8_from_float_samples(samples: np.ndarray, name: str, stem: str,
                            resample_down: int = 1) -> bytes:
    """Transform unit-scale float samples to the stem's PCM8 stream.

    ``samples * 32768`` puts them on the PCM16 amplitude scale; the frozen FIR
    decimation (if any) and the shared quantizer
    ``floor((clip(x, -32768, 32767) + 32768) / 256)`` then follow exactly as
    for the resampled PCM16 stems, so a full-scale float file lands on the
    same PCM8 values as its PCM16 rendering would.
    """
    scaled = np.asarray(samples, dtype=np.float64) * FLOAT_TO_PCM16_SCALE
    if resample_down == 1:
        pcm8 = linear_pcm_to_pcm8(scaled)
    else:
        recipe = RESAMPLE_RECIPES.get(resample_down)
        if recipe is None:
            raise AssertionError(
                f"{stem}: unsupported resample factor {resample_down}")
        pcm8 = linear_pcm_to_pcm8(resample_linear_pcm(scaled, recipe))
    expected_output_frames = (
        len(scaled) + resample_down - 1) // resample_down
    if len(pcm8) != expected_output_frames:
        raise ValueError(
            f"{name}: {len(pcm8)} transformed frames != expected "
            f"{expected_output_frames} for {stem}")
    return pcm8


def pcm8_from_pcm16_frames(frames: bytes, name: str, stem: str,
                           resample_down: int = 1) -> bytes:
    """Transform PCM16-LE frames to the stem's PCM8 stream (direct high-byte
    map for ``resample_down == 1``, else the frozen FIR decimation followed by
    the float quantizer) and check the output length."""
    frame_count = len(frames) // 2
    if resample_down == 1:
        pcm8 = pcm16le_to_pcm8(frames)
    else:
        recipe = RESAMPLE_RECIPES.get(resample_down)
        if recipe is None:
            raise AssertionError(
                f"{stem}: unsupported resample factor "
                f"{resample_down}")
        pcm8 = linear_pcm_to_pcm8(resample_pcm16le(frames, recipe))

    expected_output_frames = (
        frame_count + resample_down - 1) // resample_down
    if len(pcm8) != expected_output_frames:
        raise ValueError(
            f"{name}: {len(pcm8)} transformed frames != expected "
            f"{expected_output_frames} for {stem}")
    return pcm8


def normalized_member_name(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive member path: {name!r}")
    normalized = "/".join(part for part in path.parts if part not in ("", "."))
    if not normalized:
        raise ValueError(f"empty archive member path: {name!r}")
    return normalized


def stable_digest(name: str) -> bytes:
    return hashlib.sha256(name.encode("utf-8")).digest()


def clip_sort_key(item: tuple) -> tuple[bytes, str]:
    """Deterministic clip order: ``(sha256(name), name)``."""
    return (stable_digest(item[0]), item[0])


@dataclass(frozen=True)
class ClipPlan:
    name: str
    pcm8: bytes
    num_windows: int
    start_chunk: int
    # ``spread`` mode only: the full per-clip chunk visiting order (a
    # permutation whose first entries are evenly spaced over the clip).
    # ``None`` means the plain ``(start_chunk + round) % num_windows`` walk.
    order: tuple[int, ...] | None = None

    def chunk_index(self, round_index: int) -> int:
        if self.order is not None:
            return self.order[round_index]
        return (self.start_chunk + round_index) % self.num_windows


@dataclass(frozen=True)
class WindowPlan:
    plans: list[ClipPlan]
    total_windows: int
    total_tail_samples: int
    clips_without_window: int
    selected_windows: int


def round_robin_counts(num_windows: list[int], selected: int) -> list[int]:
    """Exact number of windows :func:`iter_capped_windows` draws from each
    clip (in plan order) when emitting ``selected`` windows in total.

    The emitter serves every clip once per round and skips clips that are
    exhausted, so clip ``i`` receives ``min(n_i, full)`` windows over the
    ``full`` complete rounds, plus one more if it is among the first clips
    still unexhausted in the partial last round.
    """
    assert 0 <= selected <= sum(num_windows)

    def served(rounds: int) -> int:
        return sum(min(n, rounds) for n in num_windows)

    full = bisect.bisect_right(
        range(max(num_windows, default=0) + 1), selected, key=served) - 1
    counts = [min(n, full) for n in num_windows]
    remainder = selected - served(full)
    for index, n in enumerate(num_windows):
        if not remainder:
            break
        if n > full:
            counts[index] += 1
            remainder -= 1
    assert sum(counts) == selected
    return counts


def _spread_order(num_windows: int, start_chunk: int, quota: int
                  ) -> tuple[int, ...]:
    """Permutation of ``range(num_windows)`` whose first ``quota`` entries are
    evenly spaced over the whole clip from ``start_chunk``: pick ``k`` sits at
    offset ``k * num_windows // quota`` (a fractional stride, so consecutive
    picks are ``floor`` or ``ceil`` of ``num_windows / quota`` apart, the wrap
    back to ``start_chunk`` included). The remaining chunks follow in plain
    order from ``start_chunk``. ``quota`` is clamped to ``[1, num_windows]``."""
    quota = min(max(quota, 1), num_windows)
    order = [(start_chunk + (k * num_windows) // quota) % num_windows
             for k in range(quota)]
    picked = set(order)
    order.extend(
        chunk for chunk in ((start_chunk + offset) % num_windows
                            for offset in range(num_windows))
        if chunk not in picked)
    assert len(order) == num_windows and len(set(order)) == num_windows
    return tuple(order)


def plan_clip_windows(clips: list[tuple[str, bytes]], window: int,
                      num_sequences: int, *, label: str,
                      spread: bool = False) -> WindowPlan:
    """Split every clip into exact windows and size the deterministic cap.

    ``clips`` must already be sorted with :func:`clip_sort_key`. Raises
    ``SystemExit`` (as the bake CLIs do) when no clip holds a full window.
    ``spread=True`` gives each clip exactly the share the round-robin emitter
    will draw from it and spaces those picks evenly over the clip's whole
    duration, so a capped bake samples long recordings end-to-end;
    ``spread=False`` is the original consecutive walk from the hashed start
    chunk.
    """
    plans = []
    total_windows = 0
    total_tail_samples = 0
    clips_without_window = 0
    for name, pcm8 in clips:
        num_windows, tail_samples = divmod(len(pcm8), window)
        total_windows += num_windows
        total_tail_samples += tail_samples
        if not num_windows:
            clips_without_window += 1
            continue
        start_chunk = int.from_bytes(stable_digest(name)[:8], "big") % num_windows
        plans.append(ClipPlan(name, pcm8, num_windows, start_chunk))

    if not total_windows:
        raise SystemExit(
            f"{label}: no recording has a full {window}-byte eval window")
    selected_windows = (
        total_windows if num_sequences == 0
        else min(num_sequences, total_windows)
    )
    if spread:
        counts = round_robin_counts(
            [plan.num_windows for plan in plans], selected_windows)
        plans = [
            ClipPlan(plan.name, plan.pcm8, plan.num_windows, plan.start_chunk,
                     _spread_order(plan.num_windows, plan.start_chunk, count))
            for plan, count in zip(plans, counts)
        ]
    print(
        f"{label}: {len(clips)} clips -> {total_windows} full {window}-byte "
        f"windows; {total_tail_samples} per-clip tail samples unusable; "
        f"{clips_without_window} clips shorter than one window")
    print(
        f"{label}: selecting {selected_windows} windows "
        f"({'all' if selected_windows == total_windows else 'deterministic cap'})")
    return WindowPlan(plans, total_windows, total_tail_samples,
                      clips_without_window, selected_windows)


def iter_capped_windows(plan: WindowPlan, window: int
                        ) -> Iterator[tuple[ClipPlan, int, bytes]]:
    """Round-robin across clips, yielding ``(clip, chunk_index, window_bytes)``
    until ``plan.selected_windows`` windows have been emitted."""
    index = 0
    round_index = 0
    while index < plan.selected_windows:
        for clip in plan.plans:
            if round_index >= clip.num_windows:
                continue
            chunk_index = clip.chunk_index(round_index)
            start = chunk_index * window
            yield clip, chunk_index, clip.pcm8[start:start + window]
            index += 1
            if index == plan.selected_windows:
                return
        round_index += 1
