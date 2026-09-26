"""Dataset-neutral helpers for rendering benchmark melody bytes as MIDI.

The byte contract uses MIDI pitches 0--127 for note attacks, 128 for a
sixteenth-note hold, 129 for an authored rest, and 130 for a silent stream
boundary marker.  JSONL row boundaries have no musical meaning: callers load
and concatenate every ``sequence`` array before decoding it.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Sequence

import mido


HOLD = 128
REST = 129
END_MARKER = 130
MIDI_TYPE = 0
TICKS_PER_QUARTER = 480
STEPS_PER_QUARTER = 4
TICKS_PER_SIXTEENTH = TICKS_PER_QUARTER // STEPS_PER_QUARTER
MIDI_CHANNEL = 0
DEFAULT_TEMPO_BPM = 120.0
DEFAULT_PROGRAM = 0
DEFAULT_VELOCITY = 80
MAX_MIDI_TEMPO = 0xFFFFFF


class InputValidationError(ValueError):
    """The JSONL payload cannot be decoded under the melody-byte contract."""


@dataclass(frozen=True)
class LoadedPayload:
    """Validated concatenated JSONL sequences and source provenance."""

    payload: bytes
    sequence_count: int
    source_jsonl_sha256: str


@dataclass(frozen=True)
class RenderStats:
    """Statistics derived solely from a validated melody payload."""

    cell_count: int
    note_attack_count: int
    hold_count: int
    rest_count: int
    end_marker_count: int
    pitch_min: int | None
    pitch_max: int | None
    duration_ticks: int


def manifest_path_for(output_path: Path) -> Path:
    """Return the adjacent manifest path for a MIDI output path."""
    return Path(output_path).with_suffix(".manifest.json")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_jsonl_payload(input_path: Path) -> LoadedPayload:
    """Load and concatenate validated ``sequence`` arrays in JSONL order."""
    input_path = Path(input_path)
    raw = input_path.read_bytes()
    source_sha256 = _sha256(raw)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InputValidationError(
            f"{input_path}: input is not valid UTF-8: {exc}"
        ) from exc

    payload = bytearray()
    sequence_count = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise InputValidationError(
                f"{input_path}:{line_number}: blank lines are not valid JSONL rows"
            )
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputValidationError(
                f"{input_path}:{line_number}: malformed JSON: {exc.msg}"
            ) from exc
        if not isinstance(row, dict):
            raise InputValidationError(
                f"{input_path}:{line_number}: row must be a JSON object"
            )
        if "sequence" not in row:
            raise InputValidationError(
                f"{input_path}:{line_number}: missing required 'sequence' field"
            )
        sequence = row["sequence"]
        if not isinstance(sequence, list):
            raise InputValidationError(
                f"{input_path}:{line_number}: 'sequence' must be a JSON array"
            )
        for column, value in enumerate(sequence):
            if isinstance(value, bool) or not isinstance(value, int):
                raise InputValidationError(
                    f"{input_path}:{line_number}: sequence[{column}] must be an "
                    f"integer, got {value!r}"
                )
            if not 0 <= value <= END_MARKER:
                raise InputValidationError(
                    f"{input_path}:{line_number}: sequence[{column}] is outside "
                    f"the allowed range 0..{END_MARKER}: {value}"
                )
            payload.append(value)
        sequence_count += 1

    if sequence_count == 0:
        raise InputValidationError(f"{input_path}: input contains no JSONL rows")
    if not payload:
        raise InputValidationError(
            f"{input_path}: concatenated sequence is empty"
        )
    return LoadedPayload(bytes(payload), sequence_count, source_sha256)


def tempo_microseconds(tempo_bpm: float) -> int:
    if isinstance(tempo_bpm, bool) or not isinstance(tempo_bpm, (int, float)):
        raise ValueError("tempo_bpm must be a number")
    tempo_bpm = float(tempo_bpm)
    if not math.isfinite(tempo_bpm) or tempo_bpm <= 0:
        raise ValueError("tempo_bpm must be finite and greater than zero")
    microseconds = int(round(60_000_000 / tempo_bpm))
    if not 1 <= microseconds <= MAX_MIDI_TEMPO:
        raise ValueError(
            "tempo_bpm is outside the range representable by a MIDI tempo event"
        )
    return microseconds


def _validate_midi_options(
    tempo_bpm: float, program: int, velocity: int
) -> int:
    tempo_microseconds_value = tempo_microseconds(tempo_bpm)
    if (
        isinstance(program, bool)
        or not isinstance(program, int)
        or not 0 <= program <= 127
    ):
        raise ValueError("program must be an integer in 0..127")
    if (
        isinstance(velocity, bool)
        or not isinstance(velocity, int)
        or not 1 <= velocity <= 127
    ):
        raise ValueError("velocity must be an integer in 1..127")
    return tempo_microseconds_value


def build_midi(
    payload: bytes | bytearray | Sequence[int],
    *,
    tempo_bpm: float = DEFAULT_TEMPO_BPM,
    program: int = DEFAULT_PROGRAM,
    velocity: int = DEFAULT_VELOCITY,
) -> tuple[bytes, RenderStats]:
    """Decode a melody payload into an in-memory type-0 MIDI file."""
    tempo_microseconds = _validate_midi_options(tempo_bpm, program, velocity)
    if not payload:
        raise InputValidationError("melody payload is empty")

    midi = mido.MidiFile(type=MIDI_TYPE, ticks_per_beat=TICKS_PER_QUARTER)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo_microseconds, time=0))
    track.append(
        mido.Message(
            "program_change", channel=MIDI_CHANNEL, program=program, time=0
        )
    )

    active_pitch: int | None = None
    last_event_tick = 0
    note_attack_count = 0
    hold_count = 0
    rest_count = 0
    end_marker_count = 0
    pitch_min: int | None = None
    pitch_max: int | None = None

    for cell_index, value in enumerate(payload):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= END_MARKER
        ):
            raise InputValidationError(
                f"payload cell {cell_index} must be an integer in "
                f"0..{END_MARKER}, got {value!r}"
            )
        tick = cell_index * TICKS_PER_SIXTEENTH
        if value <= 127:
            if active_pitch is not None:
                track.append(
                    mido.Message(
                        "note_off",
                        channel=MIDI_CHANNEL,
                        note=active_pitch,
                        velocity=0,
                        time=tick - last_event_tick,
                    )
                )
                last_event_tick = tick
            track.append(
                mido.Message(
                    "note_on",
                    channel=MIDI_CHANNEL,
                    note=value,
                    velocity=velocity,
                    time=tick - last_event_tick,
                )
            )
            last_event_tick = tick
            active_pitch = value
            note_attack_count += 1
            pitch_min = value if pitch_min is None else min(pitch_min, value)
            pitch_max = value if pitch_max is None else max(pitch_max, value)
        elif value == HOLD:
            if active_pitch is None:
                raise InputValidationError(
                    f"payload cell {cell_index} is a hold ({HOLD}) without an "
                    "active recoverable pitch"
                )
            hold_count += 1
        else:
            if value == REST:
                rest_count += 1
            else:
                assert value == END_MARKER
                end_marker_count += 1
            if active_pitch is not None:
                track.append(
                    mido.Message(
                        "note_off",
                        channel=MIDI_CHANNEL,
                        note=active_pitch,
                        velocity=0,
                        time=tick - last_event_tick,
                    )
                )
                last_event_tick = tick
                active_pitch = None

    duration_ticks = len(payload) * TICKS_PER_SIXTEENTH
    if active_pitch is not None:
        track.append(
            mido.Message(
                "note_off",
                channel=MIDI_CHANNEL,
                note=active_pitch,
                velocity=0,
                time=duration_ticks - last_event_tick,
            )
        )
        last_event_tick = duration_ticks

    # Keep trailing rests and markers in the represented stream duration.
    track.append(
        mido.MetaMessage(
            "end_of_track", time=duration_ticks - last_event_tick
        )
    )

    output = io.BytesIO()
    midi.save(file=output)
    stats = RenderStats(
        cell_count=len(payload),
        note_attack_count=note_attack_count,
        hold_count=hold_count,
        rest_count=rest_count,
        end_marker_count=end_marker_count,
        pitch_min=pitch_min,
        pitch_max=pitch_max,
        duration_ticks=duration_ticks,
    )
    return output.getvalue(), stats


def _normalized_path(path: Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def preflight_destinations(
    destinations: Sequence[Path],
    *,
    protected_paths: Sequence[Path],
    overwrite: bool,
) -> None:
    """Validate output paths before any data is staged."""
    protected = {_normalized_path(path) for path in protected_paths}
    for destination in destinations:
        destination = Path(destination)
        if _normalized_path(destination) in protected:
            raise ValueError(f"refusing to overwrite input file: {destination}")
        exists = destination.exists() or destination.is_symlink()
        if exists and not overwrite:
            raise FileExistsError(
                f"{destination} already exists; pass --overwrite to replace it"
            )
        if exists and (
            destination.is_symlink() or not destination.is_file()
        ):
            raise ValueError(
                f"refusing to overwrite non-regular output: {destination}"
            )


def _stage_bytes(path: Path, data: bytes) -> Path:
    """Write and fsync a temporary file in ``path``'s destination directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(data)
            destination.flush()
            os.fsync(destination.fileno())
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _backup_file(path: Path) -> Path:
    """Hard-link a regular output file to a private same-directory backup."""
    descriptor, backup_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".bak"
    )
    os.close(descriptor)
    backup = Path(backup_name)
    backup.unlink()
    try:
        os.link(path, backup)
    except BaseException:
        backup.unlink(missing_ok=True)
        raise
    return backup


def _commit_staged_outputs(
    staged: Sequence[tuple[Path, Path]],
    *,
    overwrite: bool,
) -> None:
    """Atomically commit a group of staged files, rolling back on failure."""
    if not overwrite:
        linked: list[tuple[Path, Path]] = []
        try:
            for temporary, destination in staged:
                os.link(temporary, destination)
                linked.append((temporary, destination))
        except BaseException:
            for temporary, destination in reversed(linked):
                try:
                    if temporary.samefile(destination):
                        destination.unlink()
                except (FileNotFoundError, OSError):
                    pass
            raise
        else:
            for temporary, _ in staged:
                temporary.unlink()
        return

    backups: dict[Path, Path] = {}
    committed: list[Path] = []
    try:
        for _, destination in staged:
            if destination.exists():
                backups[destination] = _backup_file(destination)

        for temporary, destination in staged:
            os.replace(temporary, destination)
            committed.append(destination)
    except BaseException:
        for destination in reversed(committed):
            backup = backups.pop(destination, None)
            if backup is None:
                destination.unlink(missing_ok=True)
            else:
                os.replace(backup, destination)
        raise
    finally:
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def write_outputs(
    outputs: Sequence[tuple[Path, bytes]], *, overwrite: bool
) -> None:
    """Stage and transactionally commit one or more output files."""
    staged: list[tuple[Path, Path]] = []
    try:
        for destination, contents in outputs:
            staged.append((_stage_bytes(destination, contents), destination))
        _commit_staged_outputs(staged, overwrite=overwrite)
        staged.clear()
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
