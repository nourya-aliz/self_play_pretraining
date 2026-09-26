"""Bake a curated Mutopia corpus as monophonic sixteenth-note bytes.

The benchmark reads one explicitly selected MIDI track from each pinned
Public Domain Mutopia contribution.  It discards all MIDI metadata, quantizes
note spans to a sixteenth-note grid, and takes the highest active note in the
selected track.  The byte contract is::

    0..127  selected source-note onset (MIDI pitch)
    128     continue the same selected source note
    129     rest
    130     end of piece

Usage (from the project root):
    python -m src.scripts.prepare_mutopia_benchmark --context-length 256
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import mido
import requests

from src.scripts.bake_utils import (
    RAW_SOURCES_DIR,
    bake_output_path,
    chunk_bytes,
    make_record,
    window_from_context,
    write_bake,
)


STEM = "mutopia_melody_16th"
SOURCE = "Mutopia Project curated Public Domain MIDI contributions"
SOURCE_LICENSE = "Public Domain"
SOURCE_REPOSITORY = "https://github.com/MutopiaProject/MutopiaProject"
SOURCE_REPOSITORY_COMMIT = "2144afd6f52d56c5b6995b8b589ef1268b3139f0"
STREAM_CACHE = RAW_SOURCES_DIR / f"{STEM}.raw"

STEPS_PER_QUARTER = 4
HOLD = 128
REST = 129
END_PIECE = 130
MAX_ITEM_BYTES = 4_096
MAX_MELODY_CELLS = MAX_ITEM_BYTES - 1

EXPECTED_PIECES = 40
STREAM_SIZE = 72_646
STREAM_SHA256 = "7054fc92c6fc1bdfdfb378e75ff02c0a73bf5045b541ff7638d14d180517a12b"
SOURCE_MANIFEST_SHA256 = (
    "e1cd086e62d5dccb411ce38e41cb80be5179df0ed8fa26a35b404988fe54ede4"
)
CATALOG_SHA256 = (
    "d17fc6516edf8dcccd8cc859fe8249a208447c24ac56b73119830ffb46b2c2b4"
)
EXPECTED_MATCHED_NOTES = 42_041
EXPECTED_COLLAPSED_NOTES = 2_412
EXPECTED_UNMATCHED_NOTE_OFFS = 2
EXPECTED_TRUNCATED_PIECES = 5
EXPECTED_NOTE_ON_TOKENS = 27_416
EXPECTED_HOLD_TOKENS = 35_275
EXPECTED_REST_TOKENS = 9_915
EXPECTED_END_PIECE_TOKENS = 40
EXPECTED_MELODY_TOKENS = 72_606

CATALOG_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / f"{STEM}_catalog.tsv"
)


class SourceValidationError(ValueError):
    """A source asset or deterministic transform violated the contract."""


@dataclass(frozen=True)
class Note:
    pitch: int
    start_tick: int
    end_tick: int
    event_order: int = 0


@dataclass(frozen=True)
class PieceSpec:
    item_id: str
    mutopia_id: int
    composer: str
    title: str
    selected_part: str
    source_url: str
    source_size: int
    source_sha256: str
    track_index: int
    zip_member: str | None = None
    member_size: int | None = None
    member_sha256: str | None = None
    expected_original_cells: int | None = None
    expected_melody_cells: int | None = None
    expected_collapsed_notes: int | None = None
    expected_unmatched_note_offs: int | None = None
    expected_truncated: bool | None = None

    @property
    def piece_url(self) -> str:
        return (
            "https://www.mutopiaproject.org/cgibin/"
            f"piece-info.cgi?id={self.mutopia_id}"
        )


@dataclass(frozen=True)
class EncodingStats:
    original_cells: int
    retained_cells: int
    collapsed_notes: int
    truncated: bool
    note_on_tokens: int
    hold_tokens: int
    rest_tokens: int


@dataclass(frozen=True)
class PieceResult:
    spec: PieceSpec
    stream_offset: int
    melody: bytes
    end_marker_offset: int
    original_cells: int
    collapsed_notes: int
    unmatched_note_offs: int
    truncated: bool
    matched_notes: int
    note_on_tokens: int
    hold_tokens: int
    rest_tokens: int

    @property
    def item_bytes(self) -> int:
        return len(self.melody) + 1

    @property
    def melody_sha256(self) -> str:
        return hashlib.sha256(self.melody).hexdigest()


@dataclass(frozen=True)
class BuildStats:
    pieces: int
    matched_notes: int
    collapsed_notes: int
    unmatched_note_offs: int
    truncated_pieces: int
    note_on_tokens: int
    hold_tokens: int
    rest_tokens: int
    end_piece_tokens: int
    melody_tokens: int
    total_tokens: int


_FTP = "https://www.mutopiaproject.org/ftp/"


def _piece(
    item_id: str,
    mutopia_id: int,
    composer: str,
    title: str,
    selected_part: str,
    source_path: str,
    source_size: int,
    source_sha256: str,
    track_index: int,
    original_cells: int,
    melody_cells: int,
    collapsed_notes: int,
    *,
    zip_member: str | None = None,
    member_size: int | None = None,
    member_sha256: str | None = None,
    unmatched_note_offs: int = 0,
) -> PieceSpec:
    return PieceSpec(
        item_id=item_id,
        mutopia_id=mutopia_id,
        composer=composer,
        title=title,
        selected_part=selected_part,
        source_url=_FTP + source_path,
        source_size=source_size,
        source_sha256=source_sha256,
        track_index=track_index,
        zip_member=zip_member,
        member_size=member_size,
        member_sha256=member_sha256,
        expected_original_cells=original_cells,
        expected_melody_cells=melody_cells,
        expected_collapsed_notes=collapsed_notes,
        expected_unmatched_note_offs=unmatched_note_offs,
        expected_truncated=original_cells > MAX_MELODY_CELLS,
    )


# Ordering is part of the benchmark contract.  Every URL is the download on
# the corresponding Mutopia contribution page, not a third-party mirror.
PIECES: tuple[PieceSpec, ...] = (
    _piece("941", 941, "Ludwig van Beethoven", "Symphony No. 5, I",
           "Violin I", "BeethovenLv/O67/Symphony5_1/Symphony5_1.mid",
           74649, "0ce1bca911ba8e6234fd2e474a9b6da4c968c5cfe85b0cccfe7e56505df52046",
           8, 4052, 4052, 0),
    _piece("108", 108, "Wolfgang Amadeus Mozart", "Rondo alla Turca",
           "upper piano", "MozartWA/KV331/KV331_3_RondoAllaTurca/KV331_3_RondoAllaTurca.mid",
           13772, "c14299cbdd34c8b2081da84dc06519ef8a52a0d589ee53c758d28b8e3d727d1c",
           1, 1020, 1020, 39),
    _piece("931", 931, "Ludwig van Beethoven", "Fur Elise",
           "upper piano", "BeethovenLv/WoO59/fur_Elise_WoO59/fur_Elise_WoO59.mid",
           7590, "1c12c21c7bbf4cf163896732672648a69d497636059837abd153c71abe50215a",
           1, 626, 626, 53),
    _piece("900", 900, "Wolfgang Amadeus Mozart", "Eine kleine Nachtmusik, I",
           "Violin I", "MozartWA/KV525/eine-kleine-nachtmusik-mvt1/eine-kleine-nachtmusik-mvt1.mid",
           49426, "f8b6dc991df6e90e2e2e11d561ab2cf8fec47beda747c69984c3fef4b539506b",
           1, 3068, 3068, 91),
    _piece("1780", 1780, "Johann Sebastian Bach", "Toccata and Fugue in D minor, BWV 565",
           "upper organ", "BachJS/BWV565/ToccataFugue/ToccataFugue.mid",
           30267, "1aabd00967aded08d6633482c211af46f679d97f01cfeaca43c83a59dfeeeb6b",
           1, 2288, 2288, 237),
    _piece("1778", 1778, "Claude Debussy", "Clair de lune",
           "upper piano", "DebussyC/L75/debussy_Ste_Bergamesq_Clair/debussy_Ste_Bergamesq_Clair.mid",
           12476, "4eee9a1546ffde1ff74cb9824ba0e57cbc821185a15185bfea9faed97820bf8c",
           1, 1290, 1290, 0),
    _piece("1693", 1693, "Frederic Chopin", "Fantaisie-Impromptu",
           "upper piano", "ChopinFF/O66/chopin_fantaisie-impromptu/chopin_fantaisie-impromptu.mid",
           30982, "b29f6add7473b6e931c2e66349e860bf0e884cf68e19e90420a4870aae689f96",
           1, 2208, 2208, 21),
    _piece("1888", 1888, "Edvard Grieg", "In the Hall of the Mountain King",
           "solo/upper", "GriegE/O46/Dans_l_antre_du_roi_de_la_montagne/Dans_l_antre_du_roi_de_la_montagne.mid",
           15156, "0c256a809b5af1faef81b6aad6106088a71d76b12435d5f0775d4ceee60c917e",
           1, 1400, 1400, 30),
    _piece("263", 263, "Scott Joplin", "The Entertainer",
           "upper piano", "JoplinS/entertainer/entertainer.mid",
           22084, "33e4e81ee64ffb2edf90d1c6a1ddee7276507296bfb915c2bb775231a467f066",
           1, 1214, 1214, 0),
    _piece("295", 295, "Ludwig van Beethoven", "Pathetique Sonata, II",
           "upper piano", "BeethovenLv/O13/pathetique-2/pathetique-2.mid",
           13382, "92c685ca98001473bcc21fa98110cc125ab489c1bf49be185e062d204a268492",
           1, 584, 584, 221),
    _piece("517", 517, "Johann Sebastian Bach", "Cello Suite No. 1: Prelude",
           "cello", "BachJS/BWV1007/bwv1007/bwv1007-mids.zip",
           39650, "57fccefc88f315d90e479ce400a6647fc86fbc2b5cbda64f5f29e6c58ea143ce",
           1, 512, 512, 2, zip_member="bwv1007-1.mid",
           member_size=4034,
           member_sha256="59449437fd1455537d3fd9d8963eac4a9f2d413f0d446b39539edbc7090fe619"),
    _piece("266", 266, "Wolfgang Amadeus Mozart", "Queen of the Night aria",
           "solo voice", "MozartWA/KV620/magicflute-14-aria/magicflute-14-aria-mids.zip",
           19951, "f27fc1dc545f7862f5cb03907a661c9faea585d327856c3e915053f84cba1ae9",
           1, 1528, 1528, 6, zip_member="konigindernacht.mid",
           member_size=4883,
           member_sha256="345a709eb0eef33e4a87f00c26e2dffdbca412b4bf6b4633ba56b96763b24679"),
    _piece("459", 459, "George Frideric Handel", "Hallelujah Chorus",
           "soprano", "HandelGF/hallelujah/hallelujah-mids.zip",
           19280, "74bf4f72e3c534d3cea991885106a829c0bb38e362f13432e24b85a7981277d3",
           1, 1504, 1504, 0, zip_member="chorale-score.mid",
           member_size=35611,
           member_sha256="7cc9f2623f3a306f217c56f373b4e4330d040462484308136d20690b000fdaf5"),
    _piece("483", 483, "Frederic Chopin", "Minute Waltz",
           "upper piano", "ChopinFF/O64/chopin_valse_op64_no1/chopin_valse_op64_no1.mid",
           13086, "62333ad9f9b22603d009734ab500330a12ac09aeca13d185ac0ba64a7fb9d206",
           1, 1676, 1676, 10),
    _piece("37", 37, "Erik Satie", "Gymnopedie No. 1",
           "upper piano", "SatieE/gymnopedie_1/gymnopedie_1.mid",
           2892, "09ce7337b4bf42ba12e1aa64b4c7dc7fcfa6455f1070cb12f8880a755df97384",
           1, 564, 564, 0, unmatched_note_offs=2),
    _piece("595", 595, "Ludwig van Beethoven", "Symphony No. 7, II",
           "viola", "BeethovenLv/O92/Symphony7_2/Symphony7_2.mid",
           60426, "7cf810b4625b9a272d3a0a9153d8beb5750fa3ee64122dc9f27b7661e997b903",
           10, 2204, 2204, 0),
    _piece("1054", 1054, "Franz Schubert", "Ave Maria",
           "vocal line", "SchubertF/D839/SchubertF-D839_AveMaria/SchubertF-D839_AveMaria.mid",
           22997, "289d1d2c8f5a16f1b1732bc1a60652e662f24e09531ebb0d8017b49712724684",
           1, 632, 632, 75),
    _piece("471", 471, "Frederic Chopin", "Raindrop Prelude",
           "upper piano", "ChopinFF/O28/Chop-28-15/Chop-28-15.mid",
           14195, "bb9532e59bee378a42f0c53b64bf223ba4d6e261e00c1e2efb3ed8e023833bd9",
           1, 1424, 1424, 19),
    _piece("23", 23, "Scott Joplin", "Maple Leaf Rag",
           "upper piano", "JoplinS/maple/maple.mid",
           21518, "3dd712a85fabd267f5a2cee5cb23af4683408c2f29b8814721844498f1ee4f66",
           1, 1152, 1152, 0),
    _piece("937", 937, "Ludwig van Beethoven", "Appassionata Sonata, III",
           "upper piano", "BeethovenLv/O57/LVB_Sonate_57_3/LVB_Sonate_57_3.mid",
           61865, "cba2978986da83870f6d4a84406a9ed923adebf865f19931d629fde204c388d8",
           1, 4546, 4095, 3),
    _piece("299", 299, "Ludwig van Beethoven", "Pathetique Sonata, I",
           "upper piano", "BeethovenLv/O13/pathetique-1/pathetique-1.mid",
           37697, "cfe2b012227077a304a0ed282a98dceaca6a73bd0fb4d3f2249ac4bacbe30121",
           1, 4980, 4095, 161),
    _piece("296", 296, "Ludwig van Beethoven", "Pathetique Sonata, III",
           "upper piano", "BeethovenLv/O13/pathetique-3/pathetique-3.mid",
           21362, "e0e27f158b9eac0747569f0e4096c96c54a5eba425c567b368523afef2190797",
           1, 3354, 3354, 35),
    _piece("1380", 1380, "Wolfgang Amadeus Mozart", "Symphony No. 25, I",
           "Violin I", "MozartWA/KV183/Symphony25_1/Symphony25_1.mid",
           86979, "36bcc5b01a15e5c478ea1298ea93fe92b6bba6106ceeea29a2c3ab6921bca092",
           4, 6612, 4095, 72),
    _piece("1777", 1777, "Claude Debussy", "Premiere Arabesque",
           "upper piano", "DebussyC/L66/debussy_Arabesque_1/debussy_Arabesque_1.mid",
           13154, "6732dc10799ec0af47d19d9095562d0671864a750129e0279901def169a34757",
           1, 1692, 1692, 0),
    _piece("1030", 1030, "Franz Schubert", "Erlkonig",
           "vocal line", "SchubertF/D328/Erlkoenig/Erlkoenig.mid",
           51169, "e433eca07d4d1ca85622bfc0fafeb6f74ac84710f1eebe57a1e1199876fc6603",
           1, 2348, 2348, 3),
    _piece("466-m1", 466, "Pyotr Ilyich Tchaikovsky", "Violin Concerto, I",
           "solo violin", "TchaikovskyPI/O35/tchai_op35/tchai_op35-mids.zip",
           339616, "cb450f2ba9ee28fb849b4810260ae847c888e33e5d264448c19fa4e95f04d9d5",
           10, 6144, 4095, 1067, zip_member="tchai_op35.mid",
           member_size=190806,
           member_sha256="752c04565cf1287ff3a4983ee9e561fdb7cd26c5fa78994a02f795deb516a88e"),
    _piece("466-m3", 466, "Pyotr Ilyich Tchaikovsky", "Violin Concerto, III",
           "solo violin", "TchaikovskyPI/O35/tchai_op35/tchai_op35-mids.zip",
           339616, "cb450f2ba9ee28fb849b4810260ae847c888e33e5d264448c19fa4e95f04d9d5",
           10, 5106, 4095, 78, zip_member="tchai_op35-2.mid",
           member_size=129125,
           member_sha256="22c6c11ccade18d8121e6487cde20d18a84be3c8ca3cab4bb2c7c4123421ca8d"),
    _piece("2034", 2034, "Wolfgang Amadeus Mozart", "Fantasia in D minor, K.397",
           "upper piano", "MozartWA/KV397/Fantasia/Fantasia.mid",
           13485, "22a84b967a00f9d7cf6bec68f6d4e9c0578c1275ad0eb7fc8e30979d375e58cb",
           1, 1457, 1457, 72),
    _piece("504", 504, "Robert Schumann", "Traumerei",
           "upper piano", "SchumannR/O15/SchumannOp15No07/SchumannOp15No07.mid",
           3708, "6e0045d5783ff28dedcf05813074fe407ca3846e5e2a7246f2793ac5614e4851",
           1, 388, 388, 0),
    _piece("468", 468, "Frederic Chopin", "Prelude Op.28 No.4",
           "upper piano", "ChopinFF/O28/Chop-28-4/Chop-28-4.mid",
           5444, "c1d7e05aa4e6e2baeec2e9108402a969c16d5df302b82fd66d110634697f2e3b",
           1, 404, 404, 2),
    _piece("626", 626, "John Philip Sousa", "Stars and Stripes Forever",
           "upper part", "SousaJP/TheStarsAndStripesForever/TheStarsAndStripesForever.mid",
           22170, "73e75e76c0708dbec4697ab61438b5c63ca3b5c1a56de59f1f5ab0478fd3b274",
           1, 3388, 3388, 2),
    _piece("1215", 1215, "Giuseppe Verdi", "La Traviata: Brindisi",
           "upper part", "VerdiG/Traviata_02/Traviata_02.mid",
           17810, "b6d7d5243e9543e7d43cfb427ed7f2fb567165b7d4037734295cf44cb12bc4b2",
           1, 1154, 1154, 66),
    _piece("625", 625, "John Philip Sousa", "Liberty Bell March",
           "upper part", "SousaJP/TheLibertyBell/TheLibertyBell.mid",
           26638, "55013a43c5ce0335a3da903750686ad81c8e402d90a142269977a7e5987d299b",
           1, 2542, 2542, 0),
    _piece("1193", 1193, "Franz Schubert", "Impromptu D.899 No.3",
           "upper piano", "SchubertF/D899/SchubertF-D899-3-Impromptu/SchubertF-D899-3-Impromptu.mid",
           40676, "3ae00ab41634cd65c495a0aec41798ff191bd2690f48ec93daa53dd0266edaa2",
           1, 2784, 2784, 0),
    _piece("610", 610, "Johann Strauss II", "Thunder and Lightning",
           "upper part", "StraussJJ/O324/blitz/blitz.mid",
           21227, "7c48c02311de8d2a41c2b7fca818a16256c1d7403de0897ff993650fd3484abb",
           1, 1544, 1544, 0),
    _piece("1864", 1864, "George Frideric Handel", "Royal Fireworks: La Rejouissance",
           "trumpet", "HandelGF/BWV351/LaRej/LaRej.mid",
           5063, "17bd0a0d48f93c04f5dd5e13c842ca1482952fb779a600a431fc6542e7eca356",
           1, 318, 318, 0),
    _piece("896", 896, "Pyotr Ilyich Tchaikovsky", "Swan Lake: Neapolitan Dance",
           "flute", "TchaikovskyPI/dansenapolitaine/dansenapolitaine.mid",
           7936, "084540482e5522035447a44be092d38084c3308cdd2ea4f8c468b861f9fba43b",
           1, 426, 426, 0),
    _piece("1806", 1806, "Pyotr Ilyich Tchaikovsky", "March of the Wooden Soldiers",
           "upper piano", "TchaikovskyPI/O39/05MarchOfTheWoodenSoldiers/05MarchOfTheWoodenSoldiers.mid",
           2983, "a9717d793785c3c702d85e20f99a81de60faedd29fc4e7dcb240fbc99a9b0d2b",
           1, 378, 378, 0),
    _piece("522", 522, "Joseph Haydn", "Austria",
           "top voice", "HaydnFJ/Austria/Austria.mid",
           3298, "5bb7cd104ee5e1532860b922c62862458d869db90e3af84aa9f5e17c061b9867",
           1, 384, 384, 0),
    _piece("1023", 1023, "Franz Schubert", "Moment musical No.3",
           "upper piano", "SchubertF/D780/MomentsNo3/MomentsNo3.mid",
           9097, "77b5ccc3a0f33128f9f4c162743d6a484bbfaa952206b69584a4c14bafc60358",
           1, 624, 624, 47),
)

assert len(PIECES) == EXPECTED_PIECES
assert len({piece.item_id for piece in PIECES}) == EXPECTED_PIECES


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_matches(path: Path, expected_size: int,
                   expected_sha256: str) -> bool:
    path = Path(path)
    return (
        path.is_file()
        and path.stat().st_size == expected_size
        and _sha256_file(path) == expected_sha256
    )


def quantize_tick(tick: int, ticks_per_quarter: int,
                  steps_per_quarter: int = STEPS_PER_QUARTER) -> int:
    """Round a non-negative score tick to the nearest grid cell, ties up."""
    if tick < 0:
        raise ValueError(f"tick must be non-negative, got {tick}")
    if ticks_per_quarter <= 0:
        raise ValueError(
            f"ticks_per_quarter must be positive, got {ticks_per_quarter}")
    if steps_per_quarter <= 0:
        raise ValueError(
            f"steps_per_quarter must be positive, got {steps_per_quarter}")
    return (
        2 * tick * steps_per_quarter + ticks_per_quarter
    ) // (2 * ticks_per_quarter)


def verify_source_bytes(raw: bytes, spec: PieceSpec) -> None:
    """Verify a complete downloaded asset before opening it."""
    digest = _sha256(raw)
    if len(raw) != spec.source_size or digest != spec.source_sha256:
        raise SourceValidationError(
            f"{spec.item_id}: source integrity failure: got {len(raw)} bytes, "
            f"sha256 {digest}; expected {spec.source_size} bytes, "
            f"sha256 {spec.source_sha256}")


def extract_midi_bytes(raw: bytes, spec: PieceSpec) -> bytes:
    """Return the selected MIDI member from a verified asset."""
    verify_source_bytes(raw, spec)
    if spec.zip_member is None:
        return raw
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            matches = [
                info for info in archive.infolist()
                if info.filename == spec.zip_member and not info.is_dir()
            ]
            if len(matches) != 1:
                raise SourceValidationError(
                    f"{spec.item_id}: expected exactly one ZIP member "
                    f"{spec.zip_member!r}, found {len(matches)}")
            member = archive.read(matches[0])
            if spec.member_size is not None and len(member) != spec.member_size:
                raise SourceValidationError(
                    f"{spec.item_id}: ZIP member {spec.zip_member!r} has "
                    f"{len(member)} bytes; expected {spec.member_size}")
            digest = _sha256(member)
            if (spec.member_sha256 is not None
                    and digest != spec.member_sha256):
                raise SourceValidationError(
                    f"{spec.item_id}: ZIP member {spec.zip_member!r} has "
                    f"sha256 {digest}; expected {spec.member_sha256}")
            return member
    except SourceValidationError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise SourceValidationError(
            f"{spec.item_id}: invalid source ZIP: {exc}") from exc


def parse_selected_track_notes(
    raw: bytes,
    track_index: int,
    label: str = "<memory>",
) -> tuple[int, list[Note], int]:
    """Parse one MIDI track, returning PPQ, spans, and redundant note-offs.

    Same-channel/same-pitch overlaps are paired FIFO.  A redundant note-off is
    ignored and counted; the official Gymnopedie source contains two.  Active
    notes left at end of track remain a hard source error.
    """
    try:
        midi = mido.MidiFile(file=io.BytesIO(raw), clip=False)
        ticks_per_quarter = midi.ticks_per_beat
        if ticks_per_quarter <= 0:
            raise SourceValidationError(
                f"{label}: unsupported ticks-per-quarter "
                f"{ticks_per_quarter}")
        if not 0 <= track_index < len(midi.tracks):
            raise SourceValidationError(
                f"{label}: track {track_index} is out of range for "
                f"{len(midi.tracks)} tracks")

        absolute_tick = 0
        event_order = 0
        unmatched_note_offs = 0
        active: dict[tuple[int, int], deque[tuple[int, int]]] = defaultdict(deque)
        notes: list[Note] = []
        for message in midi.tracks[track_index]:
            if not isinstance(message.time, int) or message.time < 0:
                raise SourceValidationError(
                    f"{label}: invalid delta tick {message.time!r}")
            absolute_tick += message.time
            is_note_on = message.type == "note_on" and message.velocity > 0
            is_note_off = (
                message.type == "note_off"
                or (message.type == "note_on" and message.velocity == 0)
            )
            if not (is_note_on or is_note_off):
                continue
            key = (message.channel, message.note)
            if is_note_on:
                active[key].append((absolute_tick, event_order))
                event_order += 1
                continue
            if not active[key]:
                unmatched_note_offs += 1
                continue
            start_tick, order = active[key].popleft()
            notes.append(Note(
                pitch=message.note,
                start_tick=start_tick,
                end_tick=absolute_tick,
                event_order=order,
            ))

        unfinished = [
            (channel, pitch, len(starts))
            for (channel, pitch), starts in active.items() if starts
        ]
        if unfinished:
            channel, pitch, count = min(unfinished)
            raise SourceValidationError(
                f"{label}: {count} unterminated note(s) for channel "
                f"{channel}, pitch {pitch}")
        if not notes:
            raise SourceValidationError(
                f"{label}: selected track contains no matched notes")
        return ticks_per_quarter, notes, unmatched_note_offs
    except SourceValidationError:
        raise
    except (EOFError, OSError, TypeError, ValueError) as exc:
        raise SourceValidationError(f"{label}: invalid MIDI: {exc}") from exc


def encode_skyline(
    notes: Iterable[Note],
    ticks_per_quarter: int,
    max_cells: int = MAX_MELODY_CELLS,
) -> tuple[bytes, EncodingStats]:
    """Quantize notes and encode the identity-preserving upper skyline."""
    if max_cells <= 0:
        raise ValueError(f"max_cells must be positive, got {max_cells}")
    ordered = sorted(notes, key=lambda note: (
        note.start_tick, note.event_order, note.end_tick, note.pitch))
    if not ordered:
        raise SourceValidationError("selected track contains no notes")

    # Internal IDs, not pitches, define HOLD versus re-articulation.
    quantized: list[tuple[int, int, int, int, int]] = []
    collapsed = 0
    for note_id, note in enumerate(ordered):
        if not 0 <= note.pitch <= 127:
            raise SourceValidationError(f"MIDI pitch out of range: {note.pitch}")
        if note.start_tick < 0 or note.end_tick < note.start_tick:
            raise SourceValidationError(f"invalid note interval: {note}")
        start = quantize_tick(note.start_tick, ticks_per_quarter)
        end = quantize_tick(note.end_tick, ticks_per_quarter)
        if end <= start:
            collapsed += 1
            continue
        quantized.append((
            note_id, note.pitch, start, end, note.start_tick,
        ))
    if not quantized:
        raise SourceValidationError("all selected notes collapse on the q4 grid")

    original_cells = max(note[3] for note in quantized)
    starts: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    ends: dict[int, list[int]] = defaultdict(list)
    for note_id, pitch, start, end, raw_start in quantized:
        # event order is captured by note_id after the stable source ordering.
        starts[start].append((note_id, pitch, raw_start, note_id))
        ends[end].append(note_id)

    active: dict[int, tuple[int, int, int]] = {}
    melody = bytearray()
    previous_note_id: int | None = None
    for cell in range(original_cells):
        for note_id in ends.get(cell, ()):
            active.pop(note_id, None)
        for note_id, pitch, raw_start, order in starts.get(cell, ()):
            active[note_id] = (pitch, raw_start, order)
        if not active:
            melody.append(REST)
            previous_note_id = None
            continue
        selected_id, selected = max(
            active.items(), key=lambda item: item[1])
        melody.append(
            HOLD if selected_id == previous_note_id else selected[0]
        )
        previous_note_id = selected_id

    retained = bytes(melody[:max_cells])
    stats = EncodingStats(
        original_cells=original_cells,
        retained_cells=len(retained),
        collapsed_notes=collapsed,
        truncated=original_cells > max_cells,
        note_on_tokens=sum(token <= 127 for token in retained),
        hold_tokens=retained.count(HOLD),
        rest_tokens=retained.count(REST),
    )
    return retained, stats


def source_manifest_digest(
    pieces: Iterable[PieceSpec] = PIECES,
) -> str:
    """Hash every ordered source and extraction field in canonical JSON."""
    payload = json.dumps(
        [asdict(piece) for piece in pieces],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256(payload)


assert source_manifest_digest() == SOURCE_MANIFEST_SHA256


def _require_expected(
    spec: PieceSpec,
    field: str,
    actual: int | bool,
    expected: int | bool | None,
) -> None:
    if expected is not None and actual != expected:
        raise SourceValidationError(
            f"{spec.item_id}: {field} {actual}; expected {expected}")


def build_melody_stream(
    sources: Iterable[tuple[PieceSpec, bytes]],
) -> tuple[bytes, BuildStats, tuple[PieceResult, ...]]:
    """Verify and transform ordered source assets into one byte stream."""
    stream = bytearray()
    results: list[PieceResult] = []
    seen_item_ids: set[str] = set()

    for spec, source_raw in sources:
        if spec.item_id in seen_item_ids:
            raise SourceValidationError(
                f"duplicate Mutopia item ID in source order: {spec.item_id}")
        seen_item_ids.add(spec.item_id)

        midi_raw = extract_midi_bytes(source_raw, spec)
        ticks_per_quarter, notes, unmatched_note_offs = (
            parse_selected_track_notes(
                midi_raw, spec.track_index, spec.item_id)
        )
        melody, encoding = encode_skyline(notes, ticks_per_quarter)

        _require_expected(
            spec, "original q4 cells", encoding.original_cells,
            spec.expected_original_cells)
        _require_expected(
            spec, "retained melody cells", len(melody),
            spec.expected_melody_cells)
        _require_expected(
            spec, "collapsed notes", encoding.collapsed_notes,
            spec.expected_collapsed_notes)
        _require_expected(
            spec, "redundant note-offs", unmatched_note_offs,
            spec.expected_unmatched_note_offs)
        _require_expected(
            spec, "truncation flag", encoding.truncated,
            spec.expected_truncated)

        stream_offset = len(stream)
        stream.extend(melody)
        end_marker_offset = len(stream)
        stream.append(END_PIECE)
        results.append(PieceResult(
            spec=spec,
            stream_offset=stream_offset,
            melody=melody,
            end_marker_offset=end_marker_offset,
            original_cells=encoding.original_cells,
            collapsed_notes=encoding.collapsed_notes,
            unmatched_note_offs=unmatched_note_offs,
            truncated=encoding.truncated,
            matched_notes=len(notes),
            note_on_tokens=encoding.note_on_tokens,
            hold_tokens=encoding.hold_tokens,
            rest_tokens=encoding.rest_tokens,
        ))

    if not results:
        raise SourceValidationError("Mutopia source list is empty")

    note_on_tokens = sum(result.note_on_tokens for result in results)
    hold_tokens = sum(result.hold_tokens for result in results)
    rest_tokens = sum(result.rest_tokens for result in results)
    melody_tokens = note_on_tokens + hold_tokens + rest_tokens
    stats = BuildStats(
        pieces=len(results),
        matched_notes=sum(result.matched_notes for result in results),
        collapsed_notes=sum(result.collapsed_notes for result in results),
        unmatched_note_offs=sum(
            result.unmatched_note_offs for result in results),
        truncated_pieces=sum(result.truncated for result in results),
        note_on_tokens=note_on_tokens,
        hold_tokens=hold_tokens,
        rest_tokens=rest_tokens,
        end_piece_tokens=len(results),
        melody_tokens=melody_tokens,
        total_tokens=len(stream),
    )
    return bytes(stream), stats, tuple(results)


def _validate_official_build(stream: bytes, stats: BuildStats) -> None:
    expected = BuildStats(
        pieces=EXPECTED_PIECES,
        matched_notes=EXPECTED_MATCHED_NOTES,
        collapsed_notes=EXPECTED_COLLAPSED_NOTES,
        unmatched_note_offs=EXPECTED_UNMATCHED_NOTE_OFFS,
        truncated_pieces=EXPECTED_TRUNCATED_PIECES,
        note_on_tokens=EXPECTED_NOTE_ON_TOKENS,
        hold_tokens=EXPECTED_HOLD_TOKENS,
        rest_tokens=EXPECTED_REST_TOKENS,
        end_piece_tokens=EXPECTED_END_PIECE_TOKENS,
        melody_tokens=EXPECTED_MELODY_TOKENS,
        total_tokens=STREAM_SIZE,
    )
    if stats != expected:
        raise SourceValidationError(
            f"Mutopia transform statistics {stats}; expected {expected}")
    digest = _sha256(stream)
    if len(stream) != STREAM_SIZE or digest != STREAM_SHA256:
        raise SourceValidationError(
            "Mutopia transformed stream failed integrity check: "
            f"got {len(stream)} bytes, sha256 {digest}; expected "
            f"{STREAM_SIZE} bytes, sha256 {STREAM_SHA256}")


def download_source_bytes(spec: PieceSpec) -> bytes:
    """Download and verify one complete official Mutopia asset."""
    print(f"  downloading Mutopia {spec.item_id}: {spec.source_url}")
    try:
        with requests.get(
            spec.source_url, stream=True, timeout=(30, 120)
        ) as response:
            response.raise_for_status()
            content_length = response.headers.get("Content-Length")
            if (content_length is not None
                    and int(content_length) != spec.source_size):
                raise SourceValidationError(
                    f"{spec.item_id}: HTTP Content-Length {content_length}; "
                    f"expected {spec.source_size}")
            buffer = bytearray()
            for block in response.iter_content(1 << 20):
                if block:
                    buffer.extend(block)
    except SourceValidationError:
        raise
    except (OSError, requests.RequestException, ValueError) as exc:
        raise SourceValidationError(
            f"{spec.item_id}: Mutopia download failed: {exc}") from exc
    raw = bytes(buffer)
    verify_source_bytes(raw, spec)
    return raw


def derive_official_stream(
    *,
    pieces: Iterable[PieceSpec] = PIECES,
    downloader: Callable[[PieceSpec], bytes] = download_source_bytes,
) -> tuple[bytes, BuildStats, tuple[PieceResult, ...]]:
    """Download each unique pinned asset and derive the official stream."""
    piece_list = tuple(pieces)
    source_cache: dict[tuple[str, int, str], bytes] = {}
    ordered_sources: list[tuple[PieceSpec, bytes]] = []
    for spec in piece_list:
        key = (spec.source_url, spec.source_size, spec.source_sha256)
        raw = source_cache.get(key)
        if raw is None:
            raw = downloader(spec)
            verify_source_bytes(raw, spec)
            source_cache[key] = raw
        ordered_sources.append((spec, raw))

    stream, stats, results = build_melody_stream(ordered_sources)
    if piece_list == PIECES:
        _validate_official_build(stream, stats)
    return stream, stats, results


def _write_stream_cache(cache_file: Path, stream: bytes) -> Path:
    cache_file = Path(cache_file)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_file.with_name(cache_file.name + ".tmp")
    try:
        temporary.write_bytes(stream)
        temporary.replace(cache_file)
    finally:
        temporary.unlink(missing_ok=True)
    return cache_file


def ensure_stream_cache(
    stream_cache: Path = STREAM_CACHE,
    *,
    expected_size: int = STREAM_SIZE,
    expected_sha256: str = STREAM_SHA256,
    builder: Callable[
        [], tuple[bytes, BuildStats, tuple[PieceResult, ...]]
    ] = derive_official_stream,
) -> Path:
    """Return the verified canonical stream, deriving it when necessary."""
    stream_cache = Path(stream_cache)
    if _cache_matches(stream_cache, expected_size, expected_sha256):
        print(f"  Mutopia melody stream cache hit: {stream_cache}")
        return stream_cache
    if stream_cache.exists():
        print(f"  invalid Mutopia stream cache: rebuilding {stream_cache}")

    stream, stats, _ = builder()
    digest = _sha256(stream)
    if len(stream) != expected_size or digest != expected_sha256:
        raise SourceValidationError(
            "derived Mutopia stream failed cache verification: "
            f"got {len(stream)} bytes, sha256 {digest}; expected "
            f"{expected_size} bytes, sha256 {expected_sha256}")
    if expected_size == STREAM_SIZE and expected_sha256 == STREAM_SHA256:
        _validate_official_build(stream, stats)
    print(f"  caching canonical Mutopia melody stream: {stream_cache}")
    return _write_stream_cache(stream_cache, stream)


CATALOG_FIELDS = (
    "rank",
    "stream_start",
    "melody_bytes",
    "end_marker_offset",
    "item_id",
    "mutopia_id",
    "composer",
    "title",
    "selected_part",
    "track_index",
    "source_page",
    "source_asset",
    "source_member",
    "source_bytes",
    "source_sha256",
    "source_member_bytes",
    "source_member_sha256",
    "license",
    "original_q4_cells",
    "collapsed_notes",
    "unmatched_note_offs",
    "truncated",
    "melody_sha256",
)


def catalog_bytes(results: Iterable[PieceResult]) -> bytes:
    """Serialize the human-readable catalog from verified build results."""
    result_list = tuple(results)
    output = io.StringIO(newline="")
    output.write(
        "# mutopia_melody_16th v1; q4 pinned-track skyline; "
        "0-127 onset, 128 hold, 129 rest, 130 end piece\n")
    output.write(
        f"# source manifest sha256: {SOURCE_MANIFEST_SHA256}\n")
    writer = csv.DictWriter(
        output, fieldnames=CATALOG_FIELDS, delimiter="\t",
        lineterminator="\n")
    writer.writeheader()
    for rank, result in enumerate(result_list, start=1):
        spec = result.spec
        writer.writerow({
            "rank": rank,
            "stream_start": result.stream_offset,
            "melody_bytes": len(result.melody),
            "end_marker_offset": result.end_marker_offset,
            "item_id": spec.item_id,
            "mutopia_id": spec.mutopia_id,
            "composer": spec.composer,
            "title": spec.title,
            "selected_part": spec.selected_part,
            "track_index": spec.track_index,
            "source_page": spec.piece_url,
            "source_asset": spec.source_url,
            "source_member": spec.zip_member or "",
            "source_bytes": spec.source_size,
            "source_sha256": spec.source_sha256,
            "source_member_bytes": spec.member_size or "",
            "source_member_sha256": spec.member_sha256 or "",
            "license": SOURCE_LICENSE,
            "original_q4_cells": result.original_cells,
            "collapsed_notes": result.collapsed_notes,
            "unmatched_note_offs": result.unmatched_note_offs,
            "truncated": str(result.truncated).lower(),
            "melody_sha256": result.melody_sha256,
        })
    return output.getvalue().encode("utf-8")


def write_catalog(
    results: Iterable[PieceResult],
    path: Path = CATALOG_PATH,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write the checked reference catalog."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists; pass overwrite=True to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_bytes(catalog_bytes(results))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def prepare_stream(
    stream: bytes,
    context_length: int,
    out_path: Path,
    overwrite: bool = False,
) -> int:
    """Bake every complete eval window from the canonical melody stream."""
    window = window_from_context(context_length)
    num_sequences, dropped_tail = divmod(len(stream), window)
    if not num_sequences:
        raise SystemExit(
            f"{STEM}: source has {len(stream)} bytes, fewer than the "
            f"{window}-byte eval window")
    print(
        f"{STEM}: {len(stream)} melody bytes -> {num_sequences} full "
        f"{window}-byte windows; dropping {dropped_tail} tail bytes")

    def records():
        for index, sequence in enumerate(chunk_bytes(stream, window)):
            yield make_record(
                STEM,
                sequence,
                index,
                context_length,
                {
                    "source": SOURCE,
                    "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
                    "source_repository_commit": SOURCE_REPOSITORY_COMMIT,
                    "license": SOURCE_LICENSE,
                    "encoding": (
                        "midi_pitch_onset_hold_rest_end_piece_q4_v1"),
                    "steps_per_quarter": STEPS_PER_QUARTER,
                    "transform": (
                        "pinned_track_identity_skyline_nearest_half_up_"
                        "drop_collapsed_cap4095_popularity_order_v1"),
                    "piece_count": EXPECTED_PIECES,
                    "max_piece_bytes": MAX_ITEM_BYTES,
                    "end_piece_token": END_PIECE,
                    "chunk_index": index,
                },
            )

    return write_bake(
        out_path, records(), context_length, overwrite=overwrite)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Bake curated Public Domain Mutopia melodies on a q4 byte grid."))
    parser.add_argument(
        "--context-length", type=int, required=True,
        help="Target context (prefix + melody bytes); the baked window is "
             "context_length - 1")
    parser.add_argument(
        "--output", default=None,
        help="Output path (default: "
             "<data_dir>/c{context_length}/mutopia_melody_16th.jsonl)")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite an existing output file")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    window = window_from_context(args.context_length)
    if window > STREAM_SIZE:
        raise SystemExit(
            f"{STEM}: eval window {window} exceeds the complete "
            f"{STREAM_SIZE}-byte melody stream")
    out_path = (
        Path(args.output)
        if args.output
        else bake_output_path(STEM, args.context_length)
    )
    if out_path.exists() and not args.overwrite:
        raise SystemExit(
            f"{out_path} already exists; pass --overwrite to replace it")

    try:
        stream_path = ensure_stream_cache()
    except SourceValidationError as exc:
        raise SystemExit(str(exc)) from exc
    prepare_stream(
        stream_path.read_bytes(),
        args.context_length,
        out_path,
        overwrite=args.overwrite,
    )
    print("\nAll done.")


if __name__ == "__main__":
    main()
