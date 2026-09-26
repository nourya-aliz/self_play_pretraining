#!/usr/bin/env python3
"""Generate a compact text-to-UTF-8-byte map.

Characters appear directly above the byte cells that encode them. Long excerpts
wrap at whitespace without dropping any character or byte.

Requires: pip install reportlab
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


# ---------------------------------------------------------------------------
# User-editable defaults and styling
# ---------------------------------------------------------------------------

DEFAULT_TEXT = "The process is mutual; for men learn while they teach."

COLORS = {
    "text": "#202124",
    "secondary": "#626971",
    "guide": "#aab0b8",
    "ascii_fill": "#f2f3f5",
    "ascii_stroke": "#aab0b8",
    "multi_fill": "#dce8f6",
    "multi_stroke": "#2e5f8a",
    "target_fill": "#f8dfd7",
    "target_stroke": "#a94f38",
    "background": "#ffffff",
}

PAGE_WIDTH_IN = 7.2
MIN_PAGE_HEIGHT_IN = 2.35
DEFAULT_BYTES_PER_ROW = 24
MAX_BYTES_PER_ROW = 26

REGULAR_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
BOLD_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


# ---------------------------------------------------------------------------
# Geometry and helpers
# ---------------------------------------------------------------------------

PAGE_W = PAGE_WIDTH_IN * inch
SCALE = PAGE_W / 720.0
DESIGN_HEIGHT = 235.0
STREAM_LEFT = 105.0
STREAM_RIGHT = 693.0


@dataclass(frozen=True)
class CharacterSpan:
    character: str
    byte_values: tuple[int, ...]
    start: int
    end: int


def color(name: str):
    return HexColor(COLORS[name])


def yy(y: float) -> float:
    """Convert top-origin design coordinates to ReportLab coordinates."""
    return DESIGN_HEIGHT - y


def register_fonts() -> None:
    if not REGULAR_FONT.exists() or not BOLD_FONT.exists():
        raise FileNotFoundError(
            "DejaVu Sans was not found. Edit REGULAR_FONT and BOLD_FONT near "
            "the top of this script to point to installed .ttf files."
        )
    pdfmetrics.registerFont(TTFont("DejaVuSans", str(REGULAR_FONT)))
    pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", str(BOLD_FONT)))


def draw_text(c, x, y, value, size=14, fill=None, bold=False, align="left"):
    c.setFillColor(fill or color("text"))
    c.setFont("DejaVuSans-Bold" if bold else "DejaVuSans", size)
    baseline = yy(y)
    if align == "right":
        c.drawRightString(x, baseline, value)
    elif align == "center":
        c.drawCentredString(x, baseline, value)
    else:
        c.drawString(x, baseline, value)


def draw_centered_text(c, x, y, value, size=14, fill=None, bold=False):
    c.setFillColor(fill or color("text"))
    c.setFont("DejaVuSans-Bold" if bold else "DejaVuSans", size)
    c.drawCentredString(x, yy(y) - size * 0.34, value)


def draw_line(c, x1, y1, x2, y2, stroke=None, width=1, dash=None):
    c.setStrokeColor(stroke or color("guide"))
    c.setLineWidth(width)
    c.setDash(dash or [])
    c.line(x1, yy(y1), x2, yy(y2))
    c.setDash([])


def draw_rect(c, x, y, width, height, fill, stroke, line_width=1):
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(line_width)
    c.rect(x, yy(y + height), width, height, fill=1, stroke=1)


def draw_bracket(c, x1, x2, y=163):
    draw_line(c, x1, y, x1, y + 7, width=1.2)
    draw_line(c, x1, y + 7, x2, y + 7, width=1.2)
    draw_line(c, x2, y + 7, x2, y, width=1.2)


def character_spans(text: str) -> tuple[list[CharacterSpan], list[int]]:
    spans: list[CharacterSpan] = []
    all_bytes: list[int] = []
    for character in text:
        values = tuple(character.encode("utf-8"))
        start = len(all_bytes)
        all_bytes.extend(values)
        spans.append(CharacterSpan(character, values, start, len(all_bytes)))
    return spans, all_bytes


def wrap_spans(
    spans: list[CharacterSpan], bytes_per_row: int
) -> list[list[CharacterSpan]]:
    """Wrap at the latest whitespace whose bytes fit on the current row."""
    rows: list[list[CharacterSpan]] = []
    start = 0
    while start < len(spans):
        end = start
        byte_count = 0
        last_whitespace_end: int | None = None

        while end < len(spans):
            span_width = len(spans[end].byte_values)
            if byte_count + span_width > bytes_per_row:
                break
            byte_count += span_width
            end += 1
            if spans[end - 1].character.isspace():
                last_whitespace_end = end

        if end == len(spans):
            row_end = end
        elif last_whitespace_end is not None and last_whitespace_end > start:
            row_end = last_whitespace_end
        else:
            row_end = end

        if row_end == start:
            raise ValueError(
                "A single character needs more bytes than --bytes-per-row allows."
            )
        rows.append(spans[start:row_end])
        start = row_end

    return rows


def visible_character(character: str) -> str:
    """Make spaces visible without changing the byte sequence below."""
    if character == " ":
        return "·"
    if character == "\t":
        return "⇥"
    return character


# ---------------------------------------------------------------------------
# Figure construction
# ---------------------------------------------------------------------------

def build(
    output: Path,
    text: str,
    target_last: bool = False,
    bytes_per_row: int = DEFAULT_BYTES_PER_ROW,
) -> None:
    global DESIGN_HEIGHT

    if not text:
        raise ValueError("The text must contain at least one character.")
    if "\n" in text or "\r" in text:
        raise ValueError("Use a single-line excerpt without newline characters.")
    if not 12 <= bytes_per_row <= MAX_BYTES_PER_ROW:
        raise ValueError(
            f"--bytes-per-row must be between 12 and {MAX_BYTES_PER_ROW}."
        )

    spans, byte_values = character_spans(text)
    rows = wrap_spans(spans, bytes_per_row)

    register_fonts()
    output.parent.mkdir(parents=True, exist_ok=True)

    available_width = STREAM_RIGHT - STREAM_LEFT
    longest_row = max(row[-1].end - row[0].start for row in rows)
    cell_width = min(42.0, available_width / longest_row)
    byte_font_size = 14 if cell_width >= 36 else 12 if cell_width >= 30 else 10
    char_font_size = 18 if cell_width >= 32 else 15

    row_step = 92.0
    first_character_y = 52.0
    final_cell_bottom = first_character_y + (len(rows) - 1) * row_step + 60.0
    DESIGN_HEIGHT = max(MIN_PAGE_HEIGHT_IN * 100.0, final_cell_bottom + 46.0)
    page_height = DESIGN_HEIGHT * SCALE

    c = canvas.Canvas(str(output), pagesize=(PAGE_W, page_height), pageCompression=1)
    c.setTitle("DCLM text-to-byte map")
    c.setAuthor("")
    c.scale(SCALE, SCALE)

    c.setFillColor(color("background"))
    c.rect(0, 0, 720, DESIGN_HEIGHT, fill=1, stroke=0)

    draw_text(
        c,
        696,
        18,
        "UTF-8 - 1 cell = 1 byte",
        size=12,
        fill=color("secondary"),
        align="right",
    )
    for row_index, row in enumerate(rows):
        character_y = first_character_y + row_index * row_step
        cell_y = character_y + 26.0
        row_start = row[0].start
        row_end = row[-1].end
        row_byte_values = byte_values[row_start:row_end]
        row_width = len(row_byte_values) * cell_width
        stream_x = STREAM_LEFT + (available_width - row_width) / 2

        draw_text(
            c,
            94,
            character_y + 4,
            "text",
            fill=color("secondary"),
            bold=True,
            align="right",
        )
        draw_text(
            c,
            94,
            cell_y + 23,
            "byte",
            fill=color("secondary"),
            bold=True,
            align="right",
        )

        # Characters are centered over the exact byte span that encodes them.
        for span in row:
            local_start = span.start - row_start
            local_end = span.end - row_start
            x1 = stream_x + local_start * cell_width
            x2 = stream_x + local_end * cell_width
            center = (x1 + x2) / 2
            is_multibyte = len(span.byte_values) > 1
            is_space = span.character.isspace()
            draw_centered_text(
                c,
                center,
                character_y,
                visible_character(span.character),
                size=11 if is_space else char_font_size,
                fill=(
                    color("secondary")
                    if is_space
                    else color("multi_stroke")
                    if is_multibyte
                    else color("text")
                ),
                bold=is_multibyte,
            )
            draw_line(
                c,
                center,
                character_y + 14,
                center,
                cell_y - 2,
                stroke=color("guide"),
                dash=[3, 3],
            )

        # Raw decimal byte values, matching the paper's 0-255 interface.
        byte_owner = [
            span
            for span in row
            for _ in range(len(span.byte_values))
        ]
        for local_index, (value, span) in enumerate(
            zip(row_byte_values, byte_owner, strict=True)
        ):
            global_index = row_start + local_index
            x = stream_x + local_index * cell_width
            is_multibyte = len(span.byte_values) > 1
            is_target = target_last and global_index == len(byte_values) - 1
            if is_target:
                fill = color("target_fill")
                stroke = color("target_stroke")
                line_width = 1.6
            elif is_multibyte:
                fill = color("multi_fill")
                stroke = color("multi_stroke")
                line_width = 1.6
            else:
                fill = color("ascii_fill")
                stroke = color("ascii_stroke")
                line_width = 1.0
            draw_rect(c, x, cell_y, cell_width, 34, fill, stroke, line_width=line_width)
            draw_centered_text(
                c,
                x + cell_width / 2,
                cell_y + 17,
                str(value),
                size=byte_font_size,
                bold=True,
            )

    footer_parts = ["spaces shown as ·", "decimal UTF-8 byte values"]
    if any(len(span.byte_values) > 1 for span in spans):
        footer_parts.append("blue = multibyte character")
    if target_last:
        footer_parts.append("coral = next-byte target")
    draw_text(
        c,
        696,
        DESIGN_HEIGHT - 14,
        "  |  ".join(footer_parts),
        size=10,
        fill=color("secondary"),
        align="right",
    )

    c.showPage()
    c.save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help="Single-line UTF-8 excerpt to visualize.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dclm-text-encoding-microscope.pdf"),
        help="Output PDF path.",
    )
    parser.add_argument(
        "--target-last",
        action="store_true",
        help="Highlight the final byte as the next-byte prediction target.",
    )
    parser.add_argument(
        "--bytes-per-row",
        type=int,
        default=DEFAULT_BYTES_PER_ROW,
        help=(
            "Maximum encoded bytes per row; wrapping prefers whitespace "
            f"(default: {DEFAULT_BYTES_PER_ROW}, maximum: {MAX_BYTES_PER_ROW})."
        ),
    )
    args = parser.parse_args()
    build(
        args.output,
        args.text,
        target_last=args.target_last,
        bytes_per_row=args.bytes_per_row,
    )


if __name__ == "__main__":
    main()
