#!/usr/bin/env python3
"""Generate a compact, schematic CIFAR-10 HWC encoding microscope.

The airplane thumbnail and displayed byte values are intentionally
illustrative. They are not loaded from, or presented as, a real CIFAR-10
sample.

Requires: pip install reportlab
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


# ---------------------------------------------------------------------------
# User-editable content and styling
# ---------------------------------------------------------------------------

COLORS = {
    "text": "#202124",
    "secondary": "#626971",
    "line": "#777e86",
    "guide": "#aab0b8",
    "red_fill": "#f5dede",
    "red_stroke": "#bd5a5a",
    "green_fill": "#dcece4",
    "green_stroke": "#4f8c6f",
    "blue_fill": "#dce7f4",
    "blue_stroke": "#4f75a7",
    "sky": "#9fc5cf",
    "sea": "#4f7f96",
    "horizon": "#6e9eaa",
    "cloud": "#e5ece8",
    "cloud_alt": "#d8e5e5",
    "plane_dark": "#253e4b",
    "plane_mid": "#355565",
    "plane_accent": "#d19a56",
    "background": "#ffffff",
}

PAGE_WIDTH_IN = 7.2
PAGE_HEIGHT_IN = 2.35
DESIGN_WIDTH = 720.0
DESIGN_HEIGHT = 235.0

# These values visually agree with the schematic palette, but are qualitative
# examples rather than bytes extracted from a dataset file.
FIRST_PIXEL_BYTES = (159, 197, 207)
LAST_PIXEL_BYTES = (79, 127, 150)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "cifar10-hwc-encoding-microscope.pdf"

# Update these paths if DejaVu Sans is installed elsewhere.
REGULAR_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
BOLD_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

PAGE_W = PAGE_WIDTH_IN * inch
PAGE_H = PAGE_HEIGHT_IN * inch
SCALE = PAGE_W / DESIGN_WIDTH


def color(name: str) -> HexColor:
    return HexColor(COLORS[name])


def yy(y: float) -> float:
    """Convert top-origin design coordinates to ReportLab coordinates."""
    return DESIGN_HEIGHT - y


def register_fonts() -> None:
    if not REGULAR_FONT.exists() or not BOLD_FONT.exists():
        raise FileNotFoundError(
            "DejaVu Sans was not found. Edit REGULAR_FONT and BOLD_FONT near "
            "the top of this script to point to your .ttf files."
        )
    pdfmetrics.registerFont(TTFont("DejaVuSans", str(REGULAR_FONT)))
    pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", str(BOLD_FONT)))


def draw_text(
    c: canvas.Canvas,
    x: float,
    y: float,
    value: str,
    *,
    size: float = 14,
    fill: HexColor | None = None,
    bold: bool = False,
    align: str = "left",
) -> None:
    c.setFillColor(fill or color("text"))
    c.setFont("DejaVuSans-Bold" if bold else "DejaVuSans", size)
    baseline = yy(y)
    if align == "right":
        c.drawRightString(x, baseline, value)
    elif align == "center":
        c.drawCentredString(x, baseline, value)
    else:
        c.drawString(x, baseline, value)


def draw_centered_text(
    c: canvas.Canvas,
    x: float,
    y: float,
    value: str,
    *,
    size: float = 14,
    fill: HexColor | None = None,
    bold: bool = False,
) -> None:
    """Center text optically around a design-space x/y point."""
    c.setFillColor(fill or color("text"))
    c.setFont("DejaVuSans-Bold" if bold else "DejaVuSans", size)
    c.drawCentredString(x, yy(y) - size * 0.34, value)


def draw_line(
    c: canvas.Canvas,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: HexColor | None = None,
    width: float = 1,
    dash: list[float] | None = None,
) -> None:
    c.setStrokeColor(stroke or color("line"))
    c.setLineWidth(width)
    c.setDash(dash or [])
    c.line(x1, yy(y1), x2, yy(y2))
    c.setDash([])


def draw_rect(
    c: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: HexColor | None = None,
    stroke: HexColor | None = None,
    line_width: float = 1,
) -> None:
    if fill is not None:
        c.setFillColor(fill)
    if stroke is not None:
        c.setStrokeColor(stroke)
    c.setLineWidth(line_width)
    c.rect(
        x,
        yy(y + height),
        width,
        height,
        fill=1 if fill is not None else 0,
        stroke=1 if stroke is not None else 0,
    )


def draw_arrow(
    c: canvas.Canvas,
    x1: float,
    y: float,
    x2: float,
    *,
    stroke: HexColor | None = None,
    width: float = 1,
) -> None:
    arrow_color = stroke or color("line")
    draw_line(c, x1, y, x2 - 1.2, y, stroke=arrow_color, width=width)
    c.setFillColor(arrow_color)
    c.setStrokeColor(arrow_color)
    path = c.beginPath()
    path.moveTo(x2, yy(y))
    path.lineTo(x2 - 6, yy(y - 3))
    path.lineTo(x2 - 6, yy(y + 3))
    path.close()
    c.drawPath(path, fill=1, stroke=0)


def draw_bracket(c: canvas.Canvas, x1: float, x2: float, y: float = 163) -> None:
    draw_line(c, x1, y, x1, y + 7)
    draw_line(c, x1, y + 7, x2, y + 7)
    draw_line(c, x2, y + 7, x2, y)


def draw_schematic_airplane(
    c: canvas.Canvas, x: float = 105, y: float = 26, size: float = 68
) -> None:
    """Draw a deliberately schematic 12-by-12 airplane thumbnail."""
    unit = size / 12.0
    draw_rect(c, x, y, size, size, fill=color("sky"))

    # Each tuple is (x, y, width, height, palette key) in the 12-by-12 grid.
    blocks = [
        (0, 8, 12, 4, "sea"),
        (0, 8, 12, 1, "horizon"),
        (1, 1, 3, 1, "cloud"),
        (2, 2, 4, 1, "cloud"),
        (8, 2, 2, 1, "cloud_alt"),
        (2, 5, 8, 2, "plane_dark"),
        (3, 4, 5, 1, "plane_mid"),
        (4, 3, 2, 2, "plane_accent"),
        (5, 7, 2, 2, "plane_accent"),
        (9, 4, 1, 1, "plane_dark"),
        (1, 6, 2, 1, "plane_mid"),
    ]
    for gx, gy, gw, gh, palette_key in blocks:
        draw_rect(
            c,
            x + gx * unit,
            y + gy * unit,
            gw * unit,
            gh * unit,
            fill=color(palette_key),
        )

    draw_rect(c, x, y, size, size, stroke=color("line"), line_width=1)


def draw_byte_cell(
    c: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    channel: str,
    value: int | None = None,
) -> None:
    draw_rect(
        c,
        x,
        y,
        width,
        height,
        fill=color(f"{channel}_fill"),
        stroke=color(f"{channel}_stroke"),
        line_width=1.25,
    )
    if value is not None:
        draw_centered_text(c, x + width / 2, y + height / 2, str(value), bold=True)


# ---------------------------------------------------------------------------
# Figure construction
# ---------------------------------------------------------------------------


def build(output: Path) -> None:
    register_fonts()
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(
        str(output),
        pagesize=(PAGE_W, PAGE_H),
        pageCompression=1,
        invariant=1,
    )
    c.setTitle("CIFAR-10 HWC encoding microscope (schematic)")
    c.setSubject(
        "Qualitative HWC layout; the airplane thumbnail and byte values are schematic."
    )
    c.setAuthor("")
    c.scale(SCALE, SCALE)

    draw_rect(c, 0, 0, DESIGN_WIDTH, DESIGN_HEIGHT, fill=color("background"))

    # Labels and format note.
    draw_text(
        c,
        696,
        18,
        "schematic 32 × 32 RGB · HWC order",
        size=12,
        fill=color("secondary"),
        align="right",
    )
    draw_text(c, 94, 62, "image", fill=color("secondary"), bold=True, align="right")
    draw_text(c, 94, 128, "bytes", fill=color("secondary"), bold=True, align="right")
    draw_text(c, 94, 184, "record", fill=color("secondary"), bold=True, align="right")

    # A compact qualitative thumbnail and row-major traversal guides.
    draw_schematic_airplane(c)
    for row_y, label in ((35, "row 0"), (55, "row 1"), (85, "row 31")):
        draw_text(c, 188, row_y + 4, label, size=12, fill=color("secondary"))
        draw_arrow(c, 232, row_y, 691)
    draw_text(c, 210, 72, "⋮", size=12, fill=color("secondary"), align="center")

    # Dashed visual guides from the image span to the byte record span.
    draw_line(c, 105, 96, 105, 104, stroke=color("guide"), dash=[4, 4])
    draw_line(c, 693, 96, 693, 104, stroke=color("guide"), dash=[4, 4])

    # First and last adjacent HWC triplets, with the middle of the record elided.
    channels = ("red", "green", "blue")
    for index, (channel, value) in enumerate(zip(channels, FIRST_PIXEL_BYTES)):
        draw_byte_cell(c, 105 + index * 42, 104, 42, 42, channel, value)

    draw_line(c, 231, 125, 567, 125, stroke=color("guide"))
    c.setFillColor(color("line"))
    for cx in (390, 399, 408):
        c.circle(cx, yy(125), 2, fill=1, stroke=0)

    for index, (channel, value) in enumerate(zip(channels, LAST_PIXEL_BYTES)):
        draw_byte_cell(c, 567 + index * 42, 104, 42, 42, channel, value)

    # Full record extent and size.
    draw_bracket(c, 107, 691)
    draw_text(
        c,
        399,
        190,
        "32 × 32 pixels × 3 channels × 1 byte = 3,072 bytes",
        size=12,
        fill=color("secondary"),
        align="center",
    )

    # HWC legend: channels for a pixel are contiguous in the byte stream.
    for index, channel in enumerate(channels):
        draw_byte_cell(c, 105 + index * 22, 207, 22, 15, channel)
    draw_text(
        c,
        181,
        219,
        "R, G, B are adjacent for each pixel",
        size=12,
        fill=color("secondary"),
    )

    c.showPage()
    c.save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output PDF path (default: {DEFAULT_OUTPUT.name}, beside this script)",
    )
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
