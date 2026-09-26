#!/usr/bin/env python3
"""Generate the editable Beethoven Fifth encoding-microscope PDF.

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

SEQUENCE = [67, 128, 67, 128, 67, 128, 63, 128, 128, 128, 128, 128, 128, 128]
ONSET_INDICES = {0, 2, 4, 6}

COLORS = {
    "text": "#202124",
    "secondary": "#626971",
    "staff": "#777e86",
    "guide": "#aab0b8",
    "onset_fill": "#dce8f6",
    "onset_stroke": "#2e5f8a",
    "hold_fill": "#f2f3f5",
    "hold_stroke": "#aab0b8",
    "background": "#ffffff",
}

PAGE_WIDTH_IN = 7.2
PAGE_HEIGHT_IN = 2.35

# Update these paths if DejaVu Sans is installed elsewhere.
REGULAR_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
BOLD_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

PAGE_W = PAGE_WIDTH_IN * inch
PAGE_H = PAGE_HEIGHT_IN * inch
SCALE = PAGE_W / 720.0
DESIGN_HEIGHT = 235.0


def color(name: str):
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
    c.setStrokeColor(stroke or color("staff"))
    c.setLineWidth(width)
    c.setDash(dash or [])
    c.line(x1, yy(y1), x2, yy(y2))
    c.setDash([])


def draw_rect(c, x, y, width, height, fill, stroke, line_width=1):
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(line_width)
    c.rect(x, yy(y + height), width, height, fill=1, stroke=1)


def draw_note(c, cx, cy, open_note=False, accidental=False):
    if accidental:
        draw_text(c, cx - 19, cy + 5, "♭", size=18, bold=True)

    c.saveState()
    c.translate(cx, yy(cy))
    c.rotate(18)
    c.setFillColor(color("background") if open_note else color("text"))
    c.setStrokeColor(color("text"))
    c.setLineWidth(1.8 if open_note else 1.6)
    c.ellipse(-6.8, -4.6, 6.8, 4.6, fill=1, stroke=1)
    c.restoreState()

    stem_top = cy - (28 if open_note else 25)
    draw_line(c, cx + 5, cy, cx + 5, stem_top, stroke=color("text"), width=1.6)

    if not open_note:
        c.setStrokeColor(color("text"))
        c.setLineWidth(1.6)
        path = c.beginPath()
        path.moveTo(cx + 5, yy(stem_top))
        path.curveTo(
            cx + 14,
            yy(stem_top + 3),
            cx + 15,
            yy(stem_top + 9),
            cx + 10,
            yy(stem_top + 13),
        )
        c.drawPath(path, fill=0, stroke=1)


def draw_bracket(c, x1, x2, y=163):
    draw_line(c, x1, y, x1, y + 7)
    draw_line(c, x1, y + 7, x2, y + 7)
    draw_line(c, x2, y + 7, x2, y)


# ---------------------------------------------------------------------------
# Figure construction
# ---------------------------------------------------------------------------

def build(output: Path) -> None:
    register_fonts()
    output.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(output), pagesize=(PAGE_W, PAGE_H), pageCompression=1)
    c.setTitle("Beethoven Fifth encoding microscope")
    c.setAuthor("")
    c.scale(SCALE, SCALE)

    c.setFillColor(color("background"))
    c.rect(0, 0, 720, 235, fill=1, stroke=0)

    draw_text(c, 696, 18, "1 cell = 1/16 note", size=12, fill=color("secondary"), align="right")
    draw_text(c, 94, 58, "notes", fill=color("secondary"), bold=True, align="right")
    draw_text(c, 94, 128, "byte", fill=color("secondary"), bold=True, align="right")
    draw_text(c, 94, 184, "duration", fill=color("secondary"), bold=True, align="right")

    for staff_y in (35, 43, 51, 59, 67):
        draw_line(c, 105, staff_y, 693, staff_y, stroke=color("staff"))

    draw_note(c, 126, 59)
    draw_note(c, 210, 59)
    draw_note(c, 294, 59)
    draw_note(c, 378, 67, open_note=True, accidental=True)

    for guide_x in (126, 210, 294, 378):
        draw_line(c, guide_x, 76, guide_x, 104, stroke=color("guide"), dash=[4, 4])

    for index, value in enumerate(SEQUENCE):
        x = 105 + index * 42
        if index in ONSET_INDICES:
            draw_rect(
                c,
                x,
                104,
                42,
                42,
                color("onset_fill"),
                color("onset_stroke"),
                line_width=1.6,
            )
        else:
            draw_rect(
                c,
                x,
                104,
                42,
                42,
                color("hold_fill"),
                color("hold_stroke"),
            )
        draw_centered_text(c, x + 21, 125, str(value), bold=True)

    groups = [
        (107, 187, 147, "G4 · 2 cells"),
        (191, 271, 231, "G4 · 2 cells"),
        (275, 355, 315, "G4 · 2 cells"),
        (359, 691, 525, "E♭4 · 8 cells"),
    ]
    for x1, x2, center, label in groups:
        draw_bracket(c, x1, x2)
        draw_text(c, center, 190, label, size=12, fill=color("secondary"), align="center")

    draw_rect(c, 105, 207, 22, 15, color("onset_fill"), color("onset_stroke"), line_width=1.4)
    draw_text(c, 136, 219, "pitch onset (0–127)", size=12, fill=color("secondary"))
    draw_rect(c, 287, 207, 22, 15, color("hold_fill"), color("hold_stroke"))
    draw_text(c, 318, 219, "hold previous note (128)", size=12, fill=color("secondary"))

    c.showPage()
    c.save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("beethoven-fifth-encoding-microscope.pdf"),
        help="Output PDF path",
    )
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
