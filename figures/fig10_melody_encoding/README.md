# Figure 10: Melody byte encoding

Illustration of the melody benchmark's encoding on the opening motif of
Beethoven's Fifth Symphony. Each cell is one sixteenth note: bytes 0 to 127
start a MIDI pitch and byte 128 holds the previous note. The encoded
sequence is:

```text
[67, 128, 67, 128, 67, 128, 63, 128, 128, 128, 128, 128, 128, 128]
```

## Contents

| File | Description |
|---|---|
| `beethoven-fifth-encoding-microscope.pdf` | The figure |
| `beethoven-fifth-encoding-microscope.svg` | Editable vector source |
| `generate_beethoven_fifth_microscope.py` | Draws the PDF |

## Usage

Requires `reportlab`.

```bash
python generate_beethoven_fifth_microscope.py --output beethoven-fifth-encoding-microscope.pdf
```

The script expects DejaVu Sans at the usual Linux path; set `REGULAR_FONT` and
`BOLD_FONT` at the top of the file if yours differs.
