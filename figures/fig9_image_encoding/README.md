# Figure 9: Interleaved (HWC) image encoding

Illustration of how a 32 x 32 RGB image is serialized for the image
benchmark: read row by row, with each pixel's red, green, and blue bytes
adjacent, for 32 x 32 x 3 = 3,072 bytes per image.

The thumbnail and the displayed byte values are schematic. They are not an
actual CIFAR-10 image or bytes decoded from one.

## Contents

| File | Description |
|---|---|
| `cifar10-hwc-encoding-microscope.pdf` | The figure |
| `cifar10-hwc-encoding-microscope.svg` | Editable vector source |
| `generate_cifar10_hwc_microscope.py` | Draws the PDF |

## Usage

Requires `reportlab`.

```bash
python generate_cifar10_hwc_microscope.py --output cifar10-hwc-encoding-microscope.pdf
```

The script expects DejaVu Sans at the usual Linux path; set `REGULAR_FONT` and
`BOLD_FONT` at the top of the file if yours differs.
