# Figure 8: Natural-text byte encoding

Illustration of how the text benchmark reaches the model: each character of an
ASCII excerpt is shown above its UTF-8 byte value, with spaces drawn as middle
dots (byte 32). The byte row is exactly `list(TEXT.encode("utf-8"))`.

## Contents

| File | Description |
|---|---|
| `dclm-text-encoding-microscope.pdf`, `.png` | The figure |
| `generate_dclm_text_microscope.py` | Draws it from any text excerpt |

## Usage

Requires `reportlab`.

```bash
python generate_dclm_text_microscope.py \
    --text "The process is mutual; for men learn while they teach." \
    --output dclm-text-encoding-microscope.pdf
```

`--bytes-per-row` sets the wrap width (default 24, maximum 26). `--target-last`
highlights the final cell to show the next-byte prediction target.
