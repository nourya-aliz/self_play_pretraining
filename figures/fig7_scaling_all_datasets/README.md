# Figure 7: Scaling laws across all evaluation datasets

Figure 2's three pretraining arms on every corpus in the evaluation suite,
grouped by modality: text, code and formal, speech audio, sound and music
audio, vision/melody/DNA, and binary/numeric/synthetic.

## Contents

| File | Description |
|---|---|
| `fig7.pdf`, `fig7.png` | The figure |
| `fig7.py` | Generates it |

The frontier CSVs are the ones in
`../fig2_transfer_across_modalities/data/` and are read from there rather
than duplicated, so that folder must be present.

## Usage

```bash
python fig7.py
```

The compute axis, envelope construction, and fits are identical to Figure 2;
only the panel set differs.
