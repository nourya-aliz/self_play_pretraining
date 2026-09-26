# Figure 6: Pre-pretraining accelerates pretraining on natural data

Validation BPB while pretraining a 24.4M model on three byte-encoded
modalities, from random initialization and from the final self-play
checkpoint. The leftmost point is the loss before any natural-data training;
for the warm start that is its zero-shot transfer.

## Contents

| File | Description |
|---|---|
| `fig6.pdf`, `fig6.png` | The figure as published; `fig6.py` redraws it identically |
| `fig6.py` | Generates it |
| `fig6_data_cache.json` | Exactly the plotted data: per panel and arm, the hyperparameters, per-seed curves, initial losses, and convergence points |

## Usage

```bash
python fig6.py
```

Reads `fig6_data_cache.json` only.

## Setup

Both arms are trained with the same protocol: constant learning rate with 2%
warmup until validation BPB plateaus (improvement < 0.005 for 5 consecutive
evaluations, measured every 38 steps), then a 200-step cosine decay to zero.
The reported convergence point is the post-decay final loss. Learning rate and
weight decay are tuned separately for each arm; weight decay applies to all
parameters. Curves are the mean over 4 seeds with the shaded band spanning the
seed range.

Selected hyperparameters:

| Panel | From scratch | Self-play warm start |
|---|---|---|
| DCLM (text) | lr 1e-3, wd 0.1 | lr 3e-3, wd 0.1 |
| CIFAR-10 (images) | lr 3e-3, wd 0.0 | lr 3e-3, wd 0.1 |
| ESC-50 (audio) | lr 3e-3, wd 0.1 | lr 3e-3, wd 0.1 |

The token axis is logarithmic from the first evaluation onward; the leftmost
tick is a placeholder for zero tokens. Curves are lightly smoothed for
legibility, with the initial and final points drawn unsmoothed; convergence
markers are offset vertically so the two arms do not overlap.
