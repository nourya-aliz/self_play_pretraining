# Figure 3: Training value of later generator checkpoints

For each generator-training endpoint T, a fixed corpus is sampled from 16
generator snapshots and a 1M-parameter learner is trained from scratch on it.
Panel (a) shows epiplexity of the training loss; panel (b) shows
out-of-distribution validation of the resulting learners.

## Contents

| File | Description |
|---|---|
| `fig3.pdf`, `fig3.png` | The figure as published |
| `fig3.py` | Generates the figure from `data/` |
| `data/training_losses.npz` | Per-round training loss of each learner, 6 endpoints x 8 seeds |
| `data/ood_validation/<arm>/round_004095.json` | Validation bits/byte at learner round 4095, per seed and ensemble size |
| `data/subset_ensembles.csv` | Panel (b): mean, SD, and median over the 70 four-seed ensembles |
| `subset_ensembles.py` | Recomputes `data/subset_ensembles.csv` from per-token probabilities |
| `data/plotted_values.csv` | Every value in the figure |

The shipped validation slices were rescored from the released curriculum
checkpoints, so `fig3.py` redraws the figure with panel (b) values within
0.001 bits/byte of the published ones; the two are visually identical.

## Usage

```bash
python fig3.py
```

## Definitions

Epiplexity (`robust_area` in `fig3.py`): median filter of the per-round
training loss (window 33), running-minimum envelope, then the integral of the
envelope above its final value over rounds 0 to 4095. Units: nats x rounds.
One seed of the T = 4096 arm lies far above its arm-mates and is drawn at the
top edge with its value printed.

OOD validation: 256 held-out sequences per corpus at context 4096, scored as
probability ensembles over seeds. Panel (b) plots the mean and standard
deviation over all C(8,4) = 70 four-seed ensembles; the SD is descriptive
(subsets share seeds), not a standard error.

The learner checkpoints are on Hugging Face under `curriculum/T<T>_seed-*`
(generator trained for T rounds) and `curriculum/G0_seed-*` (the untrained
generator). To rebuild `data/ood_validation/`:

```bash
cd ../../scoring
python score_from_hf.py --group curriculum \
    --targets T256,T512,T1024,T2048,T4096,G0 \
    --rounds 4095 --corpora dclm,audio_16bit,cifar10_rgb_planar \
    --data-dir data/c4096 --n-seq 256 \
    --out ../figures/fig3_curriculum_value/data/ood_validation
```

Add `--save-probs` to that command to also write the per-token probabilities
of every seed; `python subset_ensembles.py` then rebuilds
`data/subset_ensembles.csv` from them.
