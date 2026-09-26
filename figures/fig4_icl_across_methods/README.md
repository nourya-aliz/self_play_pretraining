# Figure 4: In-context learning across pretraining regimes

Exact-match accuracy on six in-context tasks for three learners of matched
architecture (24.4M) and training round (2816), each a four-seed probability
ensemble. The three differ only in what they were pretrained on: self-play,
programs from the fixed universal prior, and random PCFG output.

The y axis is symlog below the smallest nonzero self-play accuracy, so tasks
a model never answers correctly sit on the axis rather than disappearing off
a log scale.

## Contents

| File | Description |
|---|---|
| `fig4.pdf`, `fig4.png` | The figure |
| `fig4.py` | Draws all three panels from `data/` |
| `data/<arm>/icl_results.json` | The m sweep for max, min, sum, mean, first, last |
| `data/<arm>/icl_v3_results.json` | Associative recall and arithmetic relations |
| `data/<arm>/icl_v4_results.json` | Reverse string and stack |
| `data/<arm>/icl_m0_results.json` | The m = 0 cells |
| `data/<arm>/icl_sum_lowm_results.json` | 4096-trial sum cells for m <= 4 |
| `data/<arm>/icl_assoc_dict_results.json` | Associative recall indexed by examples beyond one dictionary printing |

`<arm>` is `selfplay`, `uniform_prior`, or `pcfg`. The self-play files are the
same results the harness in `../fig5_icl_sum_behavior/icl_harness/` produces.

## Usage

```bash
python fig4.py
```

The self-play arm can be regenerated from the released checkpoints with that
harness (see `../fig5_icl_sum_behavior/README.md`). The universal-prior arm
uses `baselines/uniform-prior/24M/`.
