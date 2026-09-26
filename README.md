# Self-Play Pretraining with Zero Data

Figures, tables, and evaluation code for the paper *Self-Play Pretraining with
Zero Data*. Preliminary model weights are released on Hugging Face:
[nourya-cohen/solomonoff-paper](https://huggingface.co/nourya-cohen/solomonoff-paper).

```
figures/     one folder per figure or table: the published figure,
             the code that generates it, and the data it reads
scoring/     evaluate the released checkpoints on the benchmark corpora
```

## Figures

| Folder | Paper item |
|---|---|
| `figures/fig1_overview/` | Figure 1: compute frontiers and in-context learning |
| `figures/fig2_transfer_across_modalities/` | Figure 2: transfer across modalities, three pretraining arms |
| `figures/fig3_curriculum_value/` | Figure 3: training value of later generator checkpoints |
| `figures/fig4_icl_across_methods/` | Figure 4: in-context learning across pretraining regimes |
| `figures/fig5_icl_sum_behavior/` | Figure 5: in-context SUM task behavior |
| `figures/fig6_pre_pretraining/` | Figure 6: pre-pretraining accelerates pretraining on natural data |
| `figures/fig7_scaling_all_datasets/` | Figure 7: scaling laws across all evaluation datasets |
| `figures/fig8_text_encoding/` | Figure 8: natural-text byte encoding |
| `figures/fig9_image_encoding/` | Figure 9: interleaved (HWC) image encoding |
| `figures/fig10_melody_encoding/` | Figure 10: melody byte encoding |
| `figures/table1_discovered_structures/` | Tables 1 and 3: discovered mathematical structure |
| `figures/table2_scaling_exponents/` | Table 2: per-modality compute exponents |
| `figures/table5_reward_ablations/` | Table 5: generator reward ablations |

Each folder ships the data its plot script reads, so a figure regenerates with
`numpy`, `scipy`, and `matplotlib` alone (Figures 8 to 10 also need `reportlab`):

```bash
cd figures/fig3_curriculum_value
python fig3.py --seeds 8 --subsets 4
```

See each folder's README for the exact command.

## Scoring

`scoring/` recomputes the evaluation results those figures are built on,
directly from the released checkpoints: it bakes the benchmark corpora from
their public sources and evaluates the Hugging Face weights.

```bash
cd scoring
python -m src.scripts.prepare_dclm_benchmark --context-length 4096 \
    --output data/c4096/dclm.jsonl --overwrite
python score_from_hf.py --group ladder --targets 24M --rounds 8191 \
    --corpora dclm --data-dir data/c4096 --n-seq 256 --out cache
```

Requires `torch`, `huggingface_hub`, and the packages listed in
`scoring/README.md`.

## Reproducing from the checkpoints

Each figure folder ships the data its plot script reads, so the figures
regenerate without a GPU. To rebuild that data from the released weights
instead (bake the corpora, score the checkpoints, refit), follow
[REPRODUCING.md](REPRODUCING.md).

## Checkpoints

```
<size>/seed-<seed>/learner_<round>.pth      self-play ladder (100k ... 24M)
baselines/uniform-prior/<size>/...          fixed-prior baseline ladder
ablations/reward-arms/<arm>/...             reward-ablation arms
curriculum/<arm>_seed-<seed>/...             curriculum-corpus learners (Figure 3)
```

Learner weights are saved every 256 self-play rounds. The `<size>` folder
names are the ladder's rung labels; the paper refers to the same models by
their total parameter count:

| Folder | Paper | Parameters | d_model / heads / layers |
|---|---|---|---|
| `100k` | 99k | 98,496 | 64 / 1 / 1 |
| `500k` | 558k | 557,696 | 128 / 2 / 2 |
| `1M` | 1M | 1,049,728 | 128 / 2 / 4 |
| `3M` | 3.1M | 3,148,032 | 256 / 4 / 4 |
| `6M` | 6.2M | 6,164,736 | 256 / 4 / 8 |
| `24M` | 24.4M | 24,388,096 | 512 / 8 / 8 |
