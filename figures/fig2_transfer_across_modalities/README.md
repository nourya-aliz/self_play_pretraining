# Figure 2: Transfer across modalities

Compute-optimal frontiers of zero-shot loss for three synthetic pretraining
distributions: self-play, programs sampled i.i.d. from a fixed universal
prior, and random probabilistic context-free grammars. Eight panels, one per
evaluation corpus; Figure 7 shows the same three arms on the full suite.

## Contents

| File | Description |
|---|---|
| `fig2.pdf`, `fig2.png` | The figure |
| `fig2.py` | Generates it from `data/` |
| `data/selfplay_frontier_perk.csv` | Self-play ladder: ensemble bits/byte per (rung, round, corpus, K) |
| `data/uniform_frontier_perk.csv` | Fixed-prior ladder, same layout |
| `data/pcfg_frontier_perk.csv` | Random-PCFG ladder, same layout |
| `data/prior_programs_per_round.json` | Programs per round of each fixed-prior rung, for the compute axis |
| `data/pcfg_programs_per_round.json` | Programs per round of each PCFG rung |
| `speech_selection.json` | Why the Speech panel shows LibriSpeech PCM8: of the four speech corpora, it has the largest loss reduction over the better baseline at a common compute budget |

## Usage

```bash
python fig2.py
```

Reads only the files under `data/`; no GPU required.

The construction: per-rung Pareto envelope over training rounds and ensemble
sizes, then a cross-rung envelope, with effective compute
`C = K * N * programs_per_round * 4096 * (round + 1)`. Self-play trains on
1536 programs per round at every rung.

The self-play and fixed-prior CSVs are flattened trajectory files. To rebuild
them from the released checkpoints (`<size>/` and
`baselines/uniform-prior/<size>/` on Hugging Face):

```bash
cd ../../scoring
python score_from_hf.py --group uniform --targets all --rounds all \
    --corpora all --data-dir data/c4096 --n-seq 256 --out cache_uniform
python build_traj_json.py --cache cache_uniform --out uniform_traj.json
python traj_to_csv.py --traj uniform_traj.json \
    --out ../figures/fig2_transfer_across_modalities/data/uniform_frontier_perk.csv
```

and likewise with `--group ladder` for `selfplay_frontier_perk.csv`.
