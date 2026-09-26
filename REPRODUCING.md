# Reproducing the results from the released checkpoints

Every figure regenerates from the data shipped in its folder with
`numpy`/`scipy`/`matplotlib` alone (Figures 8 to 10 also need `reportlab`). This document covers rebuilding that data
from the checkpoints on
[nourya-cohen/solomonoff-paper](https://huggingface.co/nourya-cohen/solomonoff-paper).

Install `torch`, `huggingface_hub`, and `jvp_flash_attention` (CUDA sm_75+) in
addition to the plotting packages.

## 1. Bake the benchmark corpora

Each script downloads its public source and writes a byte corpus at the
model's context length:

```bash
cd scoring
for s in dclm cifar10 speech_commands mutopia esc50 musicnet kolmogorov \
         metamath arithmetic aitdcc llm_compression; do
    python -m src.scripts.prepare_${s}_benchmark --context-length 4096
done
```

Some scripts expose several corpora through `--datasets` (for example
`prepare_kolmogorov_benchmark --datasets kolmogorov_dna`); see each script's
`--help`. Four corpora (`audio_8bit`, `audio_16bit`, `dna`, `dclm_ranked`)
are shipped already baked in `scoring/data/c4096/`. The corpora used by the
paper are listed in `figures/table2_scaling_exponents/exponents_table.tex`.

## 2. Score the checkpoints

```bash
python score_from_hf.py --group ladder \
    --targets 100k,500k,1M,3M,6M,24M --rounds all \
    --corpora all --data-dir data/c4096 --n-seq 256 --out cache_ladder
```

Groups: `ladder` (self-play ladder), `uniform` (fixed-prior baseline),
`reward` (Table 5 arms), `curriculum` (Figure 3 learners). Use a separate
`--out` directory per group. Scoring is incremental: an existing slice is
skipped, so a sweep can be interrupted and resumed. Checkpoints are saved
every 256 self-play rounds, up to round 8191.

## 3. Assemble and plot

Each figure folder's README gives the exact commands for that figure; in
outline:

**Table 2 and the Figure 1 scaling panel**: compute exponents and frontiers:

```bash
python build_traj_json.py --cache cache_ladder \
    --out ../figures/table2_scaling_exponents/frontier_traj_perk.json
cp ../figures/table2_scaling_exponents/frontier_traj_perk.json ../figures/fig1_overview/
cd ../figures/table2_scaling_exponents && python make_latex_table.py \
    && python check_fig1_table_consistency.py
cd ../fig1_overview && python fig1.py
```

**Figures 2 and 7**: the three pretraining arms across modalities. Score both
released groups, then flatten each trajectory into the CSV the figures read:

```bash
cd scoring
python build_traj_json.py --cache cache_ladder --out ladder_traj.json
python build_traj_json.py --cache cache_uniform --out uniform_traj.json
python traj_to_csv.py --traj ladder_traj.json \
    --out ../figures/fig2_transfer_across_modalities/data/selfplay_frontier_perk.csv
python traj_to_csv.py --traj uniform_traj.json \
    --out ../figures/fig2_transfer_across_modalities/data/uniform_frontier_perk.csv
cd ../figures/fig2_transfer_across_modalities && python fig2.py
cd ../fig7_scaling_all_datasets && python fig7.py
```

**Figure 3**: curriculum value. Panel (b) rescores the curriculum learners
(`--group curriculum --rounds 4095`, see the folder README); panel (a) reads
per-round training loss from `data/training_losses.npz`, which is training
telemetry rather than a checkpoint quantity and is therefore shipped with the
figure.

**Figures 1 (in-context panel), 4 and 5**: in-context learning. Build the
ensemble directory from the released weights, then run the harness:

```bash
cd scoring && python fetch_ensemble.py --out ../figures/fig5_icl_sum_behavior/ensemble
cd ../figures/fig5_icl_sum_behavior
export SOLOMONOFF_REPO=../../scoring ICL_ENSEMBLE=$PWD/ensemble
python make_sum_behavior.py         # sum_behavior_p.npz, then python fig5.py for Figure 5
python icl_harness/run_icl.py       # and run_icl_v2.py ... run_icl_extras.py for Figure 1
```

Figure 4 compares that harness output against the same harness run on the
universal-prior and PCFG learners; its universal-prior arm uses
`baselines/uniform-prior/24M/`.
Redraw it with `python fig4.py` in `figures/fig4_icl_across_methods`.

**Table 5**: reward ablations. Score `--group reward`, assemble with
`build_traj_json.py --min-seeds 1`, then `python make_table.py` in that folder.

**Figure 6**: pre-pretraining. This figure summarises downstream pretraining
runs rather than checkpoint evaluations, so the plotted curves are shipped as
`fig6_data_cache.json` and the figure regenerates with `python fig6.py`.

**Tables 1 and 3**: discovered mathematical structure. `detect.py` is the
family detector, `hits.jsonl` the scan of the self-play generators' output
tapes (which reads training-time program logs rather than the released
checkpoints), and `prior_analysis.py` derives every number in the table from
`hits.jsonl` and the uniform-prior sample counts in `prior_counts.json`.
