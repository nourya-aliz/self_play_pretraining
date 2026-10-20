# Scoring: from checkpoints to results

Recomputes every checkpoint-derived quantity in the paper from the released
weights: the scaling frontiers, the compute exponents, the reward-ablation
table, and the curriculum validation scores.

## Requirements

`torch`, `numpy`, `huggingface_hub`, `jvp_flash_attention` (CUDA sm_75 or
newer), plus `requests`, `zstandard`, and `mido` for the corpus bake scripts.
`Cython` is optional and speeds up program execution.

## Pipeline

1. Bake the evaluation corpora from their public sources (CPU):

   ```bash
   python -m src.scripts.prepare_dclm_benchmark --context-length 4096 \
       --output data/c4096/dclm.jsonl --overwrite
   ```

   One `src/scripts/prepare_*_benchmark.py` script per benchmark family.

2. Score checkpoints from Hugging Face (GPU recommended):

   ```bash
   python score_from_hf.py --group ladder --targets 100k,500k,1M,3M,6M,24M \
       --rounds all --corpora dclm,cifar10_rgb_hwc,mutopia_melody_16th \
       --data-dir data/c4096 --n-seq 256 --out cache
   ```

   Groups: `ladder` (self-play scaling ladder), `uniform` (fixed-prior
   baseline), `reward` (Table 5 arms), `curriculum` (Figure 3 learners).
   Output: one JSON per (target, round) with per-seed and per-K ensemble
   bits/byte; `--save-probs` also keeps every seed's per-token probabilities.

3. Assemble and analyze:

   ```bash
   python build_traj_json.py --cache cache --out frontier_traj.json
   python traj_to_csv.py --traj frontier_traj.json \
       --out ../figures/fig2_transfer_across_modalities/data/selfplay_frontier_perk.csv
   ```

   then run the fitting and plotting scripts in the figure folders. Each
   folder's README gives the exact `score_from_hf.py` invocation behind it.

4. For the in-context learning experiments (Figure 1's in-context panel and
   Figures 4 and 5), assemble the
   four-seed 24.4M ensemble (`24M/` on Hugging Face) at round 2816:

   ```bash
   python fetch_ensemble.py --out ../figures/fig5_icl_sum_behavior/ensemble
   ```

5. To sample programs from the generators and run them:

   ```bash
   python sample_programs.py --size 1M --seed 40354564 \
       --rounds 256:4096:256 --n 65536 --out programs.npz
   ```

   Programs are sampled as in training and run on the machine described in
   Appendix E (`src/framework/machine.py`), with each model size's training
   limits. The output holds the programs and the learner's training rows.
   Execution uses a compiled kernel when Cython and a C compiler are
   available and a slower pure-Python path otherwise; both give identical
   results.

## Conventions

Input is the byte `O` prefix followed by 4095 corpus bytes. The score is the
probability of each actual next byte. Ensembles average seed probabilities
before taking the negative log (base 2). Per-K seed subsets are enumerated
exactly when C(S, K) <= 32 and otherwise use 32 bootstrap draws from an RNG
seeded by sha256 of `target:round:0`.

`src/` holds the model definition, the program machine, the evaluation code,
and the benchmark bake scripts. Training code is not part of this release.
