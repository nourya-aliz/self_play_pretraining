# Table 2: Per-modality compute exponents

Compute-scaling exponents of the self-play ladder, fitted per corpus to the
compute-optimal frontier and compared against published exponents for direct
pretraining on the same modality.

## Contents

| File | Description |
|---|---|
| `exponents_table.tex` | The table |
| `make_latex_table.py` | Builds the frontiers, fits them, writes the table |
| `scaling_analysis.py` | Frontier construction and the fits; `../fig1_overview/fig1.py` imports the same module |
| `check_fig1_table_consistency.py` | Asserts the table's exponents equal Figure 1's legend exponents |
| `frontier_traj_perk.json` | Scored ladder trajectory: ensemble bits/byte per rung, round, corpus and ensemble size K |
| `literature_references.md` | Sources and derivations for the published exponents column |

## Usage

```bash
python make_latex_table.py
python check_fig1_table_consistency.py
```

The construction, per corpus: round 0 excluded; a Pareto front within each
rung over (round, K) with `C = K * N * 1536 * 4096 * (round + 1)`; a
cross-rung envelope over those; then `L(C) = E + A * C^-b` by nonlinear least
squares on raw bits/byte. The reported exponent is `b`.

To rebuild `frontier_traj_perk.json` from the released checkpoints:

```bash
cd ../../scoring
python score_from_hf.py --group ladder --targets all --rounds all \
    --corpora all --data-dir data/c4096 --n-seq 256 --out cache_ladder
python build_traj_json.py --cache cache_ladder \
    --out ../figures/table2_scaling_exponents/frontier_traj_perk.json
```
