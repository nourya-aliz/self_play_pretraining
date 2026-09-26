# Figure 1: Compute frontiers and in-context learning

The data panels of Figure 1: per-modality compute frontiers (left) and
in-context learning accuracy (right). The schematic at the top of the
published figure is drawn separately.

## Contents

| File | Description |
|---|---|
| `fig1_bc.pdf`, `fig1_bc.png` | The two data panels |
| `fig1.py` | Draws them |
| `scaling_analysis.py` | Frontier construction and fits, shared with `../table2_scaling_exponents/` |
| `frontier_traj_perk.json` | Scored ladder trajectory: ensemble bits/byte per rung, round, corpus and ensemble size K (same file as in `../table2_scaling_exponents/`) |
| `icl_all_data.json` | In-context learning results for the self-play ensemble |
| `icl_m0_results.json`, `icl_sum_lowm_results.json`, `icl_assoc_dict_results.json` | The m = 0 cells, 4096-trial sum cells for m <= 4, and associative recall indexed by examples beyond one dictionary printing |

## Usage

```bash
python fig1.py
```

The legend exponents come from the same fit as Table 2, and
`../table2_scaling_exponents/check_fig1_table_consistency.py` asserts that the
two agree. The in-context results are produced by the harness in
`../fig5_icl_sum_behavior/icl_harness/` and are the same cells as the
self-play panel of Figure 4.
