# Figure 5: In-context SUM task behavior

Behavior of the learner on the in-context SUM task as the number of
demonstrations m grows. Panel (a) classifies the argmax prediction at the
query position for each of the 1,024 trials and stacks the shares. Panel (b)
is the predictive uncertainty: the entropy of the next-byte distribution, mean
and interquartile range over trials.

The five categories of panel (a), in order of precedence:

| Category | Rule |
|---|---|
| Correct answer | argmax equals (q1 + q2) mod 256 |
| Preferred bytes | argmax is one of {0, 255, 1, 16}, the bytes the learner favours before any demonstration |
| Low 4 bits correct | argmax agrees with the answer in its low nibble |
| Context byte | argmax appears somewhere in the prompt |
| Other | anything else |

## Contents

| File | Description |
|---|---|
| `fig5.pdf`, `fig5.png` | The figure, both panels |
| `fig5.py` | Draws the figure from `sum_behavior_p.npz` and writes `sum_behavior_composition.csv` |
| `sum_behavior_composition.csv` | Category shares and entropy statistics per context length m |
| `sum_behavior_p.npz` | The ensemble's next-byte distribution at the query position: `P` [12, 1024, 256], context lengths `ms`, 1,024 trials, RNG seed 23 |
| `make_sum_behavior.py` | Recomputes `sum_behavior_p.npz` from the released checkpoints |
| `icl_harness/` | In-context learning evaluation harness and results |

The harness covers all ICL tasks in the paper: `run_icl.py` (max, min, sum,
mean, first, last), `run_icl_v2.py` (index, shift), `run_icl_v3.py`
(associative recall, arithmetic relations), `run_icl_v4.py` (reverse string,
stack), plus format and control ablations (`run_icl_sentinel.py`,
`run_icl_control.py`) and the extra cells used by Figure 1(b)
(`run_icl_extras.py`). Results are in `icl_*_results.json`, consolidated in
`icl_all_data.json`, and plotted by `icl_all_plot.py` (`icl_all.pdf`).

## Usage

```bash
python fig5.py
```

`fig5.py` regenerates the trial prompts from the RNG seed stored in
`sum_behavior_p.npz`, so the classification in panel (a) needs nothing beyond
that file.

Evaluation uses the 24.4M ladder checkpoints (`24M/` on Hugging Face) at self-play round 2816
(seeds 40354564 to 40354567) as a 4-seed probability ensemble. To rebuild
`sum_behavior_p.npz` or rerun the harness, assemble the ensemble directory from
Hugging Face and point the scripts at the model class in `scoring/`:

```bash
python ../../scoring/fetch_ensemble.py --out ./ensemble
export SOLOMONOFF_REPO=../../scoring     # provides src.framework.model
export ICL_ENSEMBLE=./ensemble           # config.json + seed-*/learner_2816.pth
python make_sum_behavior.py              # GPU recommended
python icl_harness/run_icl.py
```

Prompts follow the paper's format: the byte `O`, then for each demonstration a
sentinel byte 0, the k input bytes and the function value, then the query
(sentinel and inputs). The model's next-byte distribution is read at the query
position.
