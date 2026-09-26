# Table 1: Discovered mathematical structure

A scan of every program the generator produced across the six ladder rungs
for five sequence families, compared against the discovery rate under uniform
program sampling.

## Contents

| File | Description |
|---|---|
| `math_table.tex` | The table |
| `detect.py` | Family detector, with a self-test on synthetic tapes (`python detect.py`) |
| `hits.jsonl` | Complete scan output: 20,045 hits with family, rung, seed, round, parameters, and generating program |
| `prior_counts.json` | Hit counts from 1.643e8 uniformly sampled programs passed through the same detector |
| `prior_analysis.py` | Derives the numeric columns of the table from `hits.jsonl` and `prior_counts.json` |
| `SUMMARY.md` | Per-rung, per-family counts and first observed rounds |
| `SHOWCASE_PROGRAMS.md` | The example programs shown in the table |

## Method

A tape matches a family if, after at most 30 leading bytes, the remainder
satisfies the family's recurrence modulo 256 (constant first, second, or third
finite difference; Fibonacci; geometric) with minimal period at least 30.
Programs are read from the program histories saved alongside the training
checkpoints every 256 rounds (training-time logs that are not part of this
release; the scan output is shipped in `hits.jsonl`), so a family is observed
at the first saved round at or after its actual first appearance.

The baseline samples programs i.i.d. uniformly over the instruction alphabet,
executes them with the same tape length, and applies the same detector. It
found 1,526 arithmetic hits (expected first-discovery round about 105 at 1,024
programs per round) and none of the other four families (rule of three:
p <= 1.8e-8, expected first round above 53,000). The sampler itself requires
the program executor and is not part of this release; its counts are shipped
in `prior_counts.json`.

```bash
python prior_analysis.py
```
