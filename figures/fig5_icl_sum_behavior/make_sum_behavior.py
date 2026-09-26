"""Evaluate the ensemble on the in-context SUM task and write `sum_behavior_p.npz`.

Each trial draws 512 demonstrations (x1, x2) and one query (q1, q2), all
uniform bytes in 1..255. The prompt at context length m is

    ['O'] + [0, x1, x2, (x1 + x2) mod 256] * m + [0, q1, q2]

and the recorded quantity is the ensemble's softmax over the next byte at the
query position (seed probabilities averaged), for every m in MS and every
trial. Trials are generated once from a fixed seed, so the context at length m
is a prefix of the context at any larger m.

Needs an ensemble directory (see `scoring/fetch_ensemble.py`) and the model
class from `scoring/`:

    export SOLOMONOFF_REPO=../../scoring
    export ICL_ENSEMBLE=./ensemble
    python make_sum_behavior.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.environ.get("SOLOMONOFF_REPO", "../../scoring"))
from src.framework.model import ProgramLanguageModel  # noqa: E402

HERE = Path(__file__).resolve().parent
ENS = Path(os.environ.get("ICL_ENSEMBLE", HERE / "ensemble"))
if not (ENS / "config.json").is_file():
    raise SystemExit(f"ensemble not found at {ENS}; set ICL_ENSEMBLE")
OUT = Path(os.environ.get("ICL_OUT", HERE)) / "sum_behavior_p.npz"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROUND = int(os.environ.get("ICL_ROUND", 2816))
MS = [0, 1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512]
TRIALS, RNG_SEED, BATCH = 1024, 23, 32

rng = np.random.default_rng(RNG_SEED)
demos = rng.integers(1, 256, (TRIALS, MS[-1], 2))
queries = rng.integers(1, 256, (TRIALS, 2))
dans = demos.sum(-1) % 256

cfg = json.load(open(ENS / "config.json"))
cfg = cfg.get("learner", cfg)
models = []
for d in sorted(ENS.glob("seed-*")):
    m = ProgramLanguageModel(**cfg)
    blob = torch.load(d / f"learner_{ROUND}.pth", map_location="cpu", weights_only=False)
    m.load_state_dict(blob.get("learner_state_dict", blob), strict=False)
    for mod in m.modules():
        if hasattr(mod, "_kernel_ok"):
            mod._kernel_ok = False
    models.append(m.eval().to(DEV))
print(f"{len(models)} models on {DEV}", flush=True)

P = np.zeros((len(MS), TRIALS, 256), dtype=np.float32)
for mi, m_ in enumerate(MS):
    seqs = []
    for t in range(TRIALS):
        toks = [ord("O")]
        for j in range(m_):
            toks += [0, int(demos[t, j, 0]), int(demos[t, j, 1]), int(dans[t, j])]
        toks += [0, int(queries[t, 0]), int(queries[t, 1])]
        seqs.append(toks)
    X = torch.tensor(np.asarray(seqs, dtype=np.int64))
    with torch.inference_mode():
        for lo in range(0, TRIALS, BATCH):
            xb = X[lo:lo + BATCH].to(DEV)
            ps = []
            for mod in models:
                logits, _ = mod(input_ids=xb)
                ps.append(torch.softmax(logits[:, -1].float(), dim=-1))
            P[mi, lo:lo + BATCH] = torch.stack(ps).mean(0).cpu().numpy()
    acc = float((P[mi].argmax(1) == (queries.sum(1) % 256)).mean())
    print(f"m={m_:<4} acc={acc:.3f}", flush=True)

np.savez_compressed(OUT, P=P, ms=np.array(MS), trials=TRIALS, rng_seed=RNG_SEED)
print("wrote", OUT)
