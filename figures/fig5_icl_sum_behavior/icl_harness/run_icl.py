"""In-context-learning probe: (sentinel, input, output) x m, complete the last output.

Sequence = 'O' prefix, then m demonstrations [0, x_1..x_k, f(x)], then a query
[0, x_1..x_k]; the model must produce f(x) as the next byte. Inputs are uniform
bytes in 1..255 (0 is reserved as the sentinel); outputs are single bytes.

Functions: max, min, sum mod 256, floor-mean, first, last.
  first/last  = pure positional copying (induction-head territory)
  max/min     = comparison across the input
  sum/mean    = arithmetic aggregation

For each (f, m, k): TRIALS random instances, one batched forward through the
4-seed ensemble (probabilities averaged), scored as greedy exact-match accuracy
and mean probability assigned to the correct byte. Chance ~ 1/256.
Writes icl_results.json.
"""
import json, os, sys
from pathlib import Path
import numpy as np
import torch

# Directory providing src.framework.model (the `scoring/` folder of this
# repository). Override with SOLOMONOFF_REPO.
_REPO = os.environ.get("SOLOMONOFF_REPO",
                       ".")
sys.path.insert(0, _REPO)
from src.framework.model import ProgramLanguageModel

# Ensemble checkpoints (see scoring/fetch_ensemble.py). Override with ICL_ENSEMBLE.
_default_ens = Path(__file__).resolve().parent.parent / "ensemble"
if not _default_ens.is_dir():
    _default_ens = Path("./ensemble")
ENS = Path(os.environ.get("ICL_ENSEMBLE", _default_ens))
if not (ENS / "config.json").is_file():
    raise SystemExit(f"ensemble not found at {ENS}; set ICL_ENSEMBLE")
OUT = Path(__file__).parent / "icl_results.json"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
KS = [2, 4, 8, 16]
TRIALS = 64
FUNCS = {
    "max":  lambda x: int(np.max(x)),
    "min":  lambda x: int(np.min(x)),
    "sum":  lambda x: int(np.sum(x) % 256),
    "mean": lambda x: int(np.mean(x)),          # floor
    "first": lambda x: int(x[0]),
    "last": lambda x: int(x[-1]),
}

cfg = json.load(open(ENS / "config.json")); cfg = cfg.get("learner", cfg)
models = []
for d in sorted(ENS.glob("seed-*")):
    m = ProgramLanguageModel(**cfg)
    blob = torch.load(d / "learner_2816.pth", map_location="cpu", weights_only=False)
    m.load_state_dict(blob.get("learner_state_dict", blob), strict=False)
    for mod in m.modules():
        if hasattr(mod, "_kernel_ok"):
            mod._kernel_ok = False
    models.append(m.eval().to(DEV))
print(f"{len(models)} models on {DEV}", flush=True)

rng = np.random.default_rng(0)
results = {}
for fname, f in FUNCS.items():
    for m_ in MS:
        for k in KS:
            # sequence must fit the model's 4096 context:
            # 1 ('O') + m*(k+2) demos + (k+1) query
            if 1 + m_ * (k + 2) + k + 1 > 4096:
                continue
            seqs, targets = [], []
            for _ in range(TRIALS):
                toks = [ord("O")]
                for _ in range(m_):
                    x = rng.integers(1, 256, k)
                    toks += [0] + x.tolist() + [f(x)]
                xq = rng.integers(1, 256, k)
                toks += [0] + xq.tolist()
                seqs.append(toks)
                targets.append(f(xq))
            X = torch.tensor(np.asarray(seqs, dtype=np.int64), device=DEV)
            ps = []
            with torch.inference_mode():
                for mod in models:
                    logits, _ = mod(input_ids=X)
                    ps.append(torch.softmax(logits[:, -1].float(), dim=-1))
            p = torch.stack(ps).mean(0).cpu().numpy()
            tg = np.asarray(targets)
            acc = float((p.argmax(1) == tg).mean())
            pcorr = float(p[np.arange(TRIALS), tg].mean())
            results[f"{fname}|{m_}|{k}"] = {"acc": acc, "p_correct": pcorr}
            print(f"{fname:5} m={m_:<3} k={k:<3} acc={acc:.3f} p={pcorr:.4f}", flush=True)

json.dump({"model": "4-seed probability ensemble, d512h8L8 (24.3M non-emb), round 2816",
           "ms": MS, "ks": KS, "trials": TRIALS, "rng_seed": 0,
           "format": "'O', then m x [0, x_1..x_k, f(x)], then [0, x_1..x_k] -> predict f(x)",
           "results": results}, open(OUT, "w"), indent=1)
print("wrote", OUT, flush=True)
