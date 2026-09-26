"""Controls for the ICL result: is the model computing f(input), or input-blind?

DECOUPLED control: identical demos, but the target is f(hidden fresh x) while
the SHOWN query input is unrelated random bytes. Any input-independent strategy
(learned output marginal, demo-output statistics) keeps its accuracy under
decoupling; genuine computation on the shown input collapses to the marginal cap.

Also decomposes predictions on the NORMAL condition:
  == f(x) exact | == other query byte (support-restriction) | within +-4 | else
and reports mean |pred - true|, which separates "approximately computes f" from
"exactly computes f" (matters for mean).
Writes icl_control_results.json.
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
OUT = Path(__file__).parent / "icl_control_results.json"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TRIALS = 256
FUNCS = {"max": lambda x: int(np.max(x)), "min": lambda x: int(np.min(x)),
         "mean": lambda x: int(np.mean(x)), "sum": lambda x: int(np.sum(x) % 256)}
CELLS = [(f, m, 2) for f in FUNCS for m in (32, 128, 256, 512)] + \
        [("max", 128, 4), ("sum", 128, 4)]

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


def batch_probs(seqs):
    X = torch.tensor(np.asarray(seqs, dtype=np.int64), device=DEV)
    ps = []
    with torch.inference_mode():
        for mod in models:
            logits, _ = mod(input_ids=X)
            ps.append(torch.softmax(logits[:, -1].float(), dim=-1))
    return torch.stack(ps).mean(0).cpu().numpy()


rng = np.random.default_rng(1)                     # fresh seed vs the main sweep
results = {}
for fname, m_, k in CELLS:
    if 1 + m_ * (k + 2) + k + 1 > 4096:
        continue
    f = FUNCS[fname]
    normal, decoup, targets, queries = [], [], [], []
    for _ in range(TRIALS):
        demo = []
        for _ in range(m_):
            x = rng.integers(1, 256, k)
            demo += [0] + x.tolist() + [f(x)]
        xq = rng.integers(1, 256, k)               # real query
        xs = rng.integers(1, 256, k)               # shown-but-unrelated query
        normal.append([ord("O")] + demo + [0] + xq.tolist())
        decoup.append([ord("O")] + demo + [0] + xs.tolist())
        targets.append(f(xq))                      # decoupled target ignores xs
        queries.append(xq)
    pn, pd = batch_probs(normal), batch_probs(decoup)
    tg = np.asarray(targets); Q = np.asarray(queries)
    pred = pn.argmax(1)
    acc = float((pred == tg).mean())
    acc_dec = float((pd.argmax(1) == tg).mean())
    other = float(np.mean([(pred[i] in Q[i]) and pred[i] != tg[i]
                           for i in range(TRIALS)]))
    near = float((np.abs(pred.astype(int) - tg) <= 4).mean())
    mae = float(np.abs(pred.astype(int) - tg).mean())
    results[f"{fname}|{m_}|{k}"] = dict(acc=acc, acc_decoupled=acc_dec,
                                        pred_is_other_query_byte=other,
                                        acc_within_4=near, mae=mae)
    print(f"{fname:5} m={m_:<4} k={k}  acc={acc:.3f}  decoupled={acc_dec:.3f}  "
          f"otherbyte={other:.3f}  ±4={near:.3f}  mae={mae:.1f}", flush=True)

json.dump({"model": "4-seed ensemble d512h8L8 r2816", "trials": TRIALS,
           "rng_seed": 1, "results": results}, open(OUT, "w"), indent=1)
print("wrote", OUT, flush=True)
