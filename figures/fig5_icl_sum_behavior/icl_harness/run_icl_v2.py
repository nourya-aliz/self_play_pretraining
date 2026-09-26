"""ICL v2: tasks with no trivial answer.

  index  [0, x_1..x_k, i, x_i]: query byte i in 1..k addresses the input;
         retrieval with a content-dependent address (first/last are the fixed-
         address special cases). Guessing any input byte gives 1/k.
  shift  [0, x, (x+c) mod 256], k=1; c is drawn once PER SEQUENCE; the rule
         itself must be inferred from the demos and applied to a fresh x.
         Copying and marginals are both useless.
  xor    [0, x1, x2, x1 XOR x2]
  diff   [0, x1, x2, (x1 - x2) mod 256]
  sum    [0, x1, x2, (x1 + x2) mod 256]   (rerun here so all share one rng)

128 trials/cell, greedy exact-match + p(correct) + pred-in-input fraction.
Writes icl_v2_results.json.
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
OUT = Path(__file__).parent / "icl_v2_results.json"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TRIALS = 128

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


def score(tag, seqs, targets, supports):
    p = batch_probs(seqs)
    tg = np.asarray(targets)
    pred = p.argmax(1)
    acc = float((pred == tg).mean())
    insup = float(np.mean([pred[i] in supports[i] for i in range(len(tg))]))
    res = dict(acc=acc, p_correct=float(p[np.arange(len(tg)), tg].mean()),
               pred_in_input=insup)
    results[tag] = res
    print(f"{tag:22} acc={acc:.3f}  p={res['p_correct']:.4f}  in_input={insup:.3f}",
          flush=True)


rng = np.random.default_rng(3)
results = {}

# ------------------------------------------------------------------- index
for k in (2, 4, 8, 16):
    for m_ in (4, 16, 64, 256):
        if 1 + m_ * (k + 3) + k + 2 > 4096:
            continue
        seqs, targets, sup = [], [], []
        for _ in range(TRIALS):
            toks = [ord("O")]
            for _ in range(m_):
                x = rng.integers(1, 256, k)
                i = int(rng.integers(1, k + 1))
                toks += [0] + x.tolist() + [i, int(x[i - 1])]
            xq = rng.integers(1, 256, k)
            iq = int(rng.integers(1, k + 1))
            toks += [0] + xq.tolist() + [iq]
            seqs.append(toks); targets.append(int(xq[iq - 1])); sup.append(set(xq))
        score(f"index|m={m_}|k={k}", seqs, targets, sup)

# ------------------------------------------------------------------- shift
for m_ in (2, 4, 8, 16, 32, 128, 512):
    seqs, targets, sup = [], [], []
    for _ in range(TRIALS):
        c = int(rng.integers(1, 256))                     # latent per-sequence rule
        toks = [ord("O")]
        for _ in range(m_):
            x = int(rng.integers(1, 256))
            toks += [0, x, (x + c) % 256]
        xq = int(rng.integers(1, 256))
        toks += [0, xq]
        seqs.append(toks); targets.append((xq + c) % 256); sup.append({xq})
    score(f"shift|m={m_}|k=1", seqs, targets, sup)

# ------------------------------------------- binary ops (same rng, comparable)
OPS = {"sum": lambda a, b: (a + b) % 256, "diff": lambda a, b: (a - b) % 256,
       "xor": lambda a, b: a ^ b}
for op, f in OPS.items():
    for m_ in (8, 32, 128, 512):
        seqs, targets, sup = [], [], []
        for _ in range(TRIALS):
            toks = [ord("O")]
            for _ in range(m_):
                a, b = (int(v) for v in rng.integers(1, 256, 2))
                toks += [0, a, b, f(a, b)]
            a, b = (int(v) for v in rng.integers(1, 256, 2))
            toks += [0, a, b]
            seqs.append(toks); targets.append(f(a, b)); sup.append({a, b})
        score(f"{op}|m={m_}|k=2", seqs, targets, sup)

json.dump({"model": "4-seed ensemble d512h8L8 r2816", "trials": TRIALS,
           "rng_seed": 3, "results": results}, open(OUT, "w"), indent=1)
print("wrote", OUT, flush=True)
