"""ICL v3: tasks isolating specific mechanisms (see message for rationale).

  assoc    demos reuse a random key->value dictionary of size V; query a seen key.
  succ     y = (x+1) mod 256          (fixed rule, apply only)
  compl    y = 255 - x                (fixed rule, order reversal)
  double   y = 2x mod 256             (fixed rule, harder arithmetic)
  absdiff  y = |x1 - x2|              (subtraction without wraparound)
  closest  y = the x_j nearest to query byte q   (comparison -> retrieval)
  prev     y = the PREVIOUS example's input      (cross-example binding, k=1)

128 trials/cell. Writes icl_v3_results.json.
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
OUT = Path(__file__).parent / "icl_v3_results.json"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TRIALS = 128
MS = [8, 32, 128, 512]

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


results = {}


def score(tag, seqs, targets):
    p = batch_probs(seqs)
    tg = np.asarray(targets)
    acc = float((p.argmax(1) == tg).mean())
    results[tag] = dict(acc=acc, p_correct=float(p[np.arange(len(tg)), tg].mean()))
    print(f"{tag:22} acc={acc:.3f}  p={results[tag]['p_correct']:.4f}", flush=True)


rng = np.random.default_rng(4)

# assoc: dictionary of V pairs, shown as demos with reuse; query a SEEN key
for V in (4, 16, 64):
    for m_ in MS:
        if m_ < V or 1 + m_ * 3 + 2 > 4096:
            continue
        seqs, targets = [], []
        for _ in range(TRIALS):
            keys = rng.choice(np.arange(1, 256), V, replace=False)
            vals = rng.integers(1, 256, V)
            toks = [ord("O")]
            shown = set()
            for j in range(m_):
                i = int(rng.integers(0, V)) if j >= V else j   # guarantee coverage
                toks += [0, int(keys[i]), int(vals[i])]
                shown.add(i)
            iq = int(rng.choice(sorted(shown)))
            toks += [0, int(keys[iq])]
            seqs.append(toks); targets.append(int(vals[iq]))
        score(f"assoc|m={m_}|V={V}", seqs, targets)

# fixed unary maps, k=1
MAPS = {"succ": lambda x: (x + 1) % 256, "compl": lambda x: 255 - x,
        "double": lambda x: (2 * x) % 256}
for name, f in MAPS.items():
    for m_ in MS:
        seqs, targets = [], []
        for _ in range(TRIALS):
            toks = [ord("O")]
            for _ in range(m_):
                x = int(rng.integers(1, 256))
                toks += [0, x, f(x)]
            xq = int(rng.integers(1, 256))
            toks += [0, xq]
            seqs.append(toks); targets.append(f(xq))
        score(f"{name}|m={m_}|k=1", seqs, targets)

# absdiff, k=2
for m_ in MS:
    seqs, targets = [], []
    for _ in range(TRIALS):
        toks = [ord("O")]
        for _ in range(m_):
            a, b = (int(v) for v in rng.integers(1, 256, 2))
            toks += [0, a, b, abs(a - b)]
        a, b = (int(v) for v in rng.integers(1, 256, 2))
        toks += [0, a, b]
        seqs.append(toks); targets.append(abs(a - b))
    score(f"absdiff|m={m_}|k=2", seqs, targets)

# closest: k inputs then query byte q -> nearest input, k=4
for m_ in MS:
    k = 4
    if 1 + m_ * (k + 3) + k + 2 > 4096:
        continue
    seqs, targets = [], []
    for _ in range(TRIALS):
        toks = [ord("O")]
        def ex():
            x = rng.integers(1, 256, k)
            q = int(rng.integers(1, 256))
            y = int(x[np.argmin(np.abs(x.astype(int) - q))])
            return x, q, y
        for _ in range(m_):
            x, q, y = ex()
            toks += [0] + x.tolist() + [q, y]
        x, q, y = ex()
        toks += [0] + x.tolist() + [q]
        seqs.append(toks); targets.append(y)
    score(f"closest|m={m_}|k=4", seqs, targets)

# prev: y = previous example's input, k=1 (first demo's y is arbitrary noise)
for m_ in MS:
    seqs, targets = [], []
    for _ in range(TRIALS):
        toks = [ord("O")]
        prev = int(rng.integers(1, 256))
        toks += [0, prev, int(rng.integers(1, 256))]       # first y: no prev, noise
        for _ in range(m_ - 1):
            x = int(rng.integers(1, 256))
            toks += [0, x, prev]
            prev = x
        xq = int(rng.integers(1, 256))
        toks += [0, xq]
        seqs.append(toks); targets.append(prev)
    score(f"prev|m={m_}|k=1", seqs, targets)

json.dump({"model": "4-seed ensemble d512h8L8 r2816", "trials": TRIALS,
           "rng_seed": 4, "results": results}, open(OUT, "w"), indent=1)
print("wrote", OUT, flush=True)
