"""Sentinel ablation for the ICL probe.

Formats:
  zero  : [0, x_1..x_k, f(x)] per example (the baseline format)
  none  : [x_1..x_k, f(x)]: no delimiter; segmentation only from periodicity
  s255  : [255, x_1..x_k, f(x)]: same structure, different sentinel value
          (inputs are drawn 1..254 here so the sentinel stays unambiguous)

Query ends right after the input (after the sentinel-prefixed input for
zero/s255, after the bare input for none); the model predicts f(x) next.
Functions first/last/max/sum at k=2, m in {8, 32, 128, 512}; 128 trials.
Writes icl_sentinel_results.json.
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
OUT = Path(__file__).parent / "icl_sentinel_results.json"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TRIALS, K = 128, 2
MS = [8, 32, 128, 512]
FUNCS = {"first": lambda x: int(x[0]), "last": lambda x: int(x[-1]),
         "max": lambda x: int(np.max(x)), "sum": lambda x: int(np.sum(x) % 256)}
FORMATS = {"zero": 0, "s255": 255, "none": None}

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


rng = np.random.default_rng(2)
results = {}
for fmt, sent in FORMATS.items():
    lo, hi = (1, 255) if fmt != "s255" else (1, 254)      # keep 255 unambiguous
    for fname, f in FUNCS.items():
        for m_ in MS:
            unit = K + 1 + (1 if sent is not None else 0)
            if 1 + m_ * unit + unit - 1 > 4096:
                continue
            seqs, targets = [], []
            for _ in range(TRIALS):
                toks = [ord("O")]
                for _ in range(m_):
                    x = rng.integers(lo, hi + 1, K)
                    toks += ([sent] if sent is not None else []) + x.tolist() + [f(x)]
                xq = rng.integers(lo, hi + 1, K)
                toks += ([sent] if sent is not None else []) + xq.tolist()
                seqs.append(toks)
                targets.append(f(xq))
            p = batch_probs(seqs)
            tg = np.asarray(targets)
            acc = float((p.argmax(1) == tg).mean())
            results[f"{fmt}|{fname}|{m_}"] = dict(
                acc=acc, p_correct=float(p[np.arange(TRIALS), tg].mean()))
            print(f"{fmt:5} {fname:5} m={m_:<4} acc={acc:.3f}", flush=True)

json.dump({"model": "4-seed ensemble d512h8L8 r2816", "k": K, "trials": TRIALS,
           "rng_seed": 2, "results": results}, open(OUT, "w"), indent=1)
print("wrote", OUT, flush=True)
