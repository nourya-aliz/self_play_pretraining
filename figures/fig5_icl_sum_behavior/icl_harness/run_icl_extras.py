"""Additional in-context measurements used by the Figure 1 in-context panel:

  icl_m0_results.json         m = 0 (no demonstrations) points for
                              reverse-string (palin k=8), stack (L=4),
                              max and min (k=2).
  icl_sum_lowm_results.json   4096-trial sum cells for m = 0..4 (64 trials
                              cannot resolve accuracies near 1/256).
  icl_assoc_dict_results.json assoc (V=16) indexed by the number of examples
                              BEYOND one full printing of the dictionary:
                              the first 16 demos print each pair once, then
                              `extra` further random repeats, then the query.

Conventions identical to run_icl.py / run_icl_v3.py / run_icl_v4.py:
byte 'O' prefix, sentinel-0 examples, greedy decoding of the 4-seed
probability ensemble (d512h8L8, round 2816).
"""
import json, os, sys
from pathlib import Path
import numpy as np
import torch

_REPO = os.environ.get("SOLOMONOFF_REPO", ".")
sys.path.insert(0, _REPO)
from src.framework.model import ProgramLanguageModel

_default_ens = Path(__file__).resolve().parent.parent / "ensemble"
if not _default_ens.is_dir():
    _default_ens = Path("./ensemble")
ENS = Path(os.environ.get("ICL_ENSEMBLE", _default_ens))
if not (ENS / "config.json").is_file():
    raise SystemExit(f"ensemble not found at {ENS}; set ICL_ENSEMBLE")
OUTDIR = Path(os.environ.get("ICL_OUT", Path(__file__).parent))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
PUSH, POP = 250, 251

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


def batch_probs_full(seqs, chunk=256):
    """Ensemble mean softmax at EVERY position: [B, T, vocab]."""
    X = torch.tensor(np.asarray(seqs, dtype=np.int64), device=DEV)
    outs = []
    with torch.inference_mode():
        for lo in range(0, len(X), chunk):
            b = X[lo:lo + chunk]
            ps = []
            for mod in models:
                logits, _ = mod(input_ids=b)
                ps.append(torch.softmax(logits.float(), dim=-1))
            outs.append(torch.stack(ps).mean(0).cpu().numpy())
    return np.concatenate(outs)


def last_pos_acc(seqs, targets, chunk=256):
    """Greedy accuracy of the prediction at the final position."""
    X = torch.tensor(np.asarray(seqs, dtype=np.int64), device=DEV)
    preds = []
    with torch.inference_mode():
        for lo in range(0, len(X), chunk):
            b = X[lo:lo + chunk]
            ps = []
            for mod in models:
                logits, _ = mod(input_ids=b)
                ps.append(torch.softmax(logits[:, -1].float(), dim=-1))
            preds.append(torch.stack(ps).mean(0).argmax(1).cpu().numpy())
    pred = np.concatenate(preds)
    return float((pred == np.asarray(targets)).mean())


def stack_trace(rng, L):
    while True:
        toks, stack = [], []
        for _ in range(L - 1):
            if stack and rng.random() < 0.5:
                v = stack.pop()
                toks += [POP, v]
            else:
                v = int(rng.integers(1, 250))
                stack.append(v)
                toks += [PUSH, v]
        if stack:
            ans = stack[-1]
            toks += [POP]
            return toks, ans


# ------------------------------------------------------------- m = 0 points
rng = np.random.default_rng(0)
TRIALS0 = 1024
m0 = {}

# reverse string (palin), k=8, teacher-forced per position
k = 8
seqs, tgt = [], []
for _ in range(TRIALS0):
    xq = rng.integers(1, 256, k)
    seqs.append([ord("O"), 0] + xq.tolist() + xq[::-1].tolist())
    tgt.append(xq[::-1])
p = batch_probs_full(seqs)
T = len(seqs[0])
pos_acc = [float((p[:, T - k + j - 1, :].argmax(1) == [t[j] for t in tgt]).mean())
           for j in range(k)]
m0[f"palin|m=0|k={k}"] = dict(acc_mean=float(np.mean(pos_acc)),
                              acc_by_position=pos_acc)
print("palin m=0", m0[f"palin|m=0|k={k}"]["acc_mean"], flush=True)

# stack, L=4, query trace only (ragged: batch by exact length)
L = 4
seqs, tgt = [], []
for _ in range(TRIALS0):
    tr, a = stack_trace(rng, L)
    seqs.append([ord("O"), 0] + tr)
    tgt.append(a)
by_len = {}
for i, s_ in enumerate(seqs):
    by_len.setdefault(len(s_), []).append(i)
pred = np.zeros(TRIALS0, dtype=int)
for idx in by_len.values():
    p = batch_probs_full([seqs[i] for i in idx])
    pr = p[:, -1, :].argmax(1)
    for ii, i in enumerate(idx):
        pred[i] = int(pr[ii])
m0[f"stack|m=0|L={L}"] = dict(acc=float((pred == np.asarray(tgt)).mean()))
print("stack m=0", m0[f"stack|m=0|L={L}"]["acc"], flush=True)

# max / min, k=2
for fname, f in (("max", lambda x: int(x.max())), ("min", lambda x: int(x.min()))):
    seqs, targets = [], []
    for _ in range(TRIALS0):
        xq = rng.integers(1, 256, 2)
        seqs.append([ord("O"), 0] + xq.tolist())
        targets.append(f(xq))
    m0[f"{fname}|0|2"] = {"acc": last_pos_acc(seqs, targets)}
    print(fname, "m=0", m0[f"{fname}|0|2"]["acc"], flush=True)

json.dump({"model": "4-seed ensemble d512h8L8 r2816", "trials": TRIALS0,
           "rng_seed": 0, "results": m0},
          open(OUTDIR / "icl_m0_results.json", "w"), indent=1)

# ----------------------------------------------------- sum, m <= 4, 4096 trials
rng = np.random.default_rng(0)
TRIALS_SUM = 4096
lowm = {}
for m_ in (0, 1, 2, 3, 4):
    seqs, targets = [], []
    for _ in range(TRIALS_SUM):
        toks = [ord("O")]
        for _ in range(m_):
            x = rng.integers(1, 256, 2)
            toks += [0] + x.tolist() + [int(x.sum()) % 256]
        xq = rng.integers(1, 256, 2)
        toks += [0] + xq.tolist()
        seqs.append(toks)
        targets.append(int(xq.sum()) % 256)
    lowm[f"sum|{m_}|2"] = {"acc": last_pos_acc(seqs, targets)}
    print("sum m=", m_, lowm[f"sum|{m_}|2"]["acc"], flush=True)

json.dump({"model": "4-seed ensemble d512h8L8 r2816", "trials": TRIALS_SUM,
           "rng_seed": 0, "results": lowm},
          open(OUTDIR / "icl_sum_lowm_results.json", "w"), indent=1)

# ------------------------------------- assoc (V=16) vs examples beyond one printing
rng = np.random.default_rng(4)
TRIALS_A = 1024
V = 16
adict = {}
for extra in (0, 1, 2, 4, 8, 16):
    seqs, targets = [], []
    for _ in range(TRIALS_A):
        keys = rng.choice(np.arange(1, 256), V, replace=False)
        vals = rng.integers(1, 256, V)
        toks = [ord("O")]
        for j in range(V):                       # one full printing
            toks += [0, int(keys[j]), int(vals[j])]
        for _ in range(extra):                   # repeats beyond the printing
            i = int(rng.integers(0, V))
            toks += [0, int(keys[i]), int(vals[i])]
        iq = int(rng.integers(0, V))
        toks += [0, int(keys[iq])]
        seqs.append(toks)
        targets.append(int(vals[iq]))
    adict[f"assoc|extra={extra}|V={V}"] = {"acc": last_pos_acc(seqs, targets)}
    print("assoc extra=", extra, adict[f"assoc|extra={extra}|V={V}"]["acc"], flush=True)

json.dump({"model": "4-seed ensemble d512h8L8 r2816", "trials": TRIALS_A,
           "rng_seed": 4, "results": adict},
          open(OUTDIR / "icl_assoc_dict_results.json", "w"), indent=1)
print("done", flush=True)
