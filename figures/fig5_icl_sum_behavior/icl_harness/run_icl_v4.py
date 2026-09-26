"""ICL v4: palindrome (reversal) and stack simulation.

palin  demos [0, x_1..x_k, x_k..x_1]; query shows x_1..x_k and the model
       produces the reversal. Scored TEACHER-FORCED: one forward over the query
       with the true reversal appended, accuracy read at each output position
       (later positions therefore condition on correct earlier ones). Reversal
       = copying with a position-DEPENDENT lag: position j needs the token
       2(k-j)+1 back, a different offset per slot; the probe for whether the
       copying machinery generalizes beyond a single fixed offset.

stack  demos are complete traces of a random valid stack program:
       PUSH=250 v (values 1..249), POP=251 a; in demos every pop is followed
       by its answer; the query trace ends right after a final 251 and the
       model predicts the popped value. Interleaved push/pop, so the source of
       the answer depends on the full history (LIFO state tracking).
       Also reports accuracy split by the popped element's depth of burial.

128 trials/cell. Writes icl_v4_results.json.
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
OUT = Path(__file__).parent / "icl_v4_results.json"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TRIALS = 128
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


def batch_probs_full(seqs):
    """Ensemble mean softmax at EVERY position: [B, T, vocab]."""
    X = torch.tensor(np.asarray(seqs, dtype=np.int64), device=DEV)
    ps = []
    with torch.inference_mode():
        for mod in models:
            logits, _ = mod(input_ids=X)
            ps.append(torch.softmax(logits.float(), dim=-1))
    return torch.stack(ps).mean(0).cpu().numpy()


rng = np.random.default_rng(5)
results = {}

# --------------------------------------------------------------- palindrome
for k in (2, 4, 8, 16):
    for m_ in (1, 2, 4, 8, 32, 128):
        unit = 2 * k + 1
        if 1 + (m_ + 1) * unit > 4096:
            continue
        seqs, tgt = [], []
        for _ in range(TRIALS):
            toks = [ord("O")]
            for _ in range(m_):
                x = rng.integers(1, 256, k)
                toks += [0] + x.tolist() + x[::-1].tolist()
            xq = rng.integers(1, 256, k)
            toks += [0] + xq.tolist() + xq[::-1].tolist()   # teacher-forced tail
            seqs.append(toks); tgt.append(xq[::-1])
        p = batch_probs_full(seqs)
        T = len(seqs[0])
        pos_acc = []
        for j in range(k):                                   # output slot j
            col = T - k + j                                  # token index of slot j
            pred = p[:, col - 1, :].argmax(1)                # predicts token AT col
            pos_acc.append(float((pred == [t[j] for t in tgt]).mean()))
        results[f"palin|m={m_}|k={k}"] = dict(acc_mean=float(np.mean(pos_acc)),
                                              acc_by_position=pos_acc)
        print(f"palin m={m_:<4} k={k:<3} mean={np.mean(pos_acc):.3f}  "
              f"by_pos={[f'{a:.2f}' for a in pos_acc]}", flush=True)

# -------------------------------------------------------------------- stack
def trace(L):
    """Random valid stack program of L ops ending in POP; returns tokens (with
    pop answers), final answer, and its burial depth (1 = top just below...)."""
    while True:
        toks, stack, hist = [], [], []
        depth_at_final = None
        for i in range(L - 1):
            if stack and rng.random() < 0.5:
                v = stack.pop()
                toks += [POP, v]
            else:
                v = int(rng.integers(1, 250))
                stack.append(v)
                toks += [PUSH, v]
        if stack:
            ans = stack[-1]
            depth = len(stack)
            toks += [POP]
            return toks, ans, depth


for L in (4, 8, 16):
    for m_ in (1, 2, 4, 8, 32, 128):
        unit_max = 1 + 2 * L
        if 1 + (m_ + 1) * unit_max > 4096:
            continue
        seqs, tgt, dep = [], [], []
        for _ in range(TRIALS):
            toks = [ord("O")]
            for _ in range(m_):
                tr, a, _ = trace(L)
                toks += [0] + tr + [a]                       # demo pops all answered
            tr, a, d = trace(L)
            toks += [0] + tr                                  # ends right after POP
            seqs.append(toks); tgt.append(a); dep.append(d)
        # ragged lengths (traces vary): padding would put tokens before the
        # 'O' prefix (off-distribution), so batch by exact length instead
        pred = np.zeros(TRIALS, dtype=int)
        by_len = {}
        for i, s_ in enumerate(seqs):
            by_len.setdefault(len(s_), []).append(i)
        for idx in by_len.values():
            p = batch_probs_full([seqs[i] for i in idx])
            pr = p[:, -1, :].argmax(1)
            for ii, i in enumerate(idx):
                pred[i] = int(pr[ii])
        tg = np.asarray(tgt)
        acc = float((pred == tg).mean())
        by_depth = {}
        for dd in sorted(set(dep)):
            m = [i for i in range(TRIALS) if dep[i] == dd]
            by_depth[str(dd)] = [float((pred[m] == tg[m]).mean()), len(m)]
        results[f"stack|m={m_}|L={L}"] = dict(acc=acc, acc_by_burial_depth=by_depth)
        print(f"stack m={m_:<4} L={L:<3} acc={acc:.3f}  by_depth="
              + " ".join(f"{k}:{v[0]:.2f}(n{v[1]})" for k, v in by_depth.items()),
              flush=True)

json.dump({"model": "4-seed ensemble d512h8L8 r2816", "trials": TRIALS,
           "rng_seed": 5, "push_token": PUSH, "pop_token": POP,
           "results": results}, open(OUT, "w"), indent=1)
print("wrote", OUT, flush=True)
