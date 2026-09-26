#!/usr/bin/env python3
"""Score checkpoints from the Hugging Face repo on the baked eval corpora.

This is the public re-scoring path for every checkpoint-derived number in the
paper: it downloads learner weights from nourya-cohen/solomonoff-paper,
evaluates next-byte probabilities on the baked benchmark corpora, and writes
per-(target, round) cache slices in the exact schema consumed by
`build_traj_json.py` (and hence by the frontier/exponent/reward analyses):

    {rung, size, N, round, n_seeds,
     bpb: {corpus: {K: bits/byte of the K-seed probability ensemble}},
     bpb_per_seed: {corpus: {seed: bits/byte}},
     provenance: {...}}

Metric conventions match the paper's scorers exactly: input = byte 'O' prefix
+ 4095 corpus bytes; per-token probability of the actual next byte via
log-softmax gather; ensembles average seed PROBABILITIES then take -log2;
per-K subsets are enumerated exactly when C(S, K) <= 32, else 32 bootstrap
draws from an RNG seeded by sha256(f"{target}:{round}:0").

Checkpoint groups (paths inside the HF repo):
  ladder     : <size>/seed-*/learner_*.pth          (self-play scaling ladder)
  uniform    : baselines/uniform-prior/<size>/...    (fixed-prior baseline)
  reward     : ablations/reward-arms/<arm>/...       (Table 5 arms)
  curriculum : curriculum/<arm>_seed-*/...           (Figure 3 learners;
               arms T256 ... T4096 and G0)

Usage:
    python score_from_hf.py --group ladder --targets 1M --rounds 8191 \
        --corpora dclm --data-dir data/c4096 --n-seq 256 --out cache
`--targets all` scores every target of the group and `--corpora all` every
`<stem>.jsonl` in the data directory. Corpora are baked with
src/scripts/prepare_*_benchmark.py at context 4096.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from src.framework.model import ProgramLanguageModel  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download     # noqa: E402

REPO = "nourya-cohen/solomonoff-paper"
LN2 = float(np.log(2.0))
BOOTSTRAP_DRAWS = 32
RNG_SEED = 0

# Non-embedding parameter counts (compute axis) and fallback architectures.
ARCH = {
    "100k": dict(d_model=64,  n_heads=1, n_layers=1),
    "500k": dict(d_model=128, n_heads=2, n_layers=2),
    "1M":   dict(d_model=128, n_heads=2, n_layers=4),
    "3M":   dict(d_model=256, n_heads=4, n_layers=4),
    "6M":   dict(d_model=256, n_heads=4, n_layers=8),
    "24M":  dict(d_model=512, n_heads=8, n_layers=8),
}
COMMON = dict(max_len=4096, vocab_size=256, base_d_model=16)
N_NONEMB = {"100k": 65_728, "500k": 492_160, "1M": 984_192,
            "3M": 3_016_960, "6M": 6_033_664, "24M": 24_253_184}
# architecture labels used as rung keys in the shipped trajectory files
ARCH_NAME = {s: f"d{a['d_model']}h{a['n_heads']}L{a['n_layers']}" for s, a in ARCH.items()}

GROUP_PREFIX = {
    "ladder": "{t}",
    "uniform": "baselines/uniform-prior/{t}",
    "reward": "ablations/reward-arms/{t}",
    "curriculum": "curriculum/{t}",
}
# reward arms and curriculum learners are all the 1M architecture
FIXED_ARCH_GROUPS = {"reward": "1M", "curriculum": "1M"}

CKPT_RE = re.compile(r"^(?P<prefix>.+)/(?P<seed>[^/]*seed-\d+[^/]*)/learner_(?P<round>\d+)\.pth$")


def target_of(group: str, prefix: str, seed_dir: str):
    """Target name a checkpoint belongs to, or None if outside the group."""
    if group == "curriculum":
        # curriculum/<arm>_seed-<seed>/learner_*.pth: prefix is "curriculum"
        return seed_dir.split("_seed-")[0] if prefix == "curriculum" else None
    m = re.match("^" + GROUP_PREFIX[group].format(t=r"([^/]+)") + "$", prefix)
    return m.group(1) if m else None


def index_group(api: HfApi, group: str, targets: list[str] | None):
    """{target: {round: {seed_dir: hf_path}}}; targets=None means all."""
    out: dict = defaultdict(lambda: defaultdict(dict))
    for f in api.list_repo_files(REPO):
        m = CKPT_RE.match(f)
        if not m:
            continue
        t = target_of(group, m.group("prefix"), m.group("seed"))
        if t is not None and (targets is None or t in targets):
            out[t][int(m.group("round"))][m.group("seed")] = f
    return out


def load_corpora(data_dir: Path, names: list[str], n_seq: int, device: str):
    corp, prov = {}, {}
    for name in names:
        path = data_dir / f"{name}.jsonl"
        rows = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line)["sequence"])
                if len(rows) >= n_seq:
                    break
        a = np.asarray(rows, dtype=np.int64)
        a = np.concatenate([np.full((len(a), 1), ord("O"), dtype=np.int64), a], axis=1)
        corp[name] = torch.from_numpy(a).to(device)
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        prov[name] = {"file": path.name, "sha256": h, "n_seq_used": len(rows)}
    return corp, prov


def model_arch(group: str, target: str, hf_ckpt_path: str) -> dict:
    """Per-seed config.json next to the checkpoint when present, else table."""
    cfg_path = str(Path(hf_ckpt_path).parent / "config.json")
    try:
        cfg = json.load(open(hf_hub_download(repo_id=REPO, filename=cfg_path)))
        return cfg.get("learner", cfg)
    except Exception:
        size = FIXED_ARCH_GROUPS.get(group, target)
        return {**ARCH[size], **COMMON}


def seed_probs(local_path: Path, arch: dict, corpora: dict, device: str, batch: int):
    """{corpus: [n_seq, 4095] float32} probability of each actual next byte."""
    model = ProgramLanguageModel(**arch)
    blob = torch.load(local_path, map_location="cpu", weights_only=False)
    sd = blob.get("learner_state_dict", blob.get("model_state_dict", blob))
    model.load_state_dict(sd, strict=False)
    del blob
    model.eval().to(device)
    for m in model.modules():
        if hasattr(m, "_kernel_ok"):
            m._kernel_ok = False
    out = {}
    with torch.inference_mode():
        for cname, data in corpora.items():
            ps = []
            for lo in range(0, len(data), batch):
                b = data[lo:lo + batch]
                logits = model(b[:, :-1])
                if isinstance(logits, tuple):
                    logits = logits[0]
                lp = torch.log_softmax(logits.float(), dim=-1)
                ps.append(lp.gather(-1, b[:, 1:].unsqueeze(-1))
                            .squeeze(-1).exp().cpu().numpy())
            out[cname] = np.concatenate(ps)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def per_k_bpb(arr: np.ndarray, rng: np.random.Generator) -> dict:
    S = arr.shape[0]
    per_k = {}
    for K in range(1, S + 1):
        subs = list(combinations(range(S), K))
        if len(subs) > BOOTSTRAP_DRAWS:
            subs = [tuple(rng.choice(S, K, replace=False))
                    for _ in range(BOOTSTRAP_DRAWS)]
        vals = [float((-np.log(np.clip(arr[list(ix)].mean(axis=0), 1e-30, None)))
                      .mean() / LN2) for ix in subs]
        per_k[str(K)] = float(np.mean(vals))
    return per_k


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--group", choices=sorted(GROUP_PREFIX), required=True)
    ap.add_argument("--targets", required=True,
                    help="comma list or 'all': sizes (ladder/uniform), arms "
                         "(reward), or arms such as T2048 or G0 (curriculum)")
    ap.add_argument("--rounds", default="all", help="'all' or comma list")
    ap.add_argument("--corpora", required=True,
                    help="comma list of baked corpus stems, or 'all'")
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="directory holding <stem>.jsonl baked at ctx 4096")
    ap.add_argument("--n-seq", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=HERE / "cache")
    ap.add_argument("--save-probs", action="store_true",
                    help="also write round_NNNNNN_probs.npz with the per-token "
                         "probabilities of every seed (keys '<corpus>/<seed>')")
    args = ap.parse_args()

    api = HfApi()
    targets = None if args.targets == "all" else args.targets.split(",")
    index = index_group(api, args.group, targets)
    if targets is None:
        targets = sorted(index)
    names = (sorted(p.stem for p in args.data_dir.glob("*.jsonl"))
             if args.corpora == "all" else args.corpora.split(","))
    corpora, corp_prov = load_corpora(args.data_dir, names, args.n_seq, args.device)

    for t in targets:
        rounds = sorted(index.get(t, {}))
        if args.rounds != "all":
            want = {int(r) for r in args.rounds.split(",")}
            rounds = [r for r in rounds if r in want]
        if not rounds:
            print(f"[warn] no checkpoints found for {args.group}/{t}")
            continue
        for rnd in rounds:
            rung = ARCH_NAME.get(t, t)        # 1M -> d128h2L4, as in the shipped data
            out_path = args.out / t / f"round_{rnd:06d}.json"
            if out_path.exists():
                print(f"[skip] {out_path}")
                continue
            seeds = index[t][rnd]
            probs, seed_names = [], []
            arch = None
            for seed_dir, hf_path in sorted(seeds.items()):
                local = Path(hf_hub_download(repo_id=REPO, filename=hf_path))
                arch = arch or model_arch(args.group, t, hf_path)
                probs.append(seed_probs(local, arch, corpora,
                                        args.device, args.batch_size))
                seed_names.append(seed_dir)
            rng_seed = int(hashlib.sha256(
                f"{t}:{rnd}:{RNG_SEED}".encode()).hexdigest()[:16], 16)
            bpb, per_seed = {}, {}
            for cname in corpora:
                arr = np.stack([p[cname] for p in probs])
                rng = np.random.default_rng(rng_seed)
                bpb[cname] = per_k_bpb(arr, rng)
                per_seed[cname] = {
                    s: float((-np.log(np.clip(arr[i], 1e-30, None))).mean() / LN2)
                    for i, s in enumerate(seed_names)}
            size_key = FIXED_ARCH_GROUPS.get(args.group, t)
            slice_ = {
                "rung": rung, "size": size_key, "N": N_NONEMB.get(size_key), "round": rnd,
                "n_seeds": len(seed_names), "bpb": bpb, "bpb_per_seed": per_seed,
                "provenance": {
                    "source": f"hf://{REPO}", "group": args.group,
                    "checkpoints": {s: seeds[s] for s in seed_names},
                    "corpora": corp_prov, "n_seq": args.n_seq,
                    "bootstrap_draws": BOOTSTRAP_DRAWS,
                    "scorer": "score_from_hf.py",
                    "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                },
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(slice_, indent=1))
            if args.save_probs:
                np.savez_compressed(
                    out_path.with_name(out_path.stem + "_probs.npz"),
                    **{f"{c}/{s}": p[c] for c in corpora for s, p in zip(seed_names, probs)})
            print(f"[done] {out_path} ({len(seed_names)} seeds)")


if __name__ == "__main__":
    main()
