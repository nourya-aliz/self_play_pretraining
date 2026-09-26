#!/usr/bin/env python3
"""Assemble the cache slices written by `score_from_hf.py` into one trajectory JSON.

    python build_traj_json.py --cache cache --out frontier_traj.json

Output schema, consumed by the exponent fits, the frontier figures, and the
reward-ablation table:

    {target: {"N": int, "rounds": {round: {"n_seeds": int, corpus: {K: bpb}}}}}

plus a top-level "provenance" block that records how the slices were scored
(corpus hashes, sequence counts, checkpoint source), so the file states where
its numbers came from without the cache being present.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, default=Path("cache"),
                    help="directory of <target>/round_*.json slices")
    ap.add_argument("--out", type=Path, default=Path("frontier_traj.json"))
    ap.add_argument("--min-seeds", type=int, default=3,
                    help="drop slices scored from fewer seeds than this")
    args = ap.parse_args()

    slices = sorted(args.cache.glob("*/round_*.json"))
    if not slices:
        raise SystemExit(f"no cache slices under {args.cache}; run score_from_hf.py first")

    out: dict[str, dict] = {}
    corpora_prov: dict[str, dict] = {}
    targets: dict[str, dict] = {}
    sources, scored = set(), []
    n_seq_seen, draws_seen = set(), set()
    corpus_rounds: dict[str, int] = defaultdict(int)

    for p in slices:
        rec = json.loads(p.read_text())
        if rec["n_seeds"] < args.min_seeds:
            continue
        prov = rec.get("provenance", {})
        target = rec["rung"]
        entry = out.setdefault(target, {"N": rec["N"], "rounds": {}})
        row = {"n_seeds": rec["n_seeds"]}
        row.update(rec["bpb"])
        entry["rounds"][str(rec["round"])] = row

        for name, cp in prov.get("corpora", {}).items():
            corpora_prov.setdefault(name, cp)
        for c in rec["bpb"]:
            corpus_rounds[c] += 1
        t = targets.setdefault(target, {"N": rec["N"], "rounds": 0,
                                        "max_seeds": 0, "max_round": 0})
        t["rounds"] += 1
        t["max_seeds"] = max(t["max_seeds"], rec["n_seeds"])
        t["max_round"] = max(t["max_round"], rec["round"])
        sources.add(prov.get("source", ""))
        scored.append(prov.get("scored_at", ""))
        n_seq_seen.add(prov.get("n_seq"))
        draws_seen.add(prov.get("bootstrap_draws"))

    if len(n_seq_seen) > 1:
        raise SystemExit(f"slices disagree on n_seq: {sorted(n_seq_seen)}; "
                         "re-score with a single --n-seq")

    payload = dict(out)
    payload["provenance"] = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": sorted(s for s in sources if s),
        "n_slices": len(slices),
        "min_seeds": args.min_seeds,
        "n_seq": sorted(n_seq_seen)[0],
        "bootstrap_draws": (sorted(draws_seen)[0] if len(draws_seen) == 1
                            else sorted(draws_seen)),
        "targets": targets,
        "corpora": corpora_prov,
    }
    if any(scored):
        payload["provenance"]["scored_between"] = [min(t for t in scored if t),
                                                   max(t for t in scored if t)]
    args.out.write_text(json.dumps(payload, indent=1))

    print(f"wrote {args.out}")
    print(f"\n{'target':11} {'N':>11} {'rounds':>7} {'seeds':>6} {'max round':>10}")
    for target, e in out.items():
        rr = [int(r) for r in e["rounds"]]
        mk = max(v["n_seeds"] for v in e["rounds"].values())
        print(f"{target:11} {e['N']:>11,} {len(rr):>7} {mk:>6} {max(rr):>10}")
    print(f"\n{len(corpus_rounds)} corpora:")
    for c, n in sorted(corpus_rounds.items()):
        print(f"  {c:28} {n:>4} slices")


if __name__ == "__main__":
    main()
