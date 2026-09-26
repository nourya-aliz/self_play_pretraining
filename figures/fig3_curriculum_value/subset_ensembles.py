"""Recompute data/subset_ensembles.csv from per-token probabilities.

Expects data/ood_validation/<arm>/round_004095_probs.npz for every arm, as
written by `scoring/score_from_hf.py --save-probs` (keys '<corpus>/<seed>',
arrays [n_seq, 4095]). For each arm and corpus, every 4-seed subset of the 8
seeds is ensembled by averaging probabilities, scored as mean -log2, and the
mean, SD, and median over the 70 subsets are written.

    python subset_ensembles.py
"""
import json
import os
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = f'{HERE}/data'
N_SEEDS, SUBSETS, ROUND = 8, 4, 4095
ARMS = ['G0', 'T256', 'T512', 'T1024', 'T2048', 'T4096']
CORPORA = ['dclm', 'audio_16bit', 'cifar10_rgb_planar']

rows = []
for arm in ARMS:
    z = np.load(f'{DATA}/ood_validation/{arm}/round_{ROUND:06d}_probs.npz')
    ref = json.load(open(f'{DATA}/ood_validation/{arm}/round_{ROUND:06d}.json'))
    seeds = sorted(ref['bpb_per_seed'][CORPORA[0]])
    assert len(seeds) == N_SEEDS, (arm, seeds)
    cells = []
    for c in CORPORA:
        P = np.stack([z[f'{c}/{s}'] for s in seeds])           # [S, n_seq, 4095]
        vals = [float(-np.log2(np.clip(P[list(ix)].mean(0), 1e-30, None)).mean())
                for ix in combinations(range(N_SEEDS), SUBSETS)]
        full = float(-np.log2(np.clip(P.mean(0), 1e-30, None)).mean())
        assert abs(full - ref['bpb'][c][str(N_SEEDS)]) < 5e-3, (arm, c, full)
        cells += [np.mean(vals), np.std(vals, ddof=1), np.median(vals)]
    rows.append(('G_0' if arm == 'G0' else arm, cells))

with open(f'{DATA}/subset_ensembles.csv', 'w') as f:
    f.write('arm,' + ','.join(f'{c}_mean,{c}_sd,{c}_median' for c in CORPORA) + '\n')
    for arm, cells in rows:
        f.write(arm + ',' + ','.join(f'{v:.4f}' for v in cells) + '\n')
print('wrote', f'{DATA}/subset_ensembles.csv')
