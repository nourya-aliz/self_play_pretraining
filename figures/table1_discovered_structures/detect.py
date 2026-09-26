"""Series detector for BF-machine output tapes (mod-256 cells).

Families: arithmetic / quadratic / cubic (constant 1st/2nd/3rd difference),
Fibonacci (x[i+2]=x[i+1]+x[i]), geometric (x[i+1]=r*x[i]).
A hit must: start at offset <= MAX_PREFIX, run through the end of the tape,
and have minimal period >= MIN_PERIOD (mod-256 observed sequence).
Any such mod-256 sequence lifts to an integer-valued unmodded sequence of the
same family, so the "arbitrarily large cells" condition is automatic.
"""
import json, sys
import numpy as np

MAX_PREFIX = 30
MIN_PERIOD = 30


def min_period(t: bytes):
    """Smallest p with t[i] == t[i+p] for all i (KMP failure function)."""
    n = len(t)
    pi = [0] * n
    k = 0
    for i in range(1, n):
        while k and t[i] != t[k]:
            k = pi[k - 1]
        if t[i] == t[k]:
            k += 1
        pi[i] = k
    return n - pi[-1]


def _extend_back(x, s0, ok_fn):
    """Smallest s <= s0 such that ok_fn holds on x[s:]."""
    s = s0
    while s > 0 and ok_fn(x, s - 1):
        s -= 1
    return s


def classify_record(x: np.ndarray):
    """x: uint8 1-D tape. Returns list of hit dicts."""
    n = len(x)
    if n < 200:
        return []
    core = x[MAX_PREFIX:].astype(np.int64)
    hits = []

    d1 = np.diff(core) % 256
    d2 = np.diff(d1) % 256
    d3 = np.diff(d2) % 256

    poly_order = None
    if (d1 == d1[0]).all():
        poly_order = 1
    elif (d2 == d2[0]).all():
        poly_order = 2
    elif (d3 == d3[0]).all():
        poly_order = 3

    if poly_order is not None:
        k = poly_order

        def ok(xx, s, k=k):
            seg = xx[s:].astype(np.int64)
            for _ in range(k):
                seg = np.diff(seg) % 256
            return (seg == seg[0]).all()

        s = _extend_back(x, MAX_PREFIX, ok)
        fam = {1: 'arithmetic', 2: 'quadratic', 3: 'cubic'}[k]
        hits.append((fam, s, {'diffs': [int(v) for v in
                    [d1[0], d2[0] if k >= 2 else None, d3[0] if k >= 3 else None] if v is not None]}))

    fib_ok = ((core[2:] - core[1:-1] - core[:-2]) % 256 == 0).all()
    if fib_ok:
        def okf(xx, s):
            seg = xx[s:].astype(np.int64)
            return ((seg[2:] - seg[1:-1] - seg[:-2]) % 256 == 0).all()
        s = _extend_back(x, MAX_PREFIX, okf)
        hits.append(('fibonacci', s, {}))

    # geometric: candidate r from an odd element, else brute-force 256 ratios
    cand = None
    odd_idx = np.nonzero(core[:-1] & 1)[0]
    if len(odd_idx):
        i = odd_idx[0]
        cand = [int((int(core[i + 1]) * pow(int(core[i]), -1, 256)) % 256)]
    else:
        cand = range(256)
    for r in cand:
        if ((core[1:] - r * core[:-1]) % 256 == 0).all():
            def okg(xx, s, r=r):
                seg = xx[s:].astype(np.int64)
                return ((seg[1:] - r * seg[:-1]) % 256 == 0).all()
            s = _extend_back(x, MAX_PREFIX, okg)
            hits.append(('geometric', s, {'ratio': int(r)}))
            break

    out = []
    for fam, s, params in hits:
        tail = bytes(x[s:].astype(np.uint8))
        p = min_period(tail)
        # eventual period over the trailing window: kills transient-then-cycle
        # junk (e.g. one nonzero byte then all zeros). Window 2048 >= 2x the
        # largest possible true period of these families mod 256 (<=1024).
        p_eff = min_period(tail[-2048:])
        if min(p, p_eff) < MIN_PERIOD:
            continue
        out.append({'family': fam, 'start_offset': int(s), 'period': p,
                    'params': params,
                    'first_terms': [int(v) for v in x[s:s + 12]],
                    'prefix': [int(v) for v in x[:s]],
                    'tape_len': int(n)})
    return out


def scan_file(path_or_obj, loc):
    """loc: dict with rung, seed, file. Yields hit dicts with exact location."""
    d = json.load(open(path_or_obj)) if isinstance(path_or_obj, str) else path_or_obj
    for idx, rec in enumerate(d.get('records', [])):
        o = rec.get('output')
        if not o:
            continue
        if isinstance(o, str):
            try:
                arr = np.array(json.loads(o), dtype=np.int64)
            except Exception:
                continue
        else:
            arr = np.array(o, dtype=np.int64)
        for h in classify_record(arr):
            h.update(loc)
            h['record_index'] = idx
            h['record_round'] = rec.get('round')
            h['program'] = rec.get('program')
            yield h


if __name__ == '__main__':
    # self-test on synthetic tapes
    tests = []
    ar = (np.arange(4095) * 10 + 7) % 256; tests.append(('arithmetic', ar))
    fib = [7, 7]
    for _ in range(4093): fib.append((fib[-1] + fib[-2]) % 256)
    tests.append(('fibonacci', np.array(fib)))
    ge = [3]
    for _ in range(4094): ge.append((ge[-1] * 3) % 256)
    tests.append(('geometric', np.array(ge)))
    qu = np.array([(2 * n * n + 3 * n + 1) % 256 for n in range(4095)]); tests.append(('quadratic', qu))
    cu = np.array([(n ** 3 + n + 5) % 256 for n in range(4095)]); tests.append(('cubic', cu))
    junk = np.concatenate([np.array([9, 88, 3, 250, 17] * 4), ar[:4075]]); tests.append(('arithmetic+prefix20', junk))
    const = np.zeros(4095, dtype=int); tests.append(('constant(reject)', const))
    rnd = np.random.RandomState(0).randint(0, 256, 4095); tests.append(('random(reject)', rnd))
    a128 = (np.arange(4095) * 128) % 256; tests.append(('arith d=128 period2 (reject)', a128))
    for name, t in tests:
        r = classify_record(t.astype(np.int64))
        print(name, '->', [(h['family'], h['start_offset'], h['period']) for h in r])
