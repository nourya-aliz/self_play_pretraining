"""Shared helpers for the benchmark bake scripts.

Conventions (see src/framework/evaluation.py):

* The single length knob is ``context_length`` (INCLUDES the prefix byte);
  every baked sequence carries exactly ``context_length - PREFIX_LEN`` data
  bytes: the eval window. No padding, no variable lengths.
* Bakes are context-stamped: ``<data_dir>/c{context_length}/<stem>.jsonl``.
* Raw source bytes are cached once under ``<data_dir>/raw_sources/`` so baking
  a NEW context is a deterministic, network-free re-chunk; a cache that is too
  small for the requested bake is re-fetched from scratch (never partially
  extended; stream fetches are not resumable).
* Every bake ends with ``validate_baked_file``; a script can never write a
  file the eval loader would reject.
"""

import json
from pathlib import Path
from typing import Callable, Iterable, Iterator

from src.framework.constants import PREFIX_LEN
from src.framework.evaluation import context_dir, validate_baked_file
from src.scripts.constants import DEFAULT_DATA_DIR

RAW_SOURCES_DIR = Path(DEFAULT_DATA_DIR) / 'raw_sources'


def bake_output_path(stem: str, context_length: int,
                     data_dir=DEFAULT_DATA_DIR) -> Path:
    """The canonical context-stamped output path for a corpus stem."""
    return context_dir(data_dir, context_length) / f"{stem}.jsonl"


def window_from_context(context_length: int) -> int:
    if context_length <= PREFIX_LEN:
        raise SystemExit(f"--context-length must be > PREFIX_LEN ({PREFIX_LEN}), "
                         f"got {context_length}")
    return context_length - PREFIX_LEN


def ensure_raw_cache(cache_file: Path, needed_bytes: int,
                     fetch_stream: Callable[[], Iterable[bytes]]) -> bytes:
    """Return >= needed_bytes of raw source, fetching/refreshing the cache.

    ``fetch_stream()`` yields byte chunks from the network source. If the
    cache already holds enough bytes it is reused verbatim (no network);
    otherwise the source is re-streamed from scratch until ``needed_bytes``
    are cached. Raises if the source is exhausted before the quota is met.
    """
    cache_file = Path(cache_file)
    if cache_file.exists() and cache_file.stat().st_size >= needed_bytes:
        print(f"  raw cache hit: {cache_file} "
              f"({cache_file.stat().st_size} bytes >= {needed_bytes} needed)")
        return cache_file.read_bytes()[:needed_bytes]
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"  raw cache miss: fetching {needed_bytes} bytes -> {cache_file}")
    buf = bytearray()
    for chunk in fetch_stream():
        buf.extend(chunk)
        if len(buf) >= needed_bytes:
            break
    if len(buf) < needed_bytes:
        raise SystemExit(
            f"source exhausted at {len(buf)} bytes < {needed_bytes} needed "
            f"for {cache_file.stem}; reduce --num-sequences or --context-length")
    tmp = cache_file.with_suffix(cache_file.suffix + '.tmp')
    tmp.write_bytes(bytes(buf))
    tmp.rename(cache_file)
    return bytes(buf[:needed_bytes])


def chunk_bytes(raw: bytes, window: int) -> Iterator[list]:
    """Yield consecutive exact-``window`` slices of ``raw`` (tail dropped)."""
    for i in range(0, len(raw) - window + 1, window):
        yield list(raw[i:i + window])


def chunk_tokens(ids: list, window: int) -> Iterator[list]:
    """``chunk_bytes`` twin for a token-id stream (tokenized corpora): yield
    consecutive exact-``window`` slices of ``ids`` (tail dropped)."""
    for i in range(0, len(ids) - window + 1, window):
        yield list(ids[i:i + window])


def make_record(stem: str, seq: list, index: int, context_length: int,
                extra_metadata: dict) -> dict:
    window = context_length - PREFIX_LEN
    if len(seq) != window:
        raise AssertionError(f"bake bug: {stem} record {index} has "
                             f"{len(seq)} bytes, window is {window}")
    metadata = {'sequence_length': window, 'context_length': context_length}
    metadata.update(extra_metadata)
    return {'sequence': seq, 'id': f'{stem}_{window}_{index}',
            'metadata': metadata}


def write_bake(out_path: Path, records: Iterable[dict], context_length: int,
               overwrite: bool = False) -> int:
    """Write records to a context-stamped JSONL and validate the result.

    Refuses to clobber without ``overwrite``. Returns the sequence count.
    """
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        raise SystemExit(f"{out_path} already exists; pass --overwrite to replace it")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix('.jsonl.tmp')
    n = 0
    with open(tmp, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
            n += 1
    tmp.rename(out_path)
    n_validated = validate_baked_file(out_path, context_length)
    assert n_validated == n
    print(f"  wrote {n} sequences to {out_path} "
          f"(window {context_length - PREFIX_LEN}, validated)")
    return n
