"""Re-chunk a legacy variable/fixed-length benchmark JSONL to an exact window.

Legacy benchmark files (dclm, dna, text, audio_*) were baked as
consecutive chunks of an underlying byte stream, with the stream identified by
``metadata.original_file`` (KoLMo corpora), ``metadata.subset`` or
``metadata.global_shard`` (dclm), ordered by ``metadata.chunk_index`` and
carrying the tail remainder as a final short chunk. This script reconstructs
each stream by concatenation and re-slices it into sequences of exactly
``context_length - PREFIX_LEN`` data bytes (per-stream tails dropped, counts
reported): a deterministic, network-free re-bake for any context.

NOT applicable to corpora whose records are independent sequences rather than
stream chunks (e.g. ``synthetic.jsonl``: ~13k one-record groups); the script
fails loudly if the input yields (almost) no full windows.

Usage (from the scoring directory):
    python -m src.scripts.prepare_kolmo_rechunk \
        --input <source jsonl> --context-length 4096
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.scripts.bake_utils import (bake_output_path, chunk_bytes, make_record,
                                    window_from_context, write_bake)


def group_key(metadata: dict):
    """The stream a record belongs to (see module docstring)."""
    for key in ('original_file', 'subset', 'global_shard'):
        if key in metadata:
            return f"{key}={metadata[key]}"
    return '<ungrouped>'


def load_streams(input_path: Path) -> dict:
    """{group_key: bytes} with chunks concatenated in chunk_index order."""
    groups = defaultdict(list)
    with open(input_path, 'r') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            record = json.loads(line)
            meta = record.get('metadata', {})
            idx = meta.get('chunk_index')
            if idx is None:
                raise SystemExit(
                    f"{input_path} line {i}: no metadata.chunk_index; this "
                    "corpus is not stream-chunked and cannot be re-chunked; "
                    "re-bake it from its upstream source instead")
            groups[group_key(meta)].append((idx, bytes(record['sequence'])))
    streams = {}
    for key, chunks in groups.items():
        chunks.sort(key=lambda t: t[0])
        indices = [idx for idx, _ in chunks]
        if indices != sorted(set(indices)):
            raise SystemExit(f"{input_path}: duplicate chunk_index in group "
                             f"{key}; cannot reconstruct the stream")
        streams[key] = b''.join(data for _, data in chunks)
    return streams


def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-chunk a legacy benchmark JSONL to an exact eval window.")
    parser.add_argument('--input', required=True,
                        help='Legacy benchmark JSONL (stream-chunked)')
    parser.add_argument('--context-length', type=int, required=True,
                        help='Target context (prefix + data bytes); the baked '
                             'window is context_length - 1')
    parser.add_argument('--output', default=None,
                        help='Output path (default: '
                             '<data_dir>/c{context_length}/<input stem>.jsonl)')
    parser.add_argument('--num-sequences', type=int, default=0,
                        help='Cap on baked sequences, first-N (default 0 = all)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite an existing output file')
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"input not found: {input_path}")
    window = window_from_context(args.context_length)
    stem = input_path.stem
    out_path = (Path(args.output) if args.output
                else bake_output_path(stem, args.context_length))

    streams = load_streams(input_path)
    total_bytes = sum(len(s) for s in streams.values())
    print(f"{stem}: {len(streams)} stream(s), {total_bytes} bytes total")

    dropped_tail = sum(len(s) % window for s in streams.values())
    print(f"  tail bytes dropped (never crossing streams): {dropped_tail}")

    def records():
        index = 0
        for key, stream in streams.items():
            for seq in chunk_bytes(stream, window):
                if args.num_sequences and index >= args.num_sequences:
                    return
                yield make_record(stem, seq, index, args.context_length,
                                  {'source': f'rechunk:{input_path.name}',
                                   'group': key, 'chunk_index': index})
                index += 1

    n_possible = sum(len(s) // window for s in streams.values())
    if n_possible == 0:
        raise SystemExit(
            f"{stem}: no stream yields even one {window}-byte window; this "
            "corpus is not re-chunkable at this context; re-bake it from its "
            "upstream source")

    write_bake(out_path, records(), args.context_length,
               overwrite=args.overwrite)


if __name__ == '__main__':
    main()
