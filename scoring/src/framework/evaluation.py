"""Evaluation subsystem: one loader, one evaluator, one loss convention.

Shared by every evaluation path: the checkpoint scorer, validation corpora,
and the n-gram baselines.

The single length knob is ``context_length = PREFIX_LEN + data_tokens``; it
INCLUDES the prefix byte and equals the model context (``max_len``). The eval
*window* (real data tokens per sequence) is ``context_length - PREFIX_LEN``,
and every corpus must be baked at EXACTLY that window:

* no padding: every scored position is a real data byte;
* no truncation: a longer line is as fatal as a shorter one;
* no silent skips: a missing file or requested corpus raises at construction,
  with the exact ``prepare_*`` command to run;
* subsampling exists only as an EXPLICIT first-N cap, derived once at
  construction from ``0 < max_samples < n_corpus`` and reported loudly.

Baked corpora are context-stamped: ``<benchmark_dir>/c{context_length}/<stem>.jsonl``
with each record ``{"sequence": [ints], "id": str, "metadata": {...}}`` whose
``metadata.sequence_length`` (when present) must equal the window.

``__post_init__``/constructor guards VALIDATE ONLY; no mutation, coercion,
padding, or repair. The O(N·L) line-level scan runs ONCE at construction
(startup), never in the eval loop. Error style:
``f"{ClassName}.{field} <expectation>, got {actual}"``, always naming the
file and the re-bake command where one applies.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from .constants import (LOSS_TOKEN_OFFSET, OUTPUT_PREFIX,
                        PREFIX_LEN)
from .distributed import get_rank, get_world_size, is_main

# The shared currency between the resolvers and the factory: one baked corpus
# per (stem, path). Named because it crosses three function boundaries.
EvalEntries = List[Tuple[str, Path]]
# One evaluated corpus' metrics (evaluate_eval_loader): a str→float mapping.
EvalMetrics = Dict[str, float]
# A filesystem location accepted anywhere a baked-corpus path is expected.
PathLike = Union[str, Path]
# The two distributed evaluation modes (see EvaluationSuite).
EvalMode = Literal['sharded', 'rank0']

# ---------------------------------------------------------------------------
# Re-bake command registry; used verbatim in every failure message so the fix
# is always one copy-paste away. ``{C}`` = context_length, ``{path}`` = the
# expected baked file. Stems not listed fall back to the re-chunker.
# ---------------------------------------------------------------------------
# Four corpora are shipped already baked in <scoring>/data/c4096 rather than
# rebuilt from a source: audio_8bit, audio_16bit, dna, dclm_ranked.
REBAKE_COMMANDS = {
    'dclm': ('python -m src.scripts.prepare_dclm_benchmark '
             '--context-length {C} --output {path} --overwrite'),
    'arithmetic': ('python -m src.scripts.prepare_arithmetic_benchmark '
                   '--context-length {C} --output {path} --overwrite'),
    'speech_commands_pcm8': (
        'python -m src.scripts.prepare_speech_commands_benchmark '
        '--context-length {C} --output {path} --overwrite'),
    'speech_commands_pcm8_4khz': (
        'python -m src.scripts.prepare_speech_commands_benchmark '
        '--datasets speech_commands_pcm8_4khz --context-length {C} '
        '--output {path} --overwrite'),
    'speech_commands_pcm8_8khz': (
        'python -m src.scripts.prepare_speech_commands_benchmark '
        '--datasets speech_commands_pcm8_8khz --context-length {C} '
        '--output {path} --overwrite'),
    'mutopia_melody_16th': (
        'python -m src.scripts.prepare_mutopia_benchmark '
        '--context-length {C} --output {path} --overwrite'),
    'kolmogorov_text': ('python -m src.scripts.prepare_kolmogorov_benchmark '
                        '--datasets kolmogorov_text --context-length {C} '
                        '--output {path} --overwrite'),
    'kolmogorov_dna': ('python -m src.scripts.prepare_kolmogorov_benchmark '
                       '--datasets kolmogorov_dna --context-length {C} '
                       '--output {path} --overwrite'),
    'aitdcc_a_protein': ('python -m src.scripts.prepare_aitdcc_benchmark '
                         '--datasets aitdcc_a_protein --context-length {C} '
                         '--output {path} --overwrite'),
    'aitdcc_b_c_source': ('python -m src.scripts.prepare_aitdcc_benchmark '
                          '--datasets aitdcc_b_c_source --context-length {C} '
                          '--output {path} --overwrite'),
    'aitdcc_d_glibc_rand': ('python -m src.scripts.prepare_aitdcc_benchmark '
                            '--datasets aitdcc_d_glibc_rand --context-length {C} '
                            '--output {path} --overwrite'),
    'aitdcc_e_atlas_float32': (
        'python -m src.scripts.prepare_aitdcc_benchmark '
        '--datasets aitdcc_e_atlas_float32 --context-length {C} '
        '--output {path} --overwrite'),
    'aitdcc_g_astronomy': ('python -m src.scripts.prepare_aitdcc_benchmark '
                           '--datasets aitdcc_g_astronomy --context-length {C} '
                           '--output {path} --overwrite'),
    'metamath': ('python -m src.scripts.prepare_metamath_benchmark '
                 '--context-length {C} --output {path} --overwrite'),
    'cifar10_rgb_planar': (
        'python -m src.scripts.prepare_cifar10_benchmark '
        '--datasets cifar10_rgb_planar --context-length {C} '
        '--output {path} --overwrite'),
    'cifar10_rgb_hwc': (
        'python -m src.scripts.prepare_cifar10_benchmark '
        '--datasets cifar10_rgb_hwc --context-length {C} '
        '--output {path} --overwrite'),
    # hkust-nlp/llm-compression (Huang et al., COLM 2024): UTF-8 bytes of
    # GitHub Python / Common Crawl / arXiv-math documents.
    'llm_compression_python': (
        'python -m src.scripts.prepare_llm_compression_benchmark '
        '--datasets llm_compression_python --context-length {C} '
        '--output {path} --overwrite'),
    'llm_compression_cc': (
        'python -m src.scripts.prepare_llm_compression_benchmark '
        '--datasets llm_compression_cc --context-length {C} '
        '--output {path} --overwrite'),
    'llm_compression_arxiv_math': (
        'python -m src.scripts.prepare_llm_compression_benchmark '
        '--datasets llm_compression_arxiv_math --context-length {C} '
        '--output {path} --overwrite'),
    # ESC-50 (Piczak, ACM MM 2015): 44.1 kHz environmental sound as PCM8.
    'esc50_pcm8': (
        'python -m src.scripts.prepare_esc50_benchmark '
        '--datasets esc50_pcm8 --context-length {C} '
        '--output {path} --overwrite'),
    'esc50_pcm8_11khz': (
        'python -m src.scripts.prepare_esc50_benchmark '
        '--datasets esc50_pcm8_11khz --context-length {C} '
        '--output {path} --overwrite'),
    # MusicNet test split (Thickstun et al., ICLR 2017): 44.1 kHz classical
    # music as PCM8.
    'musicnet_pcm8': (
        'python -m src.scripts.prepare_musicnet_benchmark '
        '--datasets musicnet_pcm8 --context-length {C} '
        '--output {path} --overwrite'),
    'musicnet_pcm8_11khz': (
        'python -m src.scripts.prepare_musicnet_benchmark '
        '--datasets musicnet_pcm8_11khz --context-length {C} '
        '--output {path} --overwrite'),
}

_REBAKE_FALLBACK = ('python -m src.scripts.prepare_kolmo_rechunk '
                    '--input <source jsonl for {name}> --context-length {C} '
                    '--output {path}')


def rebake_command(name: str, context_length: int, path: PathLike) -> str:
    """The exact command that (re)bakes corpus ``name`` at ``context_length``."""
    template = REBAKE_COMMANDS.get(name, _REBAKE_FALLBACK)
    return template.format(name=name, C=context_length, path=path)


# ---------------------------------------------------------------------------
# EvalSpec: the declaration of ONE eval corpus. Frozen; validate-only.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EvalSpec:
    name: str            # corpus stem == results key
    path: Path           # baked JSONL (context-stamped)
    context_length: int  # PREFIX_LEN + data tokens; == model max_len
    max_samples: int = 0  # 0 = full file; > 0 = EXPLICIT first-N cap

    def __post_init__(self):
        if not self.name:
            raise ValueError(f"EvalSpec.name must be non-empty, got {self.name!r}")
        if self.context_length <= PREFIX_LEN:
            raise ValueError(f"EvalSpec.context_length must be > PREFIX_LEN "
                             f"({PREFIX_LEN}), got {self.context_length}")
        if self.max_samples < 0:
            raise ValueError(f"EvalSpec.max_samples must be >= 0, got {self.max_samples}")

    @property
    def window(self) -> int:
        """Real data tokens per sequence (the eval window, sans prefix)."""
        return self.context_length - PREFIX_LEN


# ---------------------------------------------------------------------------
# Line-level bake validator: shared by EvalCorpus and
# the post-bake verification step in the prepare_* scripts.
# ---------------------------------------------------------------------------
def validate_baked_file(path: PathLike, context_length: int) -> int:
    """Validate every line of a baked JSONL against the exact eval window.

    Returns the sequence count on success. Raises:
      FileNotFoundError: file missing (message carries the re-bake command);
      ValueError: empty file, wrong-length sequence, or metadata that
                   contradicts the declared window (first offending line named).
    """
    path = Path(path)
    window = context_length - PREFIX_LEN
    cmd = rebake_command(path.stem, context_length, path)
    if not path.exists():
        raise FileNotFoundError(
            f"eval corpus {path} not found for context_length={context_length}; "
            f"bake it: {cmd}")
    n = 0
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            record = json.loads(line)
            seq_len = len(record['sequence'])
            if seq_len != window:
                raise ValueError(
                    f"{path} line {i}: sequence length {seq_len} != eval window "
                    f"{window} (context_length {context_length} - prefix "
                    f"{PREFIX_LEN}); re-bake: {cmd}")
            meta_len = record.get('metadata', {}).get('sequence_length')
            if meta_len is not None and meta_len != window:
                raise ValueError(
                    f"{path} line {i}: metadata.sequence_length {meta_len} != "
                    f"eval window {window}; re-bake: {cmd}")
            n += 1
    if n == 0:
        raise ValueError(f"{path} contains no sequences; re-bake: {cmd}")
    return n


# ---------------------------------------------------------------------------
# EvalCorpus: the eval dataset class. Construction validates every line;
# there is no repair path. Tensor contract: prefix prepended once,
# __getitem__ -> (seq[:-1], seq[1:]) int64.
# ---------------------------------------------------------------------------
class EvalCorpus(Dataset):
    def __init__(self, spec: EvalSpec):
        self.spec = spec
        n = validate_baked_file(spec.path, spec.context_length)
        # int64, NOT uint8: tokenized corpora (e.g. Llama-2, ids up to 31999)
        # carry values > 255, and a uint8 load would silently wrap them.
        array = np.empty((n, spec.window), dtype=np.int64)
        with open(spec.path, 'r') as f:
            row = 0
            for line in f:
                if not line.strip():
                    continue
                array[row] = np.asarray(json.loads(line)['sequence'], dtype=np.int64)
                row += 1
        if array.min() < 0:
            raise ValueError(
                f"eval corpus {spec.path} contains negative token id "
                f"{int(array.min())}; token ids must be >= 0")
        self.max_token_id = int(array.max())
        prefix = torch.tensor(bytearray(OUTPUT_PREFIX), dtype=torch.long)
        body = torch.from_numpy(array)
        self.sequences = torch.cat(
            [prefix.expand(n, PREFIX_LEN), body], dim=1)  # [n, context_length]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        return seq[:-1], seq[1:]  # (inputs, targets)


# ---------------------------------------------------------------------------
# THE single evaluator (canonical float64-sums math; the all-reduce is folded
# behind ``world``). Loss semantics come entirely from output_ce_per_sequence:
# every target position in the window contributes, and via EvalCorpus every one
# of them is a real data byte.
# ---------------------------------------------------------------------------
def evaluate_eval_loader(model: nn.Module, dataloader: DataLoader,
                         device: Union[str, torch.device],
                         world: int = 1) -> EvalMetrics:
    """Returns {'seq_ce', 'loss', 'bpb'}: per-sequence CE sum, per-token CE,
    and its bit scaling, each averaged over sequences (globally when world>1)."""
    model.eval()
    sums = torch.zeros(3, dtype=torch.float64, device=device)
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            per_sequence_ce, _ = model(input_ids=inputs, targets=targets)
            content_length = targets[:, LOSS_TOKEN_OFFSET:].shape[1]
            per_sequence_ce = per_sequence_ce.to(torch.float64)
            sums[0] += per_sequence_ce.sum()
            sums[1] += (per_sequence_ce * content_length).sum()
            sums[2] += len(inputs)
    if world > 1:
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    ce_sum, seq_ce_sum, n_sequences = sums.tolist()
    if n_sequences == 0:
        # Unreachable through EvalCorpus (empty bakes fail at construction);
        # loud rather than an inf metric if a foreign loader sneaks in.
        raise ValueError("evaluate_eval_loader scored 0 sequences")
    loss = ce_sum / n_sequences
    return {'seq_ce': seq_ce_sum / n_sequences,
            'loss': loss,
            'bpb': loss / math.log(2)}


# The per-corpus result contract. Guarded where results are assembled
# (EvaluationSuite.evaluate); the serialization boundary for eval dicts.
EVAL_RESULT_KEYS = frozenset({'seq_ce', 'loss', 'bpb', 'n_seqs', 'context_length'})


# ---------------------------------------------------------------------------
# EvaluationSuite: validated {name: loader} built once at startup.
# ---------------------------------------------------------------------------
class EvaluationSuite:
    """The consolidated evaluation suite.

    mode='sharded': distributed collective: each rank scores a strided shard
      (``Subset(range(rank, n, world))``) and ``evaluate`` all-reduces the
      scalar sums, so EVERY rank must construct the suite identically (same
      CLI strings) and call ``evaluate`` together; all ranks return identical
      full-set metrics.
    mode='rank0': full corpus on the calling rank, no collective.
    """

    def __init__(self, specs: Sequence[EvalSpec], batch_size: int,
                 mode: EvalMode):
        if mode not in ('sharded', 'rank0'):
            raise ValueError(f"EvaluationSuite.mode must be 'sharded' or 'rank0', "
                             f"got {mode!r}")
        if not specs:
            raise ValueError("EvaluationSuite.specs must be non-empty")
        contexts = {s.context_length for s in specs}
        if len(contexts) != 1:
            raise ValueError(f"EvaluationSuite.specs must share one "
                             f"context_length, got {sorted(contexts)}")
        names = [s.name for s in specs]
        if len(set(names)) != len(names):
            raise ValueError(f"EvaluationSuite.specs have duplicate names: {names}")

        self._mode = mode
        self._context_length = next(iter(contexts))
        self._world = get_world_size() if mode == 'sharded' else 1
        self._specs: Dict[str, EvalSpec] = {}
        self._loaders: Dict[str, DataLoader] = {}
        self._n_corpus: Dict[str, int] = {}
        self._n_scored: Dict[str, int] = {}
        self._max_token: Dict[str, int] = {}

        for spec in specs:
            corpus = EvalCorpus(spec)                      # line-level validation
            n_corpus = len(corpus)
            dataset = corpus
            if 0 < spec.max_samples < n_corpus:
                dataset = Subset(dataset, range(spec.max_samples))  # EXPLICIT first-N
            n_scored = len(dataset)
            if self._world > 1:
                # Exact strided partition: neither pads nor duplicates rows;
                # evaluate() all-reduces the sums.
                dataset = Subset(dataset, range(get_rank(), n_scored, self._world))
            self._specs[spec.name] = spec
            self._loaders[spec.name] = DataLoader(dataset, batch_size=batch_size,
                                                  shuffle=False)
            self._n_corpus[spec.name] = n_corpus
            self._n_scored[spec.name] = n_scored
            self._max_token[spec.name] = corpus.max_token_id
            if is_main():
                cap = (f", EXPLICIT first-{n_scored} subsample"
                       if self.subsampled(spec.name) else "")
                stride = (f", strided over {self._world} ranks"
                          if self._world > 1 else "")
                print(f"  [eval] {spec.name}: context={self._context_length} "
                      f"window={spec.window} n_corpus={n_corpus} "
                      f"n_scored={n_scored}{cap}{stride}")

    # -- protocol surface ----------------------------------------------------
    @property
    def context_length(self) -> int:
        return self._context_length

    @property
    def window(self) -> int:
        return self._context_length - PREFIX_LEN

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(self._loaders)

    def subsampled(self, name: str) -> bool:
        """True iff this corpus is explicitly capped below its full size."""
        return self._n_scored[name] < self._n_corpus[name]

    def validate_vocab(self, vocab_size: int) -> None:
        """Every corpus token id must be scoreable by a ``vocab_size`` model.

        Called at startup so a tokenized corpus (e.g. Llama-2, ids up to
        31999) against a too-small model fails loudly, never as an
        out-of-range embedding lookup mid-run.
        """
        for name, max_id in self._max_token.items():
            if max_id >= vocab_size:
                raise ValueError(
                    f"eval corpus '{name}' contains token id {max_id} >= model "
                    f"vocab_size {vocab_size}; drop the corpus or use a model "
                    f"with vocab_size >= {max_id + 1}")

    def evaluate(self, model: nn.Module,
                 device: Union[str, torch.device]) -> Dict[str, EvalMetrics]:
        """{name: result}; every result carries exactly EVAL_RESULT_KEYS."""
        out: Dict[str, EvalMetrics] = {}
        for name, loader in self._loaders.items():
            result = dict(evaluate_eval_loader(model, loader, device,
                                               world=self._world))
            result['n_seqs'] = self._n_scored[name]
            result['context_length'] = self._context_length
            keys = frozenset(result)
            if keys != EVAL_RESULT_KEYS:
                raise ValueError(
                    "eval result key set diverged from EVAL_RESULT_KEYS: "
                    f"missing={sorted(EVAL_RESULT_KEYS - keys)}, "
                    f"unexpected={sorted(keys - EVAL_RESULT_KEYS)}")
            out[name] = result
        return out


# ---------------------------------------------------------------------------
# Entry resolution + factory. HARD FAILS (never warn+skip): missing file,
# missing requested stem, empty selection, wrong-length lines.
# ---------------------------------------------------------------------------
def context_dir(benchmark_dir: PathLike, context_length: int) -> Path:
    """The context-stamped bake directory: <benchmark_dir>/c{context_length}."""
    return Path(benchmark_dir) / f"c{context_length}"


def resolve_benchmark_entries(benchmark_dir: PathLike, context_length: int,
                              stems_csv: str = '') -> EvalEntries:
    """[(name, path)] for the requested stems under the context-stamped dir.

    Empty ``stems_csv`` selects every ``*.jsonl`` in the dir (which must then
    be non-empty). A requested stem with no baked file raises with the exact
    re-bake command; requested corpora are never silently omitted.
    """
    root = context_dir(benchmark_dir, context_length)
    wanted = [s.strip() for s in str(stems_csv or '').split(',') if s.strip()]
    if not wanted:
        paths = sorted(root.glob('*.jsonl'))
        if not paths:
            raise FileNotFoundError(
                f"no baked eval corpora in {root} for "
                f"context_length={context_length}; bake them, e.g.: "
                f"{rebake_command('dclm', context_length, root / 'dclm.jsonl')}")
        return [(p.stem, p) for p in paths]
    entries: EvalEntries = []
    for stem in wanted:
        path = root / f"{stem}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"eval corpus '{stem}' not baked for "
                f"context_length={context_length} (expected {path}); bake it: "
                f"{rebake_command(stem, context_length, path)}")
        entries.append((stem, path))
    return entries


def resolve_dataset_entries(datasets_csv: str, benchmark_dir: PathLike,
                            context_length: int) -> EvalEntries:
    """[(name, path)] for a comma-separated list of stems and/or paths.

    Bare stems resolve through the context-stamped dir (like benchmark
    entries); anything containing a path separator or a .jsonl suffix is used
    verbatim (and still exact-window validated by EvalCorpus).
    """
    entries: EvalEntries = []
    seen: set = set()
    for raw in str(datasets_csv or '').split(','):
        item = raw.strip()
        if not item:
            continue
        if '/' in item or item.endswith('.jsonl'):
            name, path = Path(item).stem, Path(item)
            if not path.exists():
                raise FileNotFoundError(
                    f"eval corpus {path} not found for "
                    f"context_length={context_length}; bake it: "
                    f"{rebake_command(name, context_length, path)}")
        else:
            (name, path), = resolve_benchmark_entries(
                benchmark_dir, context_length, item)
        if name not in seen:
            seen.add(name)
            entries.append((name, path))
    return entries


def build_eval_suite(entries: EvalEntries, context_length: int, batch_size: int,
                     max_samples: int = 0, mode: EvalMode = 'rank0'
                     ) -> EvaluationSuite:
    """Construct the validated suite from [(name, path)] entries.

    Everything loud happens here, at startup: line-level window validation,
    missing-corpus failures with re-bake commands, the explicit-subsample
    derivation, and the per-corpus coverage report.
    """
    specs = [EvalSpec(name=name, path=Path(path), context_length=context_length,
                      max_samples=max_samples)
             for name, path in entries]
    return EvaluationSuite(specs, batch_size=batch_size, mode=mode)
