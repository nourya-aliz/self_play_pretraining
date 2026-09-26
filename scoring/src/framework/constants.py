import numpy as np
import torch
import torch.nn.functional as F

BF_CHARS = b'<>+-[].,F'
EDIT_CHARS = np.frombuffer(BF_CHARS[:-1], dtype=np.uint8)   # BF alphabet minus 'F'

# Prefix conventions: every sequence starts with a single tag byte that marks
# what kind of sequence it is. The model discriminates the two solely by this
# first byte (see model.py output_mask), so one byte is sufficient.
#   Programs: b'S'   Outputs: b'O'
PREFIX_LEN = 1
PROGRAM_PREFIX = b'S'
OUTPUT_PREFIX = b'O'

# In causal-LM training (inputs=seq[:-1], targets=seq[1:]), the prefix occupies
# token 0; loss is computed on tokens from index LOSS_TOKEN_OFFSET onward (the
# content, i.e. everything after the prefix).
LOSS_TOKEN_OFFSET = PREFIX_LEN - 1  # = 0

# Byte-alphabet size: the table width of the byte-level n-gram baselines and
# the value range of byte corpora. Not the model vocab, which is read from
# each checkpoint's config.json.
BYTE_VOCAB_SIZE = 256

# Structural floor for the model vocab: every BF program byte and the S/O
# prefix tags must be embeddable regardless of cell_modulus. (']' = 93 is the
# largest structural byte, so this is 94.) Whether a given EVAL corpus fits is
# a per-corpus check (EvaluationSuite.validate_vocab), not a model invariant.
MIN_VOCAB_SIZE = max(max(BF_CHARS), PROGRAM_PREFIX[0], OUTPUT_PREFIX[0]) + 1


def program_prefix_ids(n: int, device) -> torch.Tensor:
    """``[n, PREFIX_LEN]`` long tensor of the program prefix; generation seeds."""
    row = torch.tensor(list(PROGRAM_PREFIX), dtype=torch.long, device=device)
    return row.unsqueeze(0).expand(n, PREFIX_LEN).contiguous()


def add_program_prefix(programs: np.ndarray) -> np.ndarray:
    """Write the program prefix over the leading ``PREFIX_LEN`` cols of ``[N, L]``.

    Used for fixed-length program arrays (e.g. uniformly-sampled programs) whose
    first ``PREFIX_LEN`` bytes are the tag; the generator path instead seeds
    generation with :func:`program_prefix_ids`.
    """
    programs[:, :PREFIX_LEN] = np.frombuffer(PROGRAM_PREFIX, dtype=np.uint8)
    return programs


def add_output_prefix(outputs: np.ndarray) -> np.ndarray:
    """Prepend the output prefix columns to ``[N, L]`` raw executor outputs."""
    prefix = np.frombuffer(OUTPUT_PREFIX, dtype=np.uint8)
    tiled = np.tile(prefix, (len(outputs), 1))
    return np.concatenate([tiled, outputs], axis=1)


def strip_prefix(seq: np.ndarray) -> np.ndarray:
    """Drop the leading ``PREFIX_LEN`` tag bytes along the last axis."""
    return seq[..., PREFIX_LEN:]


# Clip bounds for log importance weights. exp(±20) ≈ [2e-9, 5e8].
LOG_IW_CLIP = 20

def prefix_ce_loss(logits, targets, reduction='none', label_smoothing=0.0):
    """Cross-entropy loss on output content tokens (skipping prefix).

    Args:
        logits: [B, seq_len, vocab] model output (inputs=seq[:-1] convention)
        targets: [B, seq_len] target token IDs
        reduction: 'none' returns [B, content_len], 'sum' returns scalar, 'mean' returns scalar
        label_smoothing: label smoothing factor
    """
    return F.cross_entropy(
        logits[:, LOSS_TOKEN_OFFSET:].transpose(-1, -2),
        targets[:, LOSS_TOKEN_OFFSET:],
        reduction=reduction,
        label_smoothing=label_smoothing,
    )


def output_ce_per_sequence(logits, targets, label_smoothing=0.0, output_lens=None):
    """Per-sequence mean CE over output content tokens, optionally padding-masked.

    Returns a ``[B]`` tensor: the mean ``prefix_ce_loss`` over each sequence's
    content tokens. Content position ``c`` (after the prefix slice) corresponds to
    output byte ``c`` of the executed program.

    If ``output_lens`` is None, this is the plain mean over ALL content tokens
    (identical to the previous ``prefix_ce_loss(...).mean(-1)``), i.e. it includes
    the zero-padding after the program halted.

    If ``output_lens`` is given (``[B]`` int, the emitted-byte count
    ``Program.output_ptr`` per sequence), only positions ``c < output_lens[i]`` are
    averaged; the never-written padding is dropped. Sequences that emitted nothing
    (``output_lens == 0``) have no content and get loss ``0``.
    """
    ce = prefix_ce_loss(logits, targets, reduction='none',
                        label_smoothing=label_smoothing)  # [B, C]
    if output_lens is None:
        return ce.mean(dim=-1)
    n_content = ce.shape[1]
    if not torch.is_tensor(output_lens):
        output_lens = torch.as_tensor(output_lens, device=ce.device)
    output_lens = output_lens.to(ce.device)
    positions = torch.arange(n_content, device=ce.device)
    mask = positions.unsqueeze(0) < output_lens.unsqueeze(1)  # [B, C] bool
    count = mask.sum(dim=-1)
    return (ce * mask).sum(dim=-1) / count.clamp(min=1)