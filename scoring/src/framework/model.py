"""Byte-level language model: inference-only definition for the released weights.

Llama-style decoder-only transformer: RMSNorm, rotary position embeddings,
SwiGLU feed-forward, grouped-query attention, ``1/sqrt(head_dim)`` attention
scaling, and no biases. Parameter names and shapes match the released
checkpoints, so a state dict loads directly.

``forward(input_ids)`` returns ``(logits, None)``. Logits at program positions
are restricted to the instruction alphabet, and positions after a program's
terminating ``F`` byte are masked, matching the convention the checkpoints
were scored under.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import BF_CHARS, OUTPUT_PREFIX

_RMS_NORM_EPS = 1e-5
_ROPE_THETA = 10000.0
_SWIGLU_MULTIPLE = 256
_NEG_INF = torch.finfo(torch.float32).min


def llama_intermediate_size(d_model: int, multiple_of: int = _SWIGLU_MULTIPLE) -> int:
    """SwiGLU hidden width: round ``8/3 * d_model`` up to ``multiple_of``."""
    hidden = int(8 * d_model / 3)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


def llama_num_kv_heads(n_heads: int) -> int:
    """Grouped-query attention: four query heads per key/value head."""
    return max(1, n_heads // 4)


class _RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = _RMS_NORM_EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


class _RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = _ROPE_THETA):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, positions):
        # positions [T] -> cos/sin [1, 1, T, head_dim], broadcast over batch
        # and heads.
        freqs = torch.outer(positions.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)[None, None]
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rotary(q, k, cos, sin):
    return ((q * cos + _rotate_half(q) * sin).to(q.dtype),
            (k * cos + _rotate_half(k) * sin).to(k.dtype))


def _repeat_kv(x, groups: int):
    if groups == 1:
        return x
    B, H, T, D = x.shape
    return x[:, :, None].expand(B, H, groups, T, D).reshape(B, H * groups, T, D)


def _row_masks(input_ids):
    """``(active [B, T], output_mask [B])`` boolean masks.

    A position is active until its row's first ``F`` byte; rows that start
    with the output prefix are always fully active.
    """
    output_mask = input_ids[:, 0] == OUTPUT_PREFIX[0]
    f_cumsum = (input_ids == ord("F")).cumsum(dim=1)
    active = (f_cumsum < 1) | output_mask[:, None]
    return active, output_mask


class _CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_kv_groups = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim % 2 == 0, "RoPE needs an even head_dim"
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x, position_embeddings, kv_buf=None, cache_pos=0):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # Rotate before caching, so cached keys need no re-rotation later.
        q, k = _apply_rotary(q, k, *position_embeddings)

        if kv_buf is not None:
            k_buf, v_buf = kv_buf
            k_buf[:, :, cache_pos:cache_pos + T] = k
            v_buf[:, :, cache_pos:cache_pos + T] = v
            k = _repeat_kv(k_buf[:, :, :cache_pos + T], self.n_kv_groups)
            v = _repeat_kv(v_buf[:, :, :cache_pos + T], self.n_kv_groups)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1,
                                               scale=self.scale)
        else:
            y = F.scaled_dot_product_attention(q, _repeat_kv(k, self.n_kv_groups),
                                               _repeat_kv(v, self.n_kv_groups),
                                               is_causal=True, scale=self.scale)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class _MLP(nn.Module):
    """SwiGLU feed-forward: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, d_model: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 intermediate_size: int):
        super().__init__()
        self.input_layernorm = _RMSNorm(d_model)
        self.attn = _CausalSelfAttention(d_model, n_heads, n_kv_heads)
        self.post_attention_layernorm = _RMSNorm(d_model)
        self.mlp = _MLP(d_model, intermediate_size)

    def forward(self, x, position_embeddings, kv_buf=None, cache_pos=0):
        x = x + self.attn(self.input_layernorm(x), position_embeddings,
                          kv_buf=kv_buf, cache_pos=cache_pos)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class ProgramLanguageModel(nn.Module):
    """Byte-level Llama-style transformer over programs and their outputs.

    Construct with the values in a checkpoint's ``config.json`` and load its
    ``learner_state_dict``:

        model = ProgramLanguageModel(**config)
        model.load_state_dict(blob["learner_state_dict"], strict=False)
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 4,
                 max_len: int = 256, vocab_size: int = 256,
                 bf_chars: bytes = BF_CHARS, base_d_model: int = 16):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_len = max_len
        self.base_d_model = base_d_model      # kept for config round-tripping
        self.n_kv_heads = llama_num_kv_heads(n_heads)
        self.head_dim = d_model // n_heads
        self.intermediate_size = llama_intermediate_size(d_model)

        self.wte = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            _Block(d_model, n_heads, self.n_kv_heads, self.intermediate_size)
            for _ in range(n_layers)
        ])
        self.ln_f = _RMSNorm(d_model)
        self.rotary_emb = _RotaryEmbedding(self.head_dim)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        valid_chars = torch.tensor(np.array(bytearray(bf_chars)), dtype=torch.long)
        valid_mask = torch.zeros(vocab_size, dtype=torch.bool)
        valid_mask[valid_chars] = True
        self.register_buffer("program_vocab_mask", valid_mask, persistent=True)

        f_token_mask = torch.zeros(vocab_size, dtype=torch.bool)
        f_token_mask[ord("F")] = True
        self.register_buffer("f_token_mask", f_token_mask, persistent=False)

    def get_config(self) -> dict:
        """Constructor kwargs needed to rebuild this model."""
        return {"d_model": self.d_model, "n_heads": self.n_heads,
                "n_layers": self.n_layers, "max_len": self.max_len,
                "vocab_size": self.vocab_size, "base_d_model": self.base_d_model}

    def forward(self, input_ids: torch.Tensor, kv_buffers=None,
                cache_pos: int = 0) -> tuple:
        """Masked next-token logits: ``(logits [B, T, vocab_size], None)``."""
        seq_len = input_ids.shape[1]
        past_len = cache_pos if kv_buffers is not None else 0
        positions = torch.arange(past_len, past_len + seq_len,
                                 device=input_ids.device)

        x = self.wte(input_ids)
        pos_emb = self.rotary_emb(x, positions)
        for i, block in enumerate(self.blocks):
            kv = kv_buffers[i] if kv_buffers is not None else None
            x = block(x, pos_emb, kv_buf=kv, cache_pos=cache_pos)
        logits = self.lm_head(self.ln_f(x))

        # Program rows are restricted to the instruction alphabet (plus the
        # terminator); every row is masked past its first 'F'. Output rows are
        # unrestricted.
        active, output_mask = _row_masks(input_ids)
        vocab_ok = self.program_vocab_mask | self.f_token_mask
        keep = ((active.unsqueeze(-1) | self.f_token_mask)
                & (output_mask[:, None, None] | vocab_ok))
        return torch.where(keep, logits, float("-inf")), None

    @torch.inference_mode()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None) -> torch.Tensor:
        """Sample continuations with a preallocated key/value cache."""
        B = prompt.shape[0]
        total = prompt.shape[1] + max_new_tokens
        if total > self.max_len:
            raise ValueError(f"{total} tokens exceeds context {self.max_len}")
        device = prompt.device
        kv_buffers = [(torch.zeros(B, self.n_kv_heads, total, self.head_dim,
                                   device=device, dtype=self.wte.weight.dtype),
                       torch.zeros(B, self.n_kv_heads, total, self.head_dim,
                                   device=device, dtype=self.wte.weight.dtype))
                      for _ in range(self.n_layers)]
        out = prompt
        logits, _ = self.forward(prompt, kv_buffers=kv_buffers, cache_pos=0)
        pos = prompt.shape[1]
        for _ in range(max_new_tokens):
            step = logits[:, -1, :].float()
            if temperature != 1.0:
                step = step / max(temperature, 1e-6)
            if top_k is not None:
                kth = step.topk(min(top_k, step.shape[-1]), dim=-1).values[:, -1:]
                step = torch.where(step < kth, torch.full_like(step, _NEG_INF), step)
            nxt = torch.multinomial(torch.softmax(step, dim=-1), 1)
            out = torch.cat([out, nxt], dim=1)
            logits, _ = self.forward(out[:, -1:], kv_buffers=kv_buffers,
                                     cache_pos=pos)
            pos += 1
        return out
