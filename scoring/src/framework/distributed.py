"""Rank helpers for evaluating across multiple processes.

Each helper is the identity in the single-process case, so evaluation code
reads the same whether or not a process group is initialized.
"""

from __future__ import annotations

import torch.distributed as dist


def is_dist() -> bool:
    """True when a process group is initialized (multi-process run)."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main() -> bool:
    """Rank 0 (or single-process): the only rank that writes files or logs."""
    return get_rank() == 0


def barrier() -> None:
    if is_dist():
        dist.barrier()
