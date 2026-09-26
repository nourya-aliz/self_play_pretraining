"""Shared constants for pipeline scripts.

Thin compatibility layer over :mod:`src.config.paths`. Kept so existing
imports (``DEFAULT_PATTERN`` etc.) continue to work.
"""

from src.config.paths import (
    SWEEP_GLOB as DEFAULT_PATTERN,
    SWEEP_SEED_GLOB as SWEEP_DIR_PATTERN,
    TEST_DATA_DIR,
)

DEFAULT_DATA_DIR = str(TEST_DATA_DIR)
