"""Model definition and evaluation contract for the released checkpoints.

Carries what benchmark evaluation needs: the model, the evaluation contract
for baked byte corpora at an exact context length, and the shared byte and
prefix constants.

``ProgramLanguageModel`` is exported lazily so that corpus preparation
scripts can import the lightweight children (``src.framework.constants``,
``src.framework.evaluation``) without importing torch.
"""

from typing import TYPE_CHECKING

from .constants import (PREFIX_LEN, OUTPUT_PREFIX, PROGRAM_PREFIX, LOSS_TOKEN_OFFSET,
                        EDIT_CHARS,
                        prefix_ce_loss, output_ce_per_sequence,
                        program_prefix_ids, add_program_prefix, add_output_prefix,
                        strip_prefix)
from .evaluation import (EvalSpec, EvalCorpus, EvaluationSuite, EVAL_RESULT_KEYS,
                         evaluate_eval_loader, validate_baked_file,
                         build_eval_suite, resolve_benchmark_entries,
                         resolve_dataset_entries, context_dir, rebake_command,
                         REBAKE_COMMANDS)

if TYPE_CHECKING:
    from .model import ProgramLanguageModel

__all__ = [
    'ProgramLanguageModel',
    'EvalSpec',
    'EvalCorpus',
    'EvaluationSuite',
    'EVAL_RESULT_KEYS',
    'evaluate_eval_loader',
    'validate_baked_file',
    'build_eval_suite',
    'resolve_benchmark_entries',
    'resolve_dataset_entries',
    'context_dir',
    'rebake_command',
    'REBAKE_COMMANDS',
    'PREFIX_LEN',
    'OUTPUT_PREFIX',
    'PROGRAM_PREFIX',
    'LOSS_TOKEN_OFFSET',
    'EDIT_CHARS',
    'prefix_ce_loss',
    'output_ce_per_sequence',
    'program_prefix_ids',
    'add_program_prefix',
    'add_output_prefix',
    'strip_prefix',
]


_LAZY_EXPORTS = {
    'ProgramLanguageModel': ('.model', 'ProgramLanguageModel'),
}


def __getattr__(name: str):
    """Load the model root export on first use (see module docstring)."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    import importlib

    module = importlib.import_module(module_name, __name__)
    obj = getattr(module, attr)
    globals()[name] = obj
    return obj
