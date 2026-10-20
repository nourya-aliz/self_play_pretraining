"""The program machine: Brainfuck with ten macro instructions (Appendix E).

A program is the generator's output after the ``S`` tag byte. It is cut at
its first ``F``, each macro byte is replaced by its fixed Brainfuck body, and
the result runs on a circular tape of byte cells:

* ``>`` / ``<`` move the head, wrapping around the tape.
* ``+`` / ``-`` add or subtract one modulo ``cell_modulus``.
* ``.`` appends the current cell to the output.
* ``,`` writes a uniformly random value in ``[0, cell_modulus)``.
* ``[`` / ``]`` loop as usual. Unmatched brackets are no-ops.

Every executed instruction costs one step. A run stops at the end of the
program, after ``max_steps`` steps, or once ``output_length`` values have been
written. The learner trains on ``O`` followed by the full output buffer, which
is zero past the last written value.

``run_program`` is a plain-Python reference. ``run_programs`` uses the
compiled kernel in ``machine_kernel.pyx`` when Cython and a C compiler are
available (much faster for long-running programs) and falls back to the
reference otherwise. Both give identical results for the same ``rng``.
"""

from typing import Optional

import numpy as np

from .constants import OUTPUT_PREFIX, PREFIX_LEN

MACROS = {
    'Z': '[-]',               # clear
    'R': '[->+<]',            # move / add into the next cell
    'L': '[->+++<]',          # add three times into the next cell
    'N': '[-<->]',            # subtract from the previous cell
    'C': '[->+>+<<]',         # copy into the next two cells
    'G': '[>]',               # scan right to a zero cell
    'H': '[<]',               # scan left to a zero cell
    'W': '[[-]>+<]',          # test and clear: next cell += (x != 0), x = 0
    'V': '[.>]',              # print cells until a zero cell
    'X': '[-]' + '+' * 16,    # set the cell to 16
}
_TRANSLATE = {ord(k): v for k, v in MACROS.items()}


def machine_limits(d_model: int, n_layers: int) -> dict:
    """Execution limits used in training for a model of this shape.

    The step budget and tape length grow with the model; outputs fill the
    4096-token context after the one-byte ``O`` tag.
    """
    return {
        'max_steps': max(2 ** 14, n_layers * d_model ** 2 * 16),
        'tape_length': max(1024, d_model * 8),
        'output_length': 4096 - PREFIX_LEN,
        'cell_modulus': 256,
    }


def expand(program: bytes) -> bytes:
    """Strip the ``S`` tag, cut at the first ``F`` and expand macros."""
    body = program[PREFIX_LEN:].decode('ascii', errors='ignore')
    body = body.split('F', 1)[0]
    return body.translate(_TRANSLATE).encode('ascii')


def _jump_table(code: bytes) -> list:
    jumps = list(range(len(code)))  # unmatched brackets jump to themselves
    stack = []
    for i, c in enumerate(code):
        if c == 91:      # '['
            stack.append(i)
        elif c == 93 and stack:  # ']'
            j = stack.pop()
            jumps[i], jumps[j] = j, i
    return jumps


def _next_uint32(rng: np.random.Generator):
    ct = rng.bit_generator.ctypes
    return lambda: ct.next_uint32(ct.state)


def run_program(program: bytes, rng: np.random.Generator, max_steps: int,
                tape_length: int, output_length: int,
                cell_modulus: int = 256):
    """Run one program; returns ``(output, n_written, steps)``.

    ``output`` is an int64 array of length ``output_length``; entries past
    ``n_written`` are zero. ``program`` includes the leading ``S`` tag.
    """
    code = expand(program)
    jumps = _jump_table(code)
    draw = _next_uint32(rng)
    tape = [0] * tape_length
    out = np.zeros(output_length, dtype=np.int64)
    head = pc = n_out = steps = 0
    n = len(code)
    while steps < max_steps and pc < n and n_out < output_length:
        c = code[pc]
        if c == 62:      # '>'
            head = (head + 1) % tape_length
        elif c == 60:    # '<'
            head = (head - 1) % tape_length
        elif c == 43:    # '+'
            tape[head] = (tape[head] + 1) % cell_modulus
        elif c == 45:    # '-'
            tape[head] = (tape[head] - 1) % cell_modulus
        elif c == 46:    # '.'
            out[n_out] = tape[head]
            n_out += 1
        elif c == 91:    # '['
            if tape[head] == 0:
                pc = jumps[pc]
        elif c == 93:    # ']'
            if tape[head] != 0:
                pc = jumps[pc]
        elif c == 44:    # ','
            tape[head] = draw() % cell_modulus
        pc += 1
        steps += 1
    return out, n_out, steps


_kernel = None


def _load_kernel():
    """Compile ``machine_kernel.pyx`` on first use; ``False`` if unavailable."""
    global _kernel
    if _kernel is None:
        try:
            import pyximport
            pyximport.install(setup_args={'include_dirs': np.get_include()},
                              language_level=3)
            from . import machine_kernel
            _kernel = machine_kernel
        except Exception:
            _kernel = False
    return _kernel


def run_programs(programs, rng: Optional[np.random.Generator] = None, *,
                 max_steps: int, tape_length: int, output_length: int,
                 cell_modulus: int = 256, compiled: Optional[bool] = None):
    """Run a batch of programs and return the learner's training rows.

    ``programs`` is a sequence of byte strings (or a uint8 array of rows),
    each starting with ``S``. Each program gets its own child generator
    spawned from ``rng``. Returns a ``[n, 1 + output_length]`` uint8 array:
    the ``O`` tag followed by the output buffer.
    """
    rng = rng if rng is not None else np.random.default_rng()
    kernel = _load_kernel() if compiled is not False else False
    if compiled and not kernel:
        raise RuntimeError('compiled machine kernel unavailable '
                           '(needs Cython and a C compiler)')
    rows = np.zeros((len(programs), PREFIX_LEN + output_length), dtype=np.uint8)
    rows[:, :PREFIX_LEN] = np.frombuffer(OUTPUT_PREFIX, dtype=np.uint8)
    for i, (prog, child) in enumerate(zip(programs, rng.spawn(len(programs)))):
        prog = bytes(np.asarray(prog, dtype=np.uint8)) if not isinstance(prog, bytes) else prog
        if kernel:
            out = kernel.run(expand(prog), child, max_steps, tape_length,
                             output_length, cell_modulus)
        else:
            out, _, _ = run_program(prog, child, max_steps, tape_length,
                                    output_length, cell_modulus)
        rows[i, PREFIX_LEN:] = out
    return rows
