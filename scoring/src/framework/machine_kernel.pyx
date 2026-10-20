# cython: language_level=3
"""Compiled kernel for ``machine.run_programs``; same semantics as ``run_program``."""

import numpy as np
cimport numpy as np
from cpython.pycapsule cimport PyCapsule_GetPointer
from numpy.random cimport bitgen_t

np.import_array()


def run(bytes code, object rng, long max_steps, long tape_length,
        long output_length, long cell_modulus=256):
    """Run an already expanded program; returns the int64 output buffer."""
    cdef bitgen_t *bg = <bitgen_t *>PyCapsule_GetPointer(
        rng.bit_generator.capsule, "BitGenerator")
    cdef np.ndarray[np.int64_t, ndim=1] tape = np.zeros(tape_length, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] out = np.zeros(output_length, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] jumps = np.arange(len(code), dtype=np.int64)
    cdef const unsigned char[:] prog = code
    cdef long n = len(code), head = 0, pc = 0, n_out = 0, steps = 0
    cdef long cell
    cdef unsigned char c

    stack = []
    for i in range(n):
        if prog[i] == 91:
            stack.append(i)
        elif prog[i] == 93 and stack:
            j = stack.pop()
            jumps[i] = j
            jumps[j] = i

    while steps < max_steps and pc < n and n_out < output_length:
        c = prog[pc]
        if c == 62:
            head = head + 1 if head + 1 < tape_length else 0
        elif c == 60:
            head = head - 1 if head > 0 else tape_length - 1
        elif c == 43:
            cell = tape[head] + 1
            tape[head] = 0 if cell == cell_modulus else cell
        elif c == 45:
            cell = tape[head] - 1
            tape[head] = cell_modulus - 1 if cell < 0 else cell
        elif c == 46:
            out[n_out] = tape[head]
            n_out += 1
        elif c == 91:
            if tape[head] == 0:
                pc = jumps[pc]
        elif c == 93:
            if tape[head] != 0:
                pc = jumps[pc]
        elif c == 44:
            tape[head] = <np.int64_t>(bg.next_uint32(bg.state) % <np.uint64_t>cell_modulus)
        pc += 1
        steps += 1
    return out
