#!/usr/bin/env python
r"""End-to-end LaTeX exponent table for the self-play ladder.

Loads `frontier_traj_perk.json`, rebuilds the compute envelope and the
asymptotic fit y = E + A * x^-b per corpus via `scaling_analysis.py` (SciPy
curve_fit nonlinear least squares on raw bits/byte, p0=[1, 0.5, 2],
nonnegative bounds), and writes `exponents_table.tex` in the paper's grouped
format:

    Modality / dataset & Ours $b$ & Literature $b$ & Ref.

Reported "Ours b" is -b of the fit (the positive exponent alpha) to three
decimals. `../fig1_overview/fig1.py` fits its legend exponents with the same
module and the same calls, and `check_fig1_table_consistency.py` asserts the
two agree.

Excluded: the binary corpora (astronomy, ATLAS float32, glibc rand),
arithmetic sequences, KoLMogorov DNA, and protein. Only the INCLUDE subset of
ten datasets is rendered; GROUPS keeps the full catalogue so the remaining
rows can be re-enabled by editing INCLUDE. Literature values and \citet refs
are carried from the paper.
Deps: numpy, scipy.
"""
import json
from pathlib import Path

from scaling_analysis import POOL, CTX, compute_frontier, fit_curve_power_law_with_floor

HERE = Path(__file__).resolve().parent
SRC = HERE / 'frontier_traj_perk.json'   # the scored ladder trajectory; Figure 1's panel reads the same fit
OUT = HERE / 'exponents_table.tex'
EXCLUDE = {'aitdcc_d_glibc_rand', 'aitdcc_e_atlas_float32', 'aitdcc_g_astronomy',  # binary
           'arithmetic', 'kolmogorov_dna', 'aitdcc_a_protein'}
# Rows actually rendered. Everything else in GROUPS is catalogued but hidden.
INCLUDE = {'dclm',
           'cifar10_rgb_hwc', 'cifar10_rgb_planar',
           'audio_16bit', 'audio_8bit', 'mutopia_melody_16th',
           'metamath',
           'dna',
           'aitdcc_b_c_source', 'llm_compression_python'}

def cf(alpha, beta):
    """Chinchilla-form fit (alpha, beta) -> compute exponent b = alpha*beta/(alpha+beta)
    under compute-optimal allocation; rendered with a dagger."""
    return r'$%.2g^{\dagger}$' % (alpha * beta / (alpha + beta))

def lit(*items):
    return '--'.join(items)

# group -> [(corpus_key, row label, literature b, ref)] ; groups render in this order,
# rows alphabetically by label within each group. Literature cells: direct compute
# exponents as '$v$' literals; Chinchilla-form (alpha, beta) pairs via cf() so the
# dagger conversion is computed, not hand-copied. '' renders empty, '--' as a dash.
# Sources listed in literature_references.md.
GROUPS = [
    ('Text', [
        ('dclm',                       'text (dclm)',
         # Henighan language alpha_C = 0.048 (direct); Aghajanyan text alpha=0.18, beta=0.22
         lit(r'$0.048$', cf(0.18, 0.22)), r'\citet{henighan2020scaling,aghajanyan2023scaling}'),
        ('dclm_ranked',                'text (ranked vocab)',            '', ''),
        ('kolmogorov_text',            'enwik9 text',                    '', ''),
        ('llm_compression_cc',         'web text (CC)',                  '', ''),
    ]),
    ('Images', [
        ('cifar10_rgb_planar',         'CIFAR-10 image bytes',
         # Aghajanyan image alpha=beta=0.13; Henighan image 32x32 alpha_C = 0.10 (direct)
         lit(cf(0.13, 0.13), r'$0.10$'), r'\citet{aghajanyan2023scaling}; \citet{henighan2020scaling}'),
        ('cifar10_rgb_hwc',            'CIFAR-10 (HWC interl.)',         '', ''),
    ]),
    ('Audio / speech', [
        ('audio_8bit',                 'audio 8-bit PCM',
         # Cuervo speech alpha=0.25, beta=0.24; Aghajanyan speech alpha=0.31, beta=0.24
         lit(cf(0.25, 0.24), cf(0.31, 0.24)), r'\citet{cuervo2024scaling,aghajanyan2023scaling}'),
        ('audio_16bit',                'audio 16-bit PCM',               '', ''),
        ('esc50_pcm8',                 'ESC-50 44.1 kHz PCM8',           '', ''),
        ('esc50_pcm8_11khz',           'ESC-50 11 kHz PCM8',             '', ''),
        ('musicnet_pcm8',              'MusicNet 44.1 kHz PCM8',         '', ''),
        ('musicnet_pcm8_11khz',        'MusicNet 11 kHz PCM8',           '', ''),
        ('speech_commands_pcm8',       'speech 16 kHz PCM8',             '', ''),
        ('speech_commands_pcm8_8khz',  'speech 8 kHz PCM8',              '', ''),
        ('mutopia_melody_16th',        'MIDI (Mutopia, 16th note grid)', '--', ''),
    ]),
    ('Math / formal', [
        ('metamath',                   'Metamath set.mm',
         r'$0.17$', r'\citet{henighan2020scaling}'),
        ('llm_compression_arxiv_math', 'arXiv math',                     '', ''),
    ]),
    ('Biological sequences', [
        # dnaHNet: PPL = A*C^-alpha on compute-optimal nucleotide-level models,
        # alpha = 0.01 (Transformer++) .. 0.06 (dnaHNet); direct compute exponents.
        ('dna',                        'DNA (8-symbol)',
         lit(r'$0.01$', r'$0.06$'), r'\citet{shah2026dnahnet}'),
    ]),
    ('Code', [
        ('aitdcc_b_c_source',          'AITDCC C source',
         # Aghajanyan code alpha=0.37, beta=0.32
         cf(0.37, 0.32), r'\citet{aghajanyan2023scaling}'),
        ('llm_compression_python',     'Python source (GitHub)',         '', ''),
    ]),
]

def envelope_fit(M, key):
    """Compute envelope + asymptotic fit, as fig1.py does for its scaling panel."""
    return fit_curve_power_law_with_floor(compute_frontier(M, key))

M = json.load(open(SRC))
PROV = M.pop('provenance', {})
present = {k for v in M.values() for row in v['rounds'].values() for k in row if k != 'n_seeds'}
listed = {c for _, rows in GROUPS for c, *_ in rows}
assert listed <= present, f'not in frontier data: {listed - present}'
assert present - listed == EXCLUDE, f'coverage mismatch: {present - listed - EXCLUDE} unlisted, {EXCLUDE - (present - listed)} not excluded'
assert INCLUDE <= listed, f'INCLUDE names unknown datasets: {INCLUDE - listed}'

lines = [
    r'\begin{table}[t]',
    r'  \centering',
    r'  \small',
    r'  \begin{tabular}{l r r l}',
    r'  \toprule',
    r'  Modality / dataset & Ours $b$ & Literature $b$ & Ref. \\',
]
for gi, (group, rows) in enumerate(GROUPS):
    rows = [r for r in rows if r[0] in INCLUDE]
    if not rows:
        continue
    lines.append(r'  \midrule')
    lines.append(r'  \multicolumn{4}{l}{\emph{%s}} \\' % group)
    w = max(len(lab) for _, lab, *_ in rows)
    for key, label, litcell, ref in sorted(rows, key=lambda r: r[1].casefold()):
        fit = envelope_fit(M, key)          # raises if the envelope is too short
        ours = f'${fit.alpha:.3f}$'
        lines.append(f'  {label:<{w}} & {ours} & {litcell} & {ref} \\\\')
lines += [
    r'  \bottomrule',
    r'  \end{tabular}',
    r'  \caption{',
    r'Per-modality compute exponents $b$ from fits $L(C)=A\,C^{-b}+E$. Values marked $^{\dagger}$ are derived from published Chinchilla-form fits via $b=\alpha\beta/(\alpha+\beta)$ under the compute-optimal allocation. Where a range is given, values correspond to the citations in order. Dashes indicate that the authors could not find a published exponent for that modality. Exponents from \method{} are broadly similar to exponents from pre-training on the literature, if a bit higher. A detailed description and an illustration of the datasets can be found in \cref{app:benchmarks}}',
    r'  \label{tab:scaling_exponents}',
    r'  \end{table}',
]
OUT.write_text('\n'.join(lines))
print('\n'.join(lines))
