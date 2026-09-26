#!/usr/bin/env python3
"""Derive Table 1 (math_table_paper.tex) from the full math_table.tex
written by prior_analysis.py: keep family / program / output / earliest
round / universal-prior expectation."""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
rows = []
for line in (HERE / "math_table.tex").read_text().splitlines():
    m = re.match(r"^(\w+)\s+&(.*)\\\\\s*$", line)
    if m and "makecell" not in line:
        cells = [c.strip() for c in ([m.group(1)] + m.group(2).split("&"))]
        if len(cells) == 7:
            rows.append([cells[0], cells[1], cells[2], cells[4], cells[6]])
assert len(rows) == 5, rows

HEADER = "\n" + r"""\begin{table}[ht]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lllcc}
\toprule
\makecell{\textbf{Family}\\\textbf{(mod 256)}} &
\textbf{Example program} & 
\textbf{Its output} &
\makecell{\textbf{Earliest}\\\textbf{round}} &
\makecell{$\mathbb{E}[\text{first round}]$\\\textbf{(univ.\ prior)}} \\
\midrule
"""
FOOTER = r"""\bottomrule
\end{tabular}
\vspace{6pt}
\caption{\textbf{Program families with recognizable mathematical structure discovered by the generator during self-play.}  
\emph{Earliest round} gives the earliest round in which a member of the family first appears during training, while \emph{univ. prior} gives the expected first appearance round if programs are drawn from the universal prior, including the added primitives. See \cref{apx:emergent-math-structure} for additional details.}
\label{tab: discovered-math-structure}
\end{table}
"""
body = "".join(
    f"{r[0]:<10} & {r[1]:<27} & {r[2]:<21} & {r[3]:<3} & {r[4]} \\\\\n"
    for r in rows)
(HERE / "math_table_paper.tex").write_text(HEADER + body + FOOTER)
print("wrote math_table_paper.tex")
