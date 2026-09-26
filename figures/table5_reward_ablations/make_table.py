#!/usr/bin/env python3
"""Emit Table 3 (generator reward ablations) as LaTeX from the scored trajectories.

Reads the five ablation arms from `frontier_traj_reward.json`, the unablated
self-play rung from `../table2_scaling_exponents/frontier_traj_perk.json`, and
the fixed-prior rung from `../fig2_transfer_across_modalities/data/`, all at
the 1M rung and the final training round. Each entry is the bits/byte of the
4-seed probability ensemble (fewer seeds if an arm has fewer). Writes
`reward_frontier_summary.csv` and `reward_arms.tex`.

    python make_table.py
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAPTION = "\\vspace{2em}\n\\caption{\n\\textbf{Reward ablations of the self-play generator} at the 1M parameter model. Each entry shows the validation loss in bits per byte of the 4-seed ensemble on 256 held-out sequences per dataset, scored from the final-round checkpoints. Every ablation changes exactly one property of the canonical reward $r_i = {|\\langle P\\odot\\nabla L(y_i;\\theta_{\\text{now}}),\\, \\theta_{\\text{past}}-\\theta_{\\text{now}}\\rangle|}$ with $\\theta_{\\text{past}}=\\theta_{\\lfloor e/2\\rfloor}$ (\\emph{None}). \\emph{Signed} drops the absolute value, \\emph{shuffle} permutes the rewards across the program pool, \\emph{last\\_step} uses the one-step window $\\theta_{\\text{past}}=\\theta_{e-1}$, \\emph{loss\\_delta} uses the realized progress $L_i(\\theta_{\\text{pre}})-L_i(\\theta_{\\text{post}})$, \\emph{negate} flips the reward's sign, and \\emph{uniform} removes the generator entirely (i.i.d.\\ uniform programs). On the majority of datasets, the canonical reward is best, or nearly best, and it often has lower variance as well. \\\\\n$^{a}$ One loss\\_delta seed stopped at round 2587 and is excluded; its ensemble is over 3 seeds. \\\\\n$^{b}$ last\\_step and loss\\_delta are bimodal across seeds (e.g.\\ dclm single-seed 6.4 / 6.5 vs 10.6 / 11.1 for last\\_step; 7.0 vs 9.8 / 12.3 for loss\\_delta); their ensembles average over the diverged seeds.\n}"
FIG = HERE.parent
RUNG, ROUND, K = "d128h2L4", 8191, 4

# (column key, header) in table order
COLUMNS = [("none", "None"), ("uniform", "uniform"), ("signed", "signed"),
           ("shuffle", "shuffle"), ("last_step", r"last\_step$^{\,b}$"),
           ("loss_delta", r"loss\_delta$^{\,a,b}$"), ("negate", "negate")]

# (corpus key, row label) in table order
ROWS = [("dclm", "text (dclm)"), ("metamath", "Metamath"),
        ("aitdcc_b_c_source", "C source"), ("dna", "DNA (8-symbol)"),
        ("arithmetic", "arithmetic"), ("audio_8bit", "audio 8-bit PCM"),
        ("audio_16bit", "audio 16-bit PCM"),
        ("mutopia_melody_16th", "melody (Mutopia)"),
        ("cifar10_rgb_planar", "CIFAR-10 (planar)"),
        ("aitdcc_d_glibc_rand", "random bytes")]

def final_from_traj(entry):
    """{corpus: (bpb, K used)} at the last scored round of one trajectory."""
    rounds = entry["rounds"]
    row = rounds[str(max(int(r) for r in rounds))]
    out = {}
    for corpus, per_k in row.items():
        if corpus == "n_seeds":
            continue
        k = min(K, max(int(x) for x in per_k))
        out[corpus] = (per_k[str(k)], k)
    return out


def final_from_csv(path):
    per_k = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        if r["rung"] == RUNG and int(r["round"]) == ROUND:
            per_k[r["corpus"]][int(r["K"])] = float(r["bpb"])
    return {c: (d[min(K, max(d))], min(K, max(d))) for c, d in per_k.items()}


def main() -> None:
    arms = json.load(open(HERE / "frontier_traj_reward.json"))
    arms.pop("provenance", None)
    ladder = json.load(open(FIG / "table2_scaling_exponents" / "frontier_traj_perk.json"))
    final = {arm: final_from_traj(entry) for arm, entry in arms.items()}
    final["none"] = final_from_traj(
        {"rounds": {str(ROUND): ladder[RUNG]["rounds"][str(ROUND)]}})
    final["uniform"] = final_from_csv(
        FIG / "fig2_transfer_across_modalities" / "data" / "uniform_frontier_perk.csv")

    width = 27
    lines = ["", r"\begin{table}[t]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{4.5pt}",
             r"\begin{tabular}{l" + "c" * len(COLUMNS) + "}", r"\toprule",
             r" & \multicolumn{%d}{c}{ablation} \\" % len(COLUMNS),
             r"\cmidrule(lr){2-%d}" % (len(COLUMNS) + 1),
             "dataset & " + " & ".join(h for _, h in COLUMNS) + r" \\", r"\midrule"]
    for corpus, label in ROWS:
        vals = [round(final[arm][corpus][0], 2) for arm, _ in COLUMNS]
        best = min(vals)   # ties at two decimals share the bold
        cells = [(r"\textbf{%.2f}" % v) if v == best else f"{v:.2f}" for v in vals]
        w = 20 if label == "random bytes" else width  # historical padding
        lines.append(f"{label:<{w}} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", CAPTION,
              r"\label{tab:reward_ablations}", r"\end{table}"]
    (HERE / "reward_arms.tex").write_text("\n".join(lines) + "\n")
    print(HERE / "reward_arms.tex")


if __name__ == "__main__":
    main()
