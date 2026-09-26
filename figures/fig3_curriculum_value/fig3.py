"""Figure 3: training value of later generator checkpoints, two panels.

(a) Epiplexity of the learner's training loss (median filter, running-minimum
    envelope, area above the envelope's final value; nats x rounds). One dot
    per seed, line through the 8-seed mean. A seed far above its arm-mates is
    drawn at the top edge with its value printed.
(b) Learner validation bits/byte at the final round (4095) on Text (dclm),
    Audio (audio_16bit), and Images (cifar10_rgb_planar): mean over all
    C(8,4) = 70 four-seed probability ensembles, with +-1 SD bars.

Arms: g_0 (untrained generator) and T in {256, 512, 1024, 2048, 4096}.
Inputs: data/training_losses.npz, data/ood_validation/<arm>/round_004095.json,
data/subset_ensembles.csv. Writes fig3.{png,pdf} and data/plotted_values.csv.

    python fig3.py
"""
import csv
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from numpy.lib.stride_tricks import sliding_window_view

N_SEEDS, SUBSETS, ROUND = 8, 4, 4095
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = f'{HERE}/data'
ARMS = [('G_0', 'G0', 0), ('T256', 'T256', 256), ('T512', 'T512', 512),
        ('T1024', 'T1024', 1024), ('T2048', 'T2048', 2048), ('T4096', 'T4096', 4096)]
SERIES = [('Text', 'dclm', '#3b3b3b'), ('Audio', 'audio_16bit', '#7b6a8f'),
          ('Images', 'cifar10_rgb_planar', '#a88a3a')]
C_TEXT = SERIES[0][2]


def robust_area(y, w=33):
    med = np.median(sliding_window_view(np.pad(y, w // 2, mode='edge'), w), axis=1)
    env = np.minimum.accumulate(med)
    return float(np.trapezoid(env - env[-1], np.arange(len(y))))


# ---- panel (a): epiplexity per seed --------------------------------------------
L = dict(np.load(f'{DATA}/training_losses.npz'))
area = {}
for arm, pre, T in ARMS:
    keys = sorted(k for k in L if k.startswith(pre + '_seed-'))[:N_SEEDS]
    assert len(keys) == N_SEEDS and not any(np.isnan(L[k]).any() for k in keys), (arm, keys)
    area[arm] = np.array([robust_area(L[k]) for k in keys])

# ---- panel (b): validation at the final round -----------------------------------
seed_bpb, ens_bpb = {}, {}
for arm, pre, T in ARMS:
    d = json.load(open(f'{DATA}/ood_validation/{pre}/round_{ROUND:06d}.json'))
    assert d['round'] == ROUND and d['n_seeds'] == N_SEEDS, (arm, d['n_seeds'], N_SEEDS)
    seed_bpb[arm] = {c: np.array(list(d['bpb_per_seed'][c].values())) for _, c, _ in SERIES}
    ens_bpb[arm] = {c: d['bpb'][c][str(N_SEEDS)] for _, c, _ in SERIES}

# Subset-ensemble statistics need per-token probabilities for every seed, which
# are large; they are shipped precomputed (subset_ensembles.py regenerates them).
sub_mean, sub_sd = {}, {}
with open(f'{DATA}/subset_ensembles.csv') as fh:
    for row in csv.DictReader(fh):
        for _, c, _ in SERIES:
            sub_mean[row['arm'], c] = float(row[f'{c}_mean'])
            sub_sd[row['arm'], c] = float(row[f'{c}_sd'])

# ---- figure ----------------------------------------------------------------------
FS = dict(title=22, label=19, tick=16, legend=15, letter=20, note=14)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
                     'mathtext.fontset': 'dejavuserif'})
fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 7.4), dpi=170)
fig.patch.set_facecolor('white')
xs = np.arange(len(ARMS))
labels = ['$g_0$'] + [f'$g_{{{T:,}}}$' for _, _, T in ARMS[1:]]

# (a): a seed more than 2.5x the median of its arm-mates and above every other
# seed is drawn off-scale so the y-range fits the bulk.
m = np.array([area[a].mean() for a, _, _ in ARMS])
s = np.array([area[a].std(ddof=1) for a, _, _ in ARMS])
allv = np.concatenate([area[arm] for arm, _, _ in ARMS])


def _keep(arm):
    out = []
    for k, v in enumerate(area[arm]):
        others_arm = np.delete(area[arm], k)
        others_all = allv[allv != v]
        out.append(not (v > 2.5 * np.median(others_arm) and v > 1.1 * others_all.max()))
    return np.array(out)


keep = {arm: _keep(arm) for arm, _, _ in ARMS}
bulk_max = max(area[arm][keep[arm]].max() for arm, _, _ in ARMS)
clip_top = 1.12 * max(bulk_max, max(np.mean(area[arm]) for arm, _, _ in ARMS))
for i, (arm, _, _) in enumerate(ARMS):
    off = np.linspace(-.07, .07, N_SEEDS)
    k = keep[arm]
    axL.scatter(i + off[k], area[arm][k], s=16, color=C_TEXT, alpha=.45, lw=0, zorder=3,
                label='individual seed' if i == 0 else None)
    for j in np.flatnonzero(~k):
        already = 'off-scale seed (value shown)' in axL.get_legend_handles_labels()[1]
        axL.scatter([i + off[j]], [clip_top * 0.985], s=46, marker='^', color='#b0413e', lw=0,
                    zorder=2, clip_on=False, label=None if already else 'off-scale seed (value shown)')
        axL.annotate(f'{area[arm][j]:,.0f}', (i + off[j], clip_top * 0.985), xytext=(-6, 0),
                     textcoords='offset points', ha='right', va='center', fontsize=FS['note'],
                     color='#b0413e')
axL.plot(xs, m, color=C_TEXT, lw=1.6, marker='o', ms=5, mfc='white', mew=1.4, zorder=4,
         label=f'mean of {N_SEEDS} seeds')
axL.set_ylabel('Epiplexity  (nats $\\cdot$ rounds)', fontsize=FS['label'])
axL.set_title('Curriculum structure content', fontsize=FS['title'], pad=62)
axL.set_ylim(0, clip_top)

# (b)
rows = {}
for name, c, col in SERIES:
    sm = np.array([seed_bpb[a][c].mean() for a, _, _ in ARMS])
    ss = np.array([seed_bpb[a][c].std(ddof=1) for a, _, _ in ARMS])
    ke = np.array([ens_bpb[a][c] for a, _, _ in ARMS])
    km = np.array([sub_mean[a, c] for a, _, _ in ARMS])
    ks = np.array([sub_sd[a, c] for a, _, _ in ARMS])
    axR.errorbar(xs, km, yerr=ks, fmt='none', ecolor=col, elinewidth=1.4, capsize=7, capthick=1.6,
                 zorder=3, alpha=.9)
    axR.plot(xs, km, color=col, lw=1.6, marker='o', ms=6, mfc='white', mew=1.4, zorder=4, label=name)
    rows[c] = (sm, ss, ke)
axR.set_ylabel('Val BPB', fontsize=FS['label'])
axR.set_title('Learner OOD validation', fontsize=FS['title'], pad=62)

hR, lR = axR.get_legend_handles_labels()
hR += [Line2D([], [], color='#777777', lw=1.6, marker='o', ms=6, mfc='white', mew=1.4),
       axR.errorbar([np.nan], [np.nan], yerr=[1], fmt='none', ecolor='#777777', elinewidth=1.4,
                    capsize=7, capthick=1.6)]
lR += [f'mean of {SUBSETS}-seed subsets', '$\\pm$1 SD over subsets']
axR.legend(hR, lR, ncol=3, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0),
           fontsize=FS['legend'], columnspacing=1.0, handlelength=1.8, handletextpad=0.5)
axL.legend(ncol=2, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0),
           fontsize=FS['legend'], columnspacing=1.2, handletextpad=0.5)
for ax, letter in ((axL, 'a'), (axR, 'b')):
    ax.set_facecolor('white')
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=FS['tick'])
    ax.tick_params(axis='y', labelsize=FS['tick'])
    ax.set_xlabel('Generator-training endpoint $g_T$', fontsize=FS['label'])
    ax.grid(True, color='#e3e3e3', lw=.7)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.text(-0.14, 1.27, f'({letter})', transform=ax.transAxes, fontsize=FS['letter'],
            fontweight='bold', va='top')
fig.tight_layout()
for ext in ('png', 'pdf'):
    out = f'{HERE}/fig3.{ext}'
    fig.savefig(out, facecolor='white')
    print('saved', out)

with open(f'{DATA}/plotted_values.csv', 'w') as f:
    f.write('arm,T,' + ','.join(f'epiplexity_seed{i + 1}' for i in range(N_SEEDS))
            + ',epiplexity_mean,epiplexity_sd,'
            + ','.join(f'{c}_seed_mean,{c}_seed_sd,{c}_ensemble_K{N_SEEDS},{c}_subset_mean,{c}_subset_sd'
                       for _, c, _ in SERIES) + '\n')
    for i, (arm, _, T) in enumerate(ARMS):
        f.write(f'{arm},{T},' + ','.join(f'{a:.3f}' for a in area[arm]) + f',{m[i]:.3f},{s[i]:.3f},'
                + ','.join(f'{rows[c][0][i]:.5f},{rows[c][1][i]:.5f},{rows[c][2][i]:.5f},'
                           f'{sub_mean[arm, c]:.4f},{sub_sd[arm, c]:.4f}' for _, c, _ in SERIES) + '\n')
print('wrote', f'{DATA}/plotted_values.csv')

print(f"\n{'arm':<7}{'epi_mean':>9}{'sd':>7} |"
      + ''.join(f"{c[:5] + '_sub':>11}{'sd':>6}{'K' + str(N_SEEDS):>7}" for _, c, _ in SERIES))
for i, (arm, _, T) in enumerate(ARMS):
    print(f"{arm:<7}{m[i]:9.1f}{s[i]:7.1f} |"
          + ''.join(f"{sub_mean[arm, c]:11.3f}{sub_sd[arm, c]:6.3f}{rows[c][2][i]:7.3f}" for _, c, _ in SERIES))
