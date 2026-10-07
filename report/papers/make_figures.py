#!/usr/bin/env python3
"""Figures for the private paper drafts, from completed result files only (no training)."""
import glob, json, os, re, statistics
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/sleeping_machines-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'report/papers/figures'; OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GRAY, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#8a8984', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': MUTED,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6, 'axes.axisbelow': True,
                     'axes.titlesize': 9.5, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})
TPP = ROOT / 'experiments/results/tpp'


def final(prefix):
    rs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(TPP / f'{prefix}_s[0-9].json')))]
    return [r['test']['ll'] for r in rs if 'test' in r]


def fig_results():
    rows = [('Taobao', final('b1_final_taobao_v5'), 1.318, 0.017, 'IFTPP'),
            ('Taxi', final('b1_final_taxi_v5'), 0.522, 0.004, 'S2P2'),
            ('StackOverflow', final('b1_final_stackoverflow_v12'), -2.163, 0.009, 'S2P2')]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for ax, (name, ours, ref, sd, refname) in zip(axes, rows):
        ax.errorbar([0], [ref], yerr=[sd], fmt='s', color=GRAY, ms=7, capsize=4, label=f'best published ({refname})')
        ax.scatter([1] * len(ours), ours, color=BLUE, s=18, alpha=.6, zorder=3)
        ax.errorbar([1], [statistics.mean(ours)], yerr=[statistics.stdev(ours)], fmt='o', color=BLUE, ms=7, capsize=4,
                    label='ours (5 seeds)', zorder=4)
        ax.set_xticks([0, 1], [refname, 'ours']); ax.set_xlim(-.6, 1.6); ax.set_title(name)
        ax.set_ylabel('TEST LL (nats/event)' if name == 'Taobao' else '')
    fig.tight_layout(); fig.savefig(OUT / 'b1_results.png', dpi=200); plt.close(fig)


def fig_compute():
    pts = [('Taxi', 20708, 0.5250, 249856, 0.522), ('Taobao', 24362, 1.3991, 26016, 1.304),
           ('StackOverflow', 36732, -2.1444, 29216, -2.163)]
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for name, ours_c, ours_ll, ref_c, ref_ll in pts:
        ax.annotate('', xy=(ours_c, ours_ll - ref_ll), xytext=(ref_c, 0), arrowprops=dict(arrowstyle='->', color=GRAY, lw=1))
        ax.scatter([ref_c], [0], color=GRAY, marker='s', s=36, zorder=3)
        ax.scatter([ours_c], [ours_ll - ref_ll], color=BLUE, s=40, zorder=3)
        ax.annotate(name, (ours_c, ours_ll - ref_ll), textcoords='offset points', xytext=(6, 4), fontsize=8, color=INK)
    ax.axhline(0, color=GRAY, lw=.8)
    ax.set_xscale('log'); ax.set_xlabel('per-event inference multiply-accumulates (log)')
    ax.set_ylabel('TEST LL − S2P2 (nats/event)')
    ax.set_title('Likelihood gain over S2P2 vs compute')
    fig.tight_layout(); fig.savefig(OUT / 'b1_compute.png', dpi=200); plt.close(fig)


def fig_audit():
    rows = [('unconstrained delayed clocks', -5.919, -6.632), ('state clock, rotation', -6.252, -6.485),
            ('state clock, rate caps', -6.246, -6.470), ('grid-safe (floored clocks)', -6.393, -6.394)]
    fig, ax = plt.subplots(figsize=(5.6, 2.5))
    for i, (name, rec, deq) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([deq, rec], [y, y], color=GRAY, lw=1.5, zorder=1)
        ax.scatter([rec], [y], color=ORANGE, s=40, zorder=3, label='as recorded' if i == 0 else None)
        ax.scatter([deq], [y], color=BLUE, s=40, zorder=3, label='dequantized within cell' if i == 0 else None)
    ax.axvline(-6.348, color=GRAY, ls='--', lw=1)
    ax.text(-6.345, len(rows) - .45, 'NHP, best published', color=MUTED, fontsize=7)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=7.5)
    ax.set_xlabel('Retweet DEV LL (nats/event)'); ax.legend(frameon=False, fontsize=7.5, loc='lower left')
    ax.set_title('Apparent gains vanish when the time grid is removed')
    fig.tight_layout(); fig.savefig(OUT / 'b1_audit.png', dpi=200); plt.close(fig)


def fig_amazon_gaps():
    import numpy as np
    d = json.load(open(ROOT / 'data/easytpp/amazon/train.json'))
    g = np.concatenate([np.diff(x['time_since_start']) for x in d])
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    ax.hist(g, bins=200, color=BLUE)
    ax.set_xlabel('gap between consecutive events (Amazon, TRAIN)'); ax.set_ylabel('count')
    ax.set_title('Two flat boxes: 31% near 0.0125, 69% on [0.70, 0.80]')
    fig.tight_layout(); fig.savefig(OUT / 'b1_amazon_gaps.png', dpi=200); plt.close(fig)


def fig_p19():
    rs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(ROOT / 'experiments/results/irts/b2_final_p19_v3_split*.json')))]
    au = [r['test']['auroc'] for r in rs]; ap = [r['test']['auprc'] for r in rs]
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.4))
    for ax, vals, ref, sd, name in ((axes[0], au, .903, .020, 'AUROC'), (axes[1], ap, .583, .053, 'AUPRC')):
        ax.errorbar([0], [ref], yerr=[sd], fmt='s', color=GRAY, ms=7, capsize=4)
        ax.scatter([1] * len(vals), vals, color=BLUE, s=18, alpha=.6, zorder=3)
        ax.errorbar([1], [statistics.mean(vals)], yerr=[statistics.stdev(vals)], fmt='o', color=BLUE, ms=7, capsize=4, zorder=4)
        ax.set_xticks([0, 1], ['MTM (best published)', 'ours']); ax.set_xlim(-.6, 1.6); ax.set_title(f'P19 TEST {name}, 5 official splits')
    fig.tight_layout(); fig.savefig(OUT / 'b2_p19.png', dpi=200); plt.close(fig)


if __name__ == '__main__':
    fig_results(); fig_compute(); fig_audit(); fig_amazon_gaps(); fig_p19()
    print(sorted(p.name for p in OUT.glob('*.png')))
