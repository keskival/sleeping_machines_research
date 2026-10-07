#!/usr/bin/env python3
"""Figures for the investor evidence brief. Every number is read from completed result files or quoted from the
sealed-test records in experiments/B1_EASYTPP.md and experiments/B2_IRREGULAR_TS.md (published references)."""
import glob, json, statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'figures'; OUT.mkdir(exist_ok=True)
INK, MUTED, GRID = '#14213D', '#5F6B7A', '#E3E7EC'
OURS, PUB, ACCENT = '#1565C0', '#A3ABB5', '#E07A1F'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'text.color': INK, 'axes.labelcolor': MUTED,
                     'axes.edgecolor': '#C5CCD4', 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': .6, 'axes.axisbelow': True, 'axes.titlesize': 10, 'axes.titleweight': 'bold',
                     'axes.titlelocation': 'left', 'axes.titlecolor': INK})
TPP = ROOT / 'experiments/results/tpp'


def seeds(prefix):
    return [json.loads(Path(f).read_text())['test']['ll'] for f in sorted(glob.glob(str(TPP / f'{prefix}_s[0-9].json')))]


def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf'); fig.savefig(OUT / f'{name}.png', dpi=180); plt.close(fig)


def fig_runs():
    """Every independent run against the best published result."""
    p19 = json.loads((ROOT / 'experiments/results/irts/b2_final_p19_v3_summary.json').read_text())
    panels = [('Shopping\n(Taobao)', seeds('b1_final_taobao_v5'), 1.318, 'IFTPP'),
              ('Taxi rides\n(Taxi)', seeds('b1_final_taxi_v5'), 0.522, 'S2P2'),
              ('Q&A activity\n(StackOverflow)', seeds('b1_final_stackoverflow_v12'), -2.163, 'S2P2'),
              ('Social media\n(Retweet)', seeds('b1_final_retweet_v16'), -6.348, 'NHP'),
              ('Sepsis\n(P19)', [s['auprc'] for s in p19['per_split']], 0.583, 'MTM')]
    fig, axes = plt.subplots(1, 5, figsize=(7.6, 2.6))
    for ax, (name, ours, ref, refname) in zip(axes, panels):
        ax.axhline(ref, color=PUB, lw=2.2, zorder=2)
        ax.scatter(range(len(ours)), ours, color=OURS, s=26, zorder=3, edgecolor='white', linewidth=.8)
        lo, hi = min(ours + [ref]), max(ours + [ref]); pad = (hi - lo) * .35
        ax.set_ylim(lo - pad, hi + pad); ax.set_xlim(-.8, len(ours) - .2)
        ax.text(-.7, ref - pad * .12, f'best published\n({refname})', color=MUTED, fontsize=6.6, va='top')
        ax.set_xticks([]); ax.set_title(name, fontsize=8.6); ax.tick_params(labelsize=7)
        ax.set_ylabel('AUPRC, 0–1 ↑' if 'P19' in name else 'nats per event ↑', fontsize=7.2)
        ax.grid(axis='x', visible=False)
        n_above = sum(o > ref for o in ours)
        ax.text(.5, -.1, f'{n_above} of {len(ours)} ahead', transform=ax.transAxes, ha='center', va='top',
                fontsize=7.4, color=OURS, fontweight='bold')
    fig.tight_layout(w_pad=1.0); save(fig, 'runs')


def fig_size():
    """Model size against the leading published model (parameters)."""
    rows = [('Retweet', 19654, 298627, 'S2P2'), ('Taxi', 20504, 251850, 'S2P2'),
            ('Sepsis (P19)', 62681, 873000, 'MTM'), ('StackOverflow*', 29191, 30166, 'S2P2'),
            ('Taobao', 24046, 26801, 'S2P2')]
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    for i, (name, ours, ref, refname) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y + .19, ref / 1000, height=.36, color=PUB)
        ax.barh(y - .19, ours / 1000, height=.36, color=OURS)
        ax.text(ref / 1000 + 8, y + .19, f'{refname}  {ref / 1000:,.0f}K', va='center', fontsize=7.4, color=MUTED)
        ax.text(ours / 1000 + 8, y - .19, f'ours  {ours / 1000:,.1f}K', va='center', fontsize=7.4, color=OURS,
                fontweight='bold')
        if ref / ours > 2:
            ax.text(1240, y, f'{ref / ours:.0f}× smaller', va='center', ha='right', fontsize=8.6, color=INK,
                    fontweight='bold')
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=8.4)
    ax.set_xlim(0, 1250); ax.set_xlabel('learned parameters, thousands  (↓ fewer is smaller and cheaper)')
    ax.grid(axis='y', visible=False)
    fig.tight_layout(); save(fig, 'size')


def fig_compute():
    """Per-event compute against S2P2 (multiply-accumulates per event, counted from its released layers)."""
    rows = [('Retweet', 19850, 297600), ('Taxi', 20708, 249856), ('Taobao', 24362, 26016),
            ('StackOverflow*', 29660, 29216)]
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    names = [r[0] for r in rows]; x = range(len(rows))
    ax.bar([i - .2 for i in x], [r[2] / 1000 for r in rows], width=.38, color=PUB, label='S2P2 (NeurIPS 2025)')
    ax.bar([i + .2 for i in x], [r[1] / 1000 for r in rows], width=.38, color=OURS, label='ours')
    for i, (n, o, s) in enumerate(rows):
        ax.text(i, max(o, s) / 1000 + 9, f'ours:\n1/{s / o:.0f}' if s / o > 2 else f'ours:\n{o / s:.2f}×', ha='center', linespacing=.95,
                fontsize=7.6, fontweight='bold', color=INK)
    ax.set_xticks(list(x), names, fontsize=7.4); ax.set_ylabel('thousand multiply-adds per event\n(↓ lower is cheaper)', fontsize=7.6)
    ax.set_ylim(0, 350); ax.legend(frameon=False, fontsize=7.2, loc='upper right')
    ax.grid(axis='x', visible=False)
    fig.tight_layout(); save(fig, 'compute')


def fig_reasoning():
    """Learned temporal computation: event-order accuracy from 2,000 examples seen once."""
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    ax.bar([0], [99.7], color=OURS, width=.55)
    ax.bar([1], [41], color=PUB, width=.55)
    ax.text(0, 101.5, '99.7%', ha='center', fontweight='bold', fontsize=9, color=OURS)
    ax.text(1, 43, '33–41%', ha='center', fontweight='bold', fontsize=9, color=MUTED)
    ax.text(1, 20, 'range across\nmodel sizes;\nbar = best', ha='center', fontsize=7, color='white')
    ax.set_xticks([0, 1], ['Sleeping Machines', 'Transformers'], fontsize=8)
    ax.set_ylim(0, 112); ax.set_ylabel('accuracy, %  (↑ higher is better)', fontsize=7.6); ax.grid(axis='x', visible=False)
    ax.set_title('Which came first? Event order from\n2,000 examples seen once', fontsize=8.6)
    fig.tight_layout(); save(fig, 'reasoning')


if __name__ == '__main__':
    fig_runs(); fig_size(); fig_compute(); fig_reasoning()
    print(sorted(p.name for p in OUT.glob('*.pdf')))
