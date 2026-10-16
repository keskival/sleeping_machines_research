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
    """All current winning protocol entries;individual scores and their means."""
    packet=json.loads((ROOT/'investment/public_benchmark_headlines.json').read_text())
    labels={'easytpp_taxi':'Taxi\n(EasyTPP)','easytpp_taobao':'Taobao\n(EasyTPP)','easytpp_stackoverflow':'StackOverflow\n(EasyTPP)','easytpp_retweet':'Retweet\n(EasyTPP)','easytpp_amazon':'Amazon\n(EasyTPP)','p19':'Sepsis\n(P19)','pam':'Sensors\n(PAM)','tgb_trade':'Trade\n(TGB)','tgb_wiki':'Wiki\n(TGB)','b4_wikipedia':'Wikipedia\n(B4 developed)','b4_github':'Github\n(B4 developed)','b4_stackoverflow':'Stack Overflow\n(B4 developed)','b4_lastfm':'LastFM\n(B4 frozen)'}
    fig,axes=plt.subplots(2,7,figsize=(10.8,3.8))
    for ax,row in zip(axes.flat,packet['rows']):
        ours,ref=row['values'],row['reference']
        ax.axhline(ref,color=PUB,lw=2.2,zorder=2)
        ax.scatter(range(len(ours)),ours,color=OURS,s=20,zorder=3,edgecolor='white',linewidth=.6)
        ax.scatter([(len(ours)-1)/2],[row['mean']],marker='D',color=ACCENT,s=23,zorder=4)
        lo,hi=min(ours+[ref]),max(ours+[ref]);pad=max((hi-lo)*.35,1e-6)
        ax.set_ylim(lo-pad,hi+pad);ax.set_xlim(-.8,len(ours)-.2)
        ax.set_xticks([]);ax.set_title(labels[row['key']],fontsize=8.1)
        ax.tick_params(labelsize=6.5);ax.grid(axis='x',visible=False)
        ax.set_ylabel(row['metric']+(' ↑' if row['higher_is_better'] else ' ↓'),fontsize=6.6)
        ax.text(.5,-.1,f"{row['ahead']} of {row['runs']} ahead",transform=ax.transAxes,ha='center',va='top',fontsize=7,color=OURS,fontweight='bold')
    axes.flat[-1].axis('off')
    axes.flat[-1].text(.1,.8,'Blue: seed/split\nOrange: mean\nGrey: best published\nsingle model',fontsize=8,va='top',color=MUTED)
    fig.tight_layout(w_pad=.8,h_pad=1.4);save(fig,'runs')
    fig,axes=plt.subplots(1,9,figsize=(10.8,2.6))
    for ax,row in zip(axes,packet['rows'][:9]):
        values,ref=row['values'],row['reference'];lo,hi=min(values+[ref]),max(values+[ref]);pad=(hi-lo)*.35
        ax.axhline(ref,color=PUB,lw=2.2);ax.scatter(range(len(values)),values,color=OURS,s=26,edgecolor='white',linewidth=.8)
        ax.set_ylim(lo-pad,hi+pad);ax.set_xlim(-.8,len(values)-.2);ax.set_xticks([])
        ax.set_title(labels[row['key']],fontsize=8.6);ax.tick_params(labelsize=7);ax.grid(axis='x',visible=False)
        ax.set_ylabel(row['metric']+' ↑',fontsize=7.2)
        ax.text(.5,-.1,f"{row['ahead']} of {row['runs']} ahead",transform=ax.transAxes,ha='center',va='top',fontsize=7.4,color=OURS,fontweight='bold')
    fig.tight_layout(w_pad=1.0);save(fig,'runs_foundational')



def fig_size():
    """Model size against the leading published model (parameters)."""
    rows = [('Wearables (PAM)', 46316, 873000, 'MTM'), ('Retweet', 19654, 298627, 'S2P2'), ('Taxi', 20504, 251850, 'S2P2'),
            ('Amazon', 35000, 127056, 'S2P2'), ('StackOverflow*', 29191, 30166, 'S2P2'),
            ('Taobao', 24046, 26801, 'S2P2')]
    XMAX = max(r[2] for r in rows) / 1000 * 1.45
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    for i, (name, ours, ref, refname) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y + .19, ref / 1000, height=.36, color=PUB)
        ax.barh(y - .19, ours / 1000, height=.36, color=OURS)
        ax.text(ref / 1000 + 4, y + .19, f'{refname}  {ref / 1000:,.0f}K', va='center', fontsize=7.4, color=MUTED)
        ax.text(ours / 1000 + 4, y - .19, f'ours  {ours / 1000:,.1f}K', va='center', fontsize=7.4, color=OURS,
                fontweight='bold')
        if ref / ours > 2:
            ax.text(XMAX - 5, y, (f'{ref / ours:.0f}× smaller' if ref / ours >= 9.5 else f'{ref / ours:.1f}× smaller'), va='center', ha='right', fontsize=8.6, color=INK,
                    fontweight='bold')
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=8.4)
    ax.set_xlim(0, XMAX); ax.set_xlabel('learned parameters, thousands  (↓ fewer is smaller and cheaper)')
    ax.grid(axis='y', visible=False)
    fig.tight_layout(); save(fig, 'size')


def fig_compute():
    """Per-event compute against S2P2 (multiply-accumulates per event, counted from its released layers)."""
    rows = [('Retweet', 19850, 297600), ('Taxi', 20708, 249856), ('Taobao', 24362, 26016),
            ('Amazon', 35936, 125568), ('Stack-\nOverflow*', 29660, 29216)]
    fig, ax = plt.subplots(figsize=(3.7, 2.3))
    names = [r[0] for r in rows]; x = range(len(rows))
    ax.bar([i - .2 for i in x], [r[2] / 1000 for r in rows], width=.38, color=PUB, label='S2P2 (NeurIPS 2025)')
    ax.bar([i + .2 for i in x], [r[1] / 1000 for r in rows], width=.38, color=OURS, label='ours')
    for i, (n, o, s) in enumerate(rows):
        ax.text(i, max(o, s) / 1000 + 9, (f'ours:\n1/{s / o:.0f}' if s / o > 9.5 else f'ours:\n{o / s:.2f}×') if s / o > 2 else f'ours:\n{o / s:.2f}×', ha='center', linespacing=.95,
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
