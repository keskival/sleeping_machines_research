#!/usr/bin/env python3
"""Figure: native tokenized DEV NLL against count n-gram references on the same FineWeb slices (completed files only)."""
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/sleeping_machines-mpl')
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'experiments/results/token_language'
OUT = ROOT / 'report/figures/token_ngram_calibration_20261006'
BLUE, ORANGE, GRAY, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#8a8984', '#0b0b0b', '#52514e', '#e4e3df'
NATIVE = {65536: ['curie_data_growth_tokens_64k_b64_c16_p24_s6_20261005_v1.json'],
          262144: ['curie_data_growth_tokens_256k_b64_c16_p24_s6_20261005_v1.json'],
          1048576: ['curie_data_growth_tokens_1m_b64_c16_p24_s6_20261005_v1.json',
                    'curie_original_1m_p24_s7_20261006_v1.json']}
NGRAM = 'aws_token_ngram_reference_20261006T143000Z.json'


def main():
    ngram = json.loads((RES / NGRAM).read_text())
    rows = {r['train_tokens']: r['native_dev_2040'] for r in ngram['rows'] if r['train_tokens'] in NATIVE}
    xs = sorted(NATIVE)
    native = {n: [json.loads((RES / f).read_text())['best_dev_nll'] for f in NATIVE[n]] for n in xs}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.edgecolor': MUTED,
                         'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                         'grid.color': GRID, 'grid.linewidth': .6, 'axes.axisbelow': True})
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    series = [('Kneser–Ney bigram (counts, seconds)', [rows[n]['bigram_kn']['nll'] for n in xs], GRAY, 's'),
              ('Kneser–Ney trigram (counts, seconds)', [rows[n]['trigram_kn']['nll'] for n in xs], ORANGE, '^'),
              ('Ours: P24 native, 3.16M parameters, 2 passes', [sum(native[n]) / len(native[n]) for n in xs], BLUE,
               'o')]
    for label, ys, color, marker in series:
        ax.plot(xs, ys, color=color, lw=2, marker=marker, ms=7, mec='white', mew=1.5, label=label, zorder=3)
    for y in native[1048576]:
        ax.plot([1048576], [y], 'o', color=BLUE, ms=5, alpha=.55, zorder=4)
    for n, y, offset in zip(xs, series[2][1], [(-8, -14), (-10, 8), (-44, 4)]):
        ax.annotate(f'{y:.3f}', (n, y), textcoords='offset points', xytext=offset, color=INK, fontsize=7.5)
    ax.set_xscale('log', base=2)
    ax.set_xticks(xs, ['64K', '256K', '1M'])
    ax.set_xlabel('TRAIN tokens (GPT-2 BPE, FineWeb)')
    ax.set_ylabel('DEV NLL, nats/token (lower is better)')
    ax.set_title('Native tokenized language tracks a bigram on the same 2,040 DEV targets', loc='left',
                 fontsize=9.5, fontweight='bold', color=INK)
    ax.legend(frameon=False, fontsize=7.5, loc='upper right')
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(f'{OUT}.{ext}', dpi=200)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
