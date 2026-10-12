"""Read-only R1 scaling diagnosis from completed, protocol-matched fits.

No training, checkpoint loading, validation scoring, or reserved-data access.
Loss-floor sensitivity is a scenario analysis, not a confidence interval.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def analyze(root):
    rows = []
    protocol = None
    for size in ('64k', '256k', '1m'):
        for width in (16, 24, 32):
            date = '20261006' if (size, width) == ('1m', 16) else '20261005'
            path = root / f'curie_data_growth_tokens_{size}_b64_c16_p{width}_s6_{date}_v1.json'
            d = json.loads(path.read_text())
            if d['status'] != 'completed':
                raise ValueError(f'Incomplete evidence: {path}')
            a = d['args']
            # Only the registered data/width axes and checkpoint schedule vary.
            ignored = {'tag', 'payload', 'train_tokens', 'steps', 'eval_every'}
            key = {'args': {k: v for k, v in a.items() if k not in ignored},
                   'identity': d['identity']}
            if protocol is None:
                protocol = key
            elif protocol != key:
                raise ValueError(f'Protocol mismatch: {path}')
            rows.append({'source': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'width': width, 'unique_tokens': a['train_tokens'],
                         'nll': d['best_dev_nll'], 'parameters': d['parameters'],
                         'fitting_targets': d['presentations_total'],
                         'train_wall_s': d['train_wall_s_total'],
                         'host': d['host']})
    envelope = [min((r for r in rows if r['unique_tokens'] == n), key=lambda r: r['nll'])
                for n in sorted({r['unique_tokens'] for r in rows})]
    scenarios = []
    for width in (16, 24, 32):
        points = sorted((r for r in rows if r['width'] == width), key=lambda r: r['unique_tokens'])
        low, high = points[-2:]
        for floor in (0., 3., 5., 6.):
            if floor >= high['nll']:
                continue
            alpha = math.log((low['nll'] - floor) / (high['nll'] - floor)) / math.log(high['unique_tokens'] / low['unique_tokens'])
            prediction = floor + (high['nll'] - floor) * 4 ** (-alpha)
            scenarios.append({'width': width, 'assumed_floor_nll': floor,
                              'local_data_exponent': alpha, 'conditional_4m_nll': prediction})
    return {'status': 'completed_analysis', 'battle': 'R1', 'evidence': 'single-seed development',
            'decision': 'Retain measured width winner; do not infer a universal capacity exponent or Transformer crossover.',
            'rows': rows, 'measured_quality_envelope': envelope, 'floor_sensitivity': scenarios,
            'scope': 'Same seed, inputs, source identities and non-axis settings. Two passes; initial-inclusive DEV selection. Wall is observed training time, not audited arithmetic or cross-host normalized work. Forecasts are conditional scenarios, not measurements or intervals.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('experiments/results/token_language'))
    args = p.parse_args()
    print(json.dumps(analyze(args.root), indent=2))
