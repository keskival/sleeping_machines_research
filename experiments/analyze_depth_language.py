"""Matched data/exposure comparison of existing depth2 and depth4 R1 members."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); a = p.parse_args()
    inputs = {}; sources = {}; levels = []
    for name, reference in [('128k', 'aws_integrated_common_prior_fresh_20261012_s6_v1'),
                            ('512k', 'aws_integrated_common_prior_512k_fresh_20261012_s6_v1')]:
        records = []
        for tag in (reference, f'aws_integrated_depth4_{name}_20261012_s6_v1'):
            path = ROOT/f'experiments/results/token_language/{tag}.json'
            blob = path.read_bytes(); d = json.loads(blob); assert d['status'] == 'completed'
            inputs[str(path.relative_to(ROOT))] = hashlib.sha256(blob).hexdigest()
            for source, digest in d['source_sha256'].items():
                assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest() == digest, source
                assert source not in sources or sources[source] == digest
                sources[source] = digest
            records.append(d)
        r, c = records
        assert {k:v for k,v in r['args'].items() if k not in ('tag','depth')} == {k:v for k,v in c['args'].items() if k not in ('tag','depth')}
        assert r['identity'] == c['identity'] and r['presentations_total'] == c['presentations_total']
        rows = [dict(depth=d['args']['depth'], targets=d['presentations_total'],
                     parameters=d['parameters'],
                     initial_nll=d['curve'][0]['dev_nll'], selected_nll=d['best_dev_nll'],
                     final_nll=d['curve'][-1]['dev_nll'], training_s=d['train_wall_s_total'],
                     training_ms_per_target=1000*d['train_wall_s_total']/d['presentations_total'],
                     selected_writes_per_target=d['args']['depth']*d['args']['heads'],
                     available_receivers=d['args']['depth']*d['args']['heads']*d['args']['pool']) for d in records]
        levels.append(dict(exposure=name, rows=rows,
                           selected_depth4_gain=r['best_dev_nll']-c['best_dev_nll'],
                           final_depth4_gain=r['curve'][-1]['dev_nll']-c['curve'][-1]['dev_nll'],
                           training_wall_ratio=c['train_wall_s_total']/r['train_wall_s_total']))
    source = str(Path(__file__).relative_to(ROOT)); sources[source] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result = dict(status='completed', battle='R1', levels=levels, source_sha256=sources, input_sha256=inputs,
                  scope='First-pass single-seed depth4 member at two registered exposure levels versus completed depth2. '
                        'Same GPT2 tokens, prior, presentations, optimizer count and validation opportunities; topology and initialization differ. '
                        'Depth4 doubles selected core writes and available receivers; complete CPU training wall measured, no FLOP or iso-compute claim. '
                        'Core temporal/key-value/hard-route/counterfactual mechanisms retained; full factual backprop unchanged. '
                        'No trained-credit efficiency, family verdict, public scoring or inferred scaling exponent.')
    out = ROOT/f'experiments/results/diagnostics/{a.tag}.json'
    with out.open('x') as f: json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(levels), flush=True)


if __name__ == '__main__': main()
