"""R1 fixed-prior fresh-acquisition curves; completed evidence only."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def canonical(row):
    return {k: v for k, v in row.items() if k not in ('wall_s', 'learner_fingerprint')}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    p.add_argument('--stage', choices=('2m', '8m'), required=True)
    p.add_argument('--pool', type=int, choices=(4, 8), default=4)
    a = p.parse_args()
    inputs, sources, rows = {}, {}, []

    def load(tag):
        path = ROOT / f'experiments/results/token_language/{tag}.json'
        blob = path.read_bytes(); d = json.loads(blob)
        assert d['status'] == 'completed'
        inputs[str(path.relative_to(ROOT))] = hashlib.sha256(blob).hexdigest()
        for name, digest in d['source_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
            assert name not in sources or sources[name] == digest
            sources[name] = digest
        assert d['args']['train_tokens'] == 262144
        assert d['args']['exposure'] == 'fresh'
        assert d['exposure_protocol']['unique_admitted_tokens'] > d['presentations_total']
        assert d['max_rss_kb'] < 1500000
        return d

    pools = (4, 8) if a.stage == '2m' else (a.pool,)
    for pool in pools:
        previous = (f'aws_integrated_{"window_replay" if pool == 4 else "clock_capacity8"}_512k_20261012_s6_v1'
                    if a.stage == '2m' else f'aws_integrated_fresh_scaling_2m_pool{pool}_20261012_s6_v1')
        before = load(previous)
        current = load(f'aws_integrated_fresh_scaling_{a.stage}_pool{pool}_20261012_s6_v1')
        ignored = ('tag', 'steps', 'clock_reference_pool')
        assert {k:v for k,v in before['args'].items() if k not in ignored} == {k:v for k,v in current['args'].items() if k not in ignored}
        for key in ('train_sha256', 'dev_sha256'):
            assert before['identity'][key] == current['identity'][key]
        prefix = [r for r in current['curve'] if r['step'] <= before['args']['steps']]
        assert [canonical(r) for r in prefix] == [canonical(r) for r in before['curve']], 'Acquisition prefix changed'
        rows.append(dict(pool=pool, targets=current['presentations_total'],
            unique_admitted_tokens=current['exposure_protocol']['unique_admitted_tokens'],
            prior_tokens=262144, parameters=current['parameters'],
            selected_nll=current['best_dev_nll'], final_nll=current['curve'][-1]['dev_nll'],
            previous_final_nll=before['curve'][-1]['dev_nll'],
            final_learning_gain=before['curve'][-1]['dev_nll']-current['curve'][-1]['dev_nll'],
            training_s=current['train_wall_s_total'],
            ms_per_target=1000*current['train_wall_s_total']/current['presentations_total'],
            max_rss_kb=current['max_rss_kb'], exact_recorded_acquisition_prefix=True,
            selected_writes_per_target=4, scored_keys_per_target=4*pool))
    selected = rows[0]
    if len(rows) == 2:
        old, new = rows
        if new['final_nll'] < old['final_nll'] and new['selected_nll'] < old['selected_nll'] and new['training_s'] <= 1.05*old['training_s']:
            selected = new
    scope = ('Seed6 fixed-prior GPT2 FineWeb fresh-acquisition development. Same initial TRAIN262144-token prior, '
        'credit16, lane/block boundaries and DEV interval; every256 updates evaluated. '
        'Recorded earlier acquisition prefixes must agree exactly within each member. '
        'Final NLL is the acquisition-curve measure; selected NLL also shown with its increasing selection opportunities. '
        'Pool-dependent initialization differs between members. Complete measured CPU fitting wall, '
        'all-key scoring and optimizer included. Public validation/reserved4M untouched; no fitted exponent or external-model comparison.')
    source = str(Path(__file__).relative_to(ROOT)); sources[source] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result = dict(status='completed', battle='R1', stage=a.stage, rows=rows, selected_pool=selected['pool'],
        continue_larger=selected['final_learning_gain'] > .05 and selected['ms_per_target'] < 1.5,
        input_sha256=inputs, source_sha256=sources, scope=scope)
    with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_sha256','source_sha256')}), flush=True)


if __name__ == '__main__': main()
