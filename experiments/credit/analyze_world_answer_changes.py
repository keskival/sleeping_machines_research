"""R1 causal replay: are native route outputs changing the persistence answer?"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/causal_induction'))
from public_stream import records


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); a = p.parse_args()
    parent = ROOT/'experiments/results/causal_induction/curie_causal_nonlinear_20261012_v2_prepare.json'
    manifest = json.loads(parent.read_text()); expected = {r['path']:r['sha256'] for r in manifest['records']}
    inputs = {str(parent.relative_to(ROOT)):sha(parent)}; output = []
    for arm in ('learned', 'prior', 'exact'):
        path = ROOT/f'experiments/results/causal_induction/curie_world_credit_20261012_v2_{arm}_reuse.json'
        d = json.loads(path.read_text()); assert d['status'] == 'completed'; inputs[str(path.relative_to(ROOT))] = sha(path)
        cases = []
        for row in d['rows']:
            public = ROOT/row['path']; assert sha(public) == expected[row['path']]; inputs[row['path']] = sha(public)
            bits = {}; issued = {}; feedback = {}
            for event in records(public):
                if event['kind'] == 'event': bits[event['node']] = event['value']
                elif event['kind'] == 'query': issued[event['query_id']] = (bits.get(event['node'], 0), event['time'])
                else: feedback[event['query_id']] = (event['target'], event['time'])
            flips = good = bad = baseline_correct = actor_correct = 0
            for trace in row['issued_trace']:
                bit, time = issued[trace['query_id']]; label, arrived = feedback[trace['query_id']]
                assert time == trace['issued_time'] and arrived == trace['feedback_time'] and label == trace['target']
                prediction = int(trace['probability'] >= .5)
                flipped = prediction != bit
                flips += int(flipped); good += int(flipped and prediction == label); bad += int(flipped and prediction != label)
                baseline_correct += int(bit == label); actor_correct += int(prediction == label)
            cases.append(dict(path=row['path'], depth=row['depth'], variant=row['variant'], targets=row['targets'],
                              answer_changes=flips, beneficial_changes=good, harmful_changes=bad,
                              persistence_correct=baseline_correct, actor_correct=actor_correct))
        output.append(dict(arm=arm, targets=sum(c['targets'] for c in cases),
                           answer_changes=sum(c['answer_changes'] for c in cases),
                           beneficial_changes=sum(c['beneficial_changes'] for c in cases),
                           harmful_changes=sum(c['harmful_changes'] for c in cases),
                           cases=cases))
    source = str(Path(__file__).relative_to(ROOT)); reader = 'experiments/causal_induction/public_stream.py'
    result = dict(status='completed', battle='B5/R1', rows=output, input_sha256=inputs,
                  source_sha256={source:sha(Path(__file__)), reader:sha(ROOT/reader)},
                  scope='Read-only replay of byte-matched public observations with saved issued forecasts. Last observed target bit is captured at query time, never reconstructed from later events. Counts describe factual answer changes relative to persistence; they do not identify unexecuted route effects or full future learning. No new fit or oracle access.')
    out = ROOT/f'experiments/results/diagnostics/{a.tag}.json'
    with out.open('x') as f: json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps([{k:v for k,v in r.items() if k!='cases'} for r in output]), flush=True)


if __name__ == '__main__': main()
