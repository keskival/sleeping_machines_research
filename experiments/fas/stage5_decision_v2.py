"""B3 Stage 5: registered decision from source-bound, ledger-completed scores.

Retains v1's margin, seed unanimity, verdict mapping and native rank averaging.
Repairs the reference first-seed bootstrap and missing-sample pairing.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/fas'))
from sealed_registry import atomic_json, sha
from stage5_statistics_v2 import analyze

DETECTORS = ('elapsed','tick_count','gap_z','ngram3','gap_quantile','gap_cusum','gap_robust_z','order3','timed_ngram')
PREFIXES = [32, 64, 128, 256, 512, 1024]


def load_record(path, kind, ledger, inputs):
    path = ROOT/path; inputs[str(path.relative_to(ROOT))] = sha(path)
    record = json.loads(path.read_text())
    if record['status'] != 'completed' or record['battle'] != 'B3' or record['kind'] != kind:
        raise ValueError('Completed eligible Stage 4 scorer result required')
    entries = [r for r in ledger if r.get('tag') == record['trained_tag'] and r.get('status') == 'completed']
    if len(entries) != 1 or entries[0]['identity'] != record['identity'] or entries[0]['result_sha256'] != sha(path):
        raise ValueError('Completed ledger identity/result binding differs')
    score_path = ROOT/record['per_run_scores']['path']; digest = sha(score_path)
    if digest != record['per_run_scores']['sha256'] or digest != entries[0]['scores_sha256']:
        raise ValueError('Ledger/record score digest differs')
    inputs[str(score_path.relative_to(ROOT))] = digest
    with np.load(score_path, allow_pickle=False) as archive:
        arrays = {k:archive[k] for k in archive.files}
    if arrays['prefixes'].tolist() != PREFIXES or record['prefixes'] != PREFIXES:
        raise ValueError('Registered merged prefixes required')
    for side in ('clean','faulty'):
        ids = arrays[side+'_sample_ids']
        if not np.array_equal(ids,np.arange(len(ids))):
            raise ValueError('Complete generator-order sample IDs required')
    return record, arrays


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--native',nargs=3,required=True)
    parser.add_argument('--lstm',nargs=3,required=True); parser.add_argument('--transformer',nargs=3,required=True)
    parser.add_argument('--classical',required=True); parser.add_argument('--tag',required=True)
    parser.add_argument('--out',required=True); args = parser.parse_args()
    output = ROOT/args.out
    if output.exists(): raise FileExistsError(output)
    ledger_path = ROOT/'experiments/results/fas/fas_v2_test_ledger.jsonl'
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
    inputs = {str(ledger_path.relative_to(ROOT)):sha(ledger_path)}
    loaded = []
    for kind, paths in (('native',args.native),('lstm',args.lstm),('transformer',args.transformer)):
        records = [load_record(path,kind,ledger,inputs) for path in paths]
        expected = [6,7,8] if kind == 'native' else [0,1,2]
        if [r[0]['seed'] for r in records] != expected:
            raise ValueError('Registered seed ordering required')
        loaded.extend(records)
    classical = load_record(args.classical,'classical',ledger,inputs); loaded.append(classical)
    reference_record, reference_arrays = loaded[0]
    for record, arrays in loaded:
        if record['data_manifest_sha256'] != reference_record['data_manifest_sha256']:
            raise ValueError('Scored data differs')
        for key in ('clean_sample_ids','faulty_sample_ids','kinds'):
            if not np.array_equal(arrays[key],reference_arrays[key]):
                raise ValueError('Paired sample order/fault kinds differ')
    def pair(arrays,key='total'):
        return arrays['clean_'+key][:,-1], arrays['faulty_'+key][:,-1]
    native = [pair(arrays) for _,arrays in loaded[:3]]
    baselines = {name:[pair(arrays) for _,arrays in loaded[start:start+3]]
                 for name,start in (('lstm',3),('transformer',6))}
    baselines.update({name:[pair(classical[1],name)] for name in DETECTORS})
    evidence = analyze(native, baselines, resamples=10000)
    sources = ('experiments/fas/stage5_decision_v2.py','experiments/fas/stage5_statistics_v2.py',
               'experiments/fas/sealed_registry.py')
    record = dict(evidence,status='completed',battle='B3',tag=args.tag,primary_rule='total',
                  merged_prefix=1024,per_line_prefix=512,input_sha256=inputs,
                  source_sha256={path:sha(ROOT/path) for path in sources})
    atomic_json(output,record); print(json.dumps(record),flush=True)


if __name__ == '__main__': main()
