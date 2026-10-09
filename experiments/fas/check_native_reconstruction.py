"""B3 selected-native validation reconstruction and suffix causality preflight.

Consumes TRAIN/VAL and existing selected checkpoints only. It neither loads
TEST arrays nor fits a model or changes a selected configuration.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/fas'))
from sealed_registry import atomic_json,sha
from score_sealed_v2 import native_model


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True)
    parser.add_argument('--inputs',required=True);parser.add_argument('--inputs-sha256',required=True)
    args=parser.parse_args();start=time.monotonic();torch.set_num_threads(1)
    output=ROOT/'experiments/results/fas'/f'{args.tag}.json'
    if output.exists():raise FileExistsError(output)
    if sha(ROOT/args.inputs) != args.inputs_sha256:raise ValueError('Pinned preflight inputs changed')
    inputs=json.loads((ROOT/args.inputs).read_text())
    for path,expected in inputs['input_sha256'].items():
        if sha(ROOT/path) != expected:raise ValueError('Preflight input changed: '+path)
    for path,expected in inputs['source_sha256'].items():
        if sha(ROOT/path) != expected:raise ValueError('Preflight source changed: '+path)
    rows=[]
    for path in inputs['results']:
        result=json.loads((ROOT/path).read_text());a=result['args']
        if result['status'] != 'completed' or a['seed'] not in (6,7,8):
            raise ValueError('Selected complete native confirmation required')
        if any(inputs['source_sha256'].get(p) != h for p,h in result['source_sha256'].items()):
            raise ValueError('Selected native source differs')
        stage_start=time.monotonic()
        module,score=native_model(result,ROOT/result['checkpoint'])
        data=ROOT/'experiments/data/fas'/a['data']
        runs,_=module.load(data/'val_clean.npz',a['max_events']);runs=runs[:a['eval_runs']]
        _,nll=score(runs);error=abs(nll-result['val_clean_nll'])
        if not math.isfinite(nll) or error>1e-9:raise ValueError('Selected native validation fails reconstruction')
        witnesses=[run for run in runs if len(run[0])>128][:4]
        if len(witnesses)!=4:raise ValueError('Four validation suffix witnesses required')
        original,_=score(witnesses);changed=[]
        for marks,times in witnesses:
            marks=marks.copy();times=times.copy()
            marks[128:]=1+marks[128:]%45
            times[128:]=times[127]+.007+1.4*(times[128:]-times[128])
            changed.append((marks,times))
        mutated,_=score(changed)
        prefix=[i for i,n in enumerate(module.PREFIXES) if n<=128]
        causal_error=max(float(np.max(np.abs(original[rule][:,prefix]-mutated[rule][:,prefix])))
                         for rule in original)
        if not math.isfinite(causal_error) or causal_error>1e-11:raise ValueError('Future suffix changes earlier prefix scores')
        rows.append(dict(seed=a['seed'],trained_tag=a['tag'],validation_nll=nll,
                         saved_validation_nll=result['val_clean_nll'],reconstruction_error=error,
                         suffix_causality_error=causal_error,mutation_cut=128,witness_runs=4,
                         validation_runs=len(runs),wall_s=time.monotonic()-stage_start))
        print(json.dumps(rows[-1]),flush=True)
    if [r['seed'] for r in rows] != [6,7,8]:raise ValueError('All three native confirmations required')
    record=dict(status='completed',battle='B3',tag=args.tag,training=False,metrics=rows,
                decision='Certify selected native checkpoint construction and causal prefix scoring before Stage 4 TEST',
                scope='TRAIN scale/window reconstruction and unchanged clean VAL; no TEST file opened, no fit or selection',
                source_sha256=inputs['source_sha256'],input_sha256=inputs['input_sha256'],wall_s=time.monotonic()-start)
    atomic_json(output,record)


if __name__ == '__main__':main()
