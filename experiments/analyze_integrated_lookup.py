"""R1 paired quality/work decision from completed integrated token fits."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(); p.add_argument('--tag',required=True)
    p.add_argument('--ordinary',required=True); p.add_argument('--batched',required=True)
    a=p.parse_args(); folder=ROOT/'experiments/results/token_language'
    records=[]; inputs={}
    for tag in (a.ordinary,a.batched):
        path=folder/f'{tag}.json'; blob=path.read_bytes(); d=json.loads(blob)
        assert d['status']=='completed'
        for name,digest in d['source_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
        inputs[str(path.relative_to(ROOT))]=hashlib.sha256(blob).hexdigest()
        records.append(d)
    baseline,candidate=records
    assert {k:v for k,v in baseline['args'].items() if k!='tag'} == {k:v for k,v in candidate['args'].items() if k!='tag'}
    assert baseline['identity']==candidate['identity']
    assert baseline['host']==candidate['host']
    assert baseline['presentations_total']==candidate['presentations_total']
    assert [r['step'] for r in baseline['curve']]==[r['step'] for r in candidate['curve']]
    assert baseline['curve'][0]['dev_nll']==candidate['curve'][0]['dev_nll']
    bq=min(r['dev_nll'] for r in baseline['curve'])
    cq=min(r['dev_nll'] for r in candidate['curve'])
    ratio=baseline['train_wall_s_total']/candidate['train_wall_s_total']
    rows=[]
    for label,d in zip(('ordinary','batched'),records):
        rows.append(dict(arm=label,selected_dev_nll=min(r['dev_nll'] for r in d['curve']),
            final_dev_nll=d['curve'][-1]['dev_nll'],fitting_targets=d['presentations_total'],
            unique_train_tokens=d['args']['train_tokens'],parameters=d['parameters'],
            training_wall_s=d['train_wall_s_total'],engine_wall_s=d['wall_s'],
            process_wall_s=d['profile']['process_wall_s'],peak_rss_kib=d['max_rss_kb'],
            stage_wall_s=d['profile']['stage_wall_s']))
    result=dict(status='completed',battle='R1',rows=rows,quality_gain=bq-cq,
        ordinary_over_batched_training_wall=ratio,
        development_gate_passed=cq<=bq and ratio>=1/.95,
        input_sha256=inputs,source_sha256={'experiments/analyze_integrated_lookup.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        scope='One seed; same CPU host/data/updates/selection opportunities and integrated mechanisms. Initial-inclusive quality. Complete measured training wall includes core, alternative replay, head, backward, clipping, diagnostics and optimizer; evaluation/serialization separately in engine wall. Process wall includes acquisition/hash preparation. No fitting-FLOP, energy, Transformer or asymptotic-exponent claim. Exact real-arithmetic lookup equivalence; changed floating-point accumulation can change long hard-route trajectories.')
    path=folder/f'{a.tag}.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
