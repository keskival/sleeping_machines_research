"""R1 paired quality/work decision from completed integrated token fits."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(); p.add_argument('--tag',required=True)
    p.add_argument('--ordinary',required=True); p.add_argument('--batched',required=True)
    p.add_argument('--fp32-reference',required=True); p.add_argument('--continue-seeds',action='store_true')
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
    max_curve_error=max(abs(x['dev_nll']-y['dev_nll']) for x,y in zip(baseline['curve'],candidate['curve']))
    rp=folder/f'{a.fp32_reference}.json'; rb=rp.read_bytes(); reference=json.loads(rb)
    assert reference['status']=='completed'
    assert {k:v for k,v in reference['args'].items() if k not in ('tag','precision')} == {k:v for k,v in candidate['args'].items() if k not in ('tag','precision')}
    for key in ('train_sha256','dev_sha256'):assert reference['identity'][key]==candidate['identity'][key]
    assert reference['host']==candidate['host']
    for name,digest in reference['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    inputs[str(rp.relative_to(ROOT))]=hashlib.sha256(rb).hexdigest()
    fp32_quality=min(r['dev_nll'] for r in reference['curve'])
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
        development_gate_passed=cq<=bq+1e-8 and max_curve_error<=1e-8 and ratio>=1/.95,
        max_dev_curve_error=max_curve_error,precision_parity_tolerance=1e-8,
        versus_fp32=dict(quality_gain=fp32_quality-cq,ordinary_over_batched_training_wall=reference['train_wall_s_total']/candidate['train_wall_s_total']),
        input_sha256=inputs,source_sha256={'experiments/analyze_integrated_lookup_precision_v2.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        scope='One seed; same CPU host/data/updates/selection opportunities and integrated mechanisms. Initial-inclusive quality. Complete measured training wall includes core, alternative replay, head, backward, clipping, diagnostics and optimizer; evaluation/serialization separately in engine wall. Process wall includes acquisition/hash preparation. No fitting-FLOP, energy, Transformer or asymptotic-exponent claim. Exact real-arithmetic lookup equivalence; changed floating-point accumulation can change long hard-route trajectories.')
    path=folder/f'{a.tag}.json'
    if path.exists():raise FileExistsError(path)
    result['useful_quality_work_gate_passed'] = result['development_gate_passed'] and result['versus_fp32']['quality_gain'] >= 0 and result['versus_fp32']['ordinary_over_batched_training_wall'] >= 1/.95
    if a.continue_seeds and result['useful_quality_work_gate_passed']:
        import ast, subprocess
        deps={'experiments/horizon_token_language_engine.py','experiments/batched_token_episodes.py'}
        for n in ast.walk(ast.parse((ROOT/'experiments/horizon_token_language_engine.py').read_text())):
            if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith('sleeping_machines/') and n.value.endswith('.py'):deps.add(n.value)
        cmd=[str(ROOT/'.venv-docker/bin/python'),str(ROOT/'experiments/tpp/submit.py'),'--slot','2','--battle','R1','--source','experiments/profile_integrated_language_v3.py','--result-dir','experiments/results/token_language','--timeout','600','--rss-kb','1500000','--purpose','R1: unchanged precision-stabilized integrated lookup confirmations, admitted only after seed6 full-curve parity and five-percent complete-training saving']
        for dep in sorted(deps):cmd.extend(['--dep',dep])
        for seed in (7,8):
            base=f'--train-tokens 65536 --dev-tokens 2048 --steps 256 --chunk 64 --credit-window 16 --payload 24 --heads 2 --depth 2 --pool 4 --lanes 8 --eval-lanes 8 --eval-every 64 --readout-init frequency --input-init balanced --future-score-positions 4 --seed {seed} --double'
            for arm,extra in [('ordinary',''),('batched',' --batched-lookup')]:
                cmd.extend(['--job',f'aws_integrated_p24_64k_double_{arm}_s{seed}_20261012_v1|{base}{extra}'])
        receipt=subprocess.check_output(cmd,cwd=ROOT,text=True)
        result['continuation_receipt']=receipt
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
