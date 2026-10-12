"""R1 causal pairing check and post-acquisition exposure decision."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True)
    p.add_argument('--reference',required=True);p.add_argument('--candidate',required=True)
    a=p.parse_args();folder=ROOT/'experiments/results/token_language';records=[];inputs={}
    for tag in (a.reference,a.candidate):
        path=folder/f'{tag}.json';blob=path.read_bytes();d=json.loads(blob);assert d['status']=='completed'
        for name,digest in d['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
        inputs[str(path.relative_to(ROOT))]=hashlib.sha256(blob).hexdigest();records.append(d)
    r,c=records
    assert {k:v for k,v in r['args'].items() if k not in ('tag','exposure')}=={k:v for k,v in c['args'].items() if k not in ('tag','exposure')}
    assert r['identity']==c['identity'] and r['host']==c['host']
    prior=r['args']['train_tokens']; branch=r['args']['steps']//2
    assert r['presentations_total']==c['presentations_total']==2*(prior-r['args']['lanes'])
    assert r['exposure_protocol']['prior_tokens']==c['exposure_protocol']['prior_tokens']==prior
    assert r['args']['exposure']=='repeated' and c['args']['exposure']=='fresh'
    before_r=[x for x in r['curve'] if x['step']<=branch]
    before_c=[x for x in c['curve'] if x['step']<=branch]
    assert [x['step'] for x in before_r]==[0,r['args']['eval_every'],branch]==[x['step'] for x in before_c]
    prefix_error=max(abs(x['dev_nll']-y['dev_nll']) for x,y in zip(before_r,before_c))
    assert prefix_error==0
    assert before_r[-1]['learner_fingerprint']==before_c[-1]['learner_fingerprint']
    rows=[]
    for d in records:
        selected=min(x['dev_nll'] for x in d['curve'])
        post=min(x['dev_nll'] for x in d['curve'] if x['step']>branch)
        rows.append(dict(exposure=d['args']['exposure'],unique_tokens=d['exposure_protocol']['unique_admitted_tokens'],
            prior_tokens=d['exposure_protocol']['prior_tokens'],fitting_targets=d['presentations_total'],
            initial_dev_nll=d['curve'][0]['dev_nll'],branch_dev_nll=before_r[-1]['dev_nll'],
            selected_dev_nll=selected,post_branch_dev_nll=post,final_dev_nll=d['curve'][-1]['dev_nll'],
            training_wall_s=d['train_wall_s_total'],engine_wall_s=d['wall_s'],process_wall_s=d['process_wall_s']))
    gain=rows[0]['post_branch_dev_nll']-rows[1]['post_branch_dev_nll']
    result=dict(status='completed',battle='R1',rows=rows,prefix_curve_error=prefix_error,
        shared_learner_fingerprint=before_r[-1]['learner_fingerprint'],post_branch_quality_gain=gain,
        selected_quality_gain=rows[0]['selected_dev_nll']-rows[1]['selected_dev_nll'],
        development_gate_passed=gain>0 and rows[1]['training_wall_s']<=1.05*rows[0]['training_wall_s'],
        input_sha256=inputs,source_sha256={'experiments/analyze_common_prior_stream_v2.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        scope=f'Single-seed private DEV, same first{prior}tokens/prior/first{branch}updates and bit-identical full acquired-state fingerprint (model, AdamW, numeric state, all generators and cursor). Both reset forward state at the same block boundary; actor/optimizer/RNG continue. Second{branch}updates use repeated versus fresh{prior}tokens. Same{r["presentations_total"]}fitting targets,{r["args"]["steps"]}updates and selection opportunities. This isolates post-acquisition exposure under this lane/block protocol; no learned exploration-policy or Transformer claim.')
    out=folder/f'{a.tag}.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
