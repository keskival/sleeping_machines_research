"""Publish completed B1/R1 receiver-credit evidence and rebuild current report."""
import fcntl,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TAGS=['curie_unseen_receiver_v2_20261010T2140Z_pilot',*[f'curie_unseen_receiver_confirm_s{s}_20261010T2125Z' for s in (171,172)]]

def main():
    rows=[];paths=[]
    for tag in TAGS:
        p=ROOT/'experiments/results/credit'/f'{tag}.json';d=json.loads(p.read_text());assert d['status']=='completed'
        for source,digest in d['source_sha256'].items():assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==digest,source
        assert hashlib.sha256((ROOT/d['metrics']['checkpoint']).read_bytes()).hexdigest()==d['metrics']['checkpoint_sha256']
        paths.extend([str(p.relative_to(ROOT)),d['metrics']['checkpoint']])
        for arm in d['metrics']['arms']:
            rows.append(dict(seed=d['args']['seed'],arm=arm['arm'],inactive=arm['scopes']['unseen_inactive'],utility=arm['utility'],wall_s=d['wall_s'],supported_flops=d['supported_flops']))
    out=ROOT/'experiments/results/credit/curie_unseen_receiver_confirm_20261010T2125Z_decision.json';assert not out.exists()
    structured=[r for r in rows if r['arm']=='structured']
    decision=dict(status='completed',battle='B1/R1',rows=rows,all_structured_inactive_better_than_role_mean=all(r['inactive']['relative_mse_vs_zero']<r['inactive']['role_constant_relative_mse'] for r in structured),all_structured_pool_gains_positive=all(r['utility']['gain_from_expanding_counterfactual_pool']>0 for r in structured),scope='Three fitted seeds, frozen native depth4 actor, synthetic fresh DEV continuations. Optional hard key writes; no online actor fitting, general internal routing, real-stream TEST or efficiency claim. Each phase fits both nearly parameter-matched critics and charges all receiver queries.')
    with open('/tmp/aws-language-publication.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='main'
        out.write_text(json.dumps(decision,indent=2)+'\n');paths.append(str(out.relative_to(ROOT)))
        h=ROOT/'experiments/HANDOFF.md';gains=[round(r['utility']['gain_from_expanding_counterfactual_pool'],6) for r in structured]
        h.write_text(f"**Curie unseen receiver confirmation completed.** Seeds170/171/172 structured expanded-pool gains{gains}; all-positive gate{decision['all_structured_pool_gains_positive']}, all inactive calibration better than TRAIN-role mean{decision['all_structured_inactive_better_than_role_mean']}. Evidence in credit/curie_unseen_receiver_confirm_20261010T2125Z_decision.json. Frozen native synthetic diagnostic; no TEST/efficiency claim. Next: integrate route-credit learning with causal changing-weight transitions and paid lazy discovery.\n\n"+h.read_text());paths.append('experiments/HANDOFF.md')
        subprocess.run(['git','add','-f',*paths],cwd=ROOT,check=True)
        subprocess.run(['git','commit','--only','-m','Publish three seed native unseen receiver credit confirmation',*paths],cwd=ROOT,check=True)
    subprocess.run([str(ROOT/'.venv-docker/bin/python'),'scripts/build_curie_episode_report.py','--tag','curie_unseen_receiver_report_20261010T2135Z'],cwd=ROOT,check=True)
if __name__=='__main__':main()
