"""Bounded sequential confirmation and completed-evidence publication."""
import fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TAG='curie_real_boundary_confirm_20261010T2210Z'
def main():
    manifest=json.loads((ROOT/'experiments/queue/enabler'/TAG/'manifest.json').read_text());env=dict(os.environ,MEM_CAP_KB='8000000',MEM_CAP_RSS_KB='1500000',MIN_AVAIL_MB='9216',CGROUP_RESERVE_MB='2048',JOB_TIMEOUT_S='600')
    for job in manifest['jobs']:
        for s,h in manifest['sources'].items():assert hashlib.sha256((ROOT/s).read_bytes()).hexdigest()==h,s
        if not (ROOT/job['result']).exists():subprocess.run(['bash','experiments/queue/run_safe.sh',job['queue']],cwd=ROOT,env=env,check=True)
        result=json.loads((ROOT/job['result']).read_text());assert result['status']=='completed'
        assert result['source_sha256']==manifest['sources']
    v3=json.loads((ROOT/'experiments/results/credit/curie_real_boundary_v3_20261010T2200Z_measure.json').read_text());v2=json.loads((ROOT/'experiments/results/credit/curie_real_boundary_v2_20261010T2150Z_measure.json').read_text())
    rows=[dict(seed=181,arms=[r for r in v2['metrics']['rows'] if r['arm'] in ('bptt','untrained_audit')]+[r for r in v3['metrics']['rows'] if r['arm']=='calibrated_audit'])]
    for job in manifest['jobs'][1:]:
        d=json.loads((ROOT/job['result']).read_text());rows.append(dict(seed=d['args']['seed'],arms=d['metrics']['rows']))
    gains=[]
    for row in rows:
        by={r['arm']:r for r in row['arms']};gains.append(dict(seed=row['seed'],over_untrained=by['untrained_audit']['nll']-by['calibrated_audit']['nll'],over_bptt=by['bptt']['nll']-by['calibrated_audit']['nll']))
    decision=dict(status='completed',battle='B10/R1',gains=gains,positive_all_seeds=all(g['over_untrained']>0 and g['over_bptt']>0 for g in gains),scope='Three initializations and audit seeds on same real TRAIN fragment. No optimized efficiency, domain-generalization or sealed TEST claim.')
    decisionpath=ROOT/'experiments/results/credit'/f'{TAG}_decision.json';assert not decisionpath.exists();decisionpath.write_text(json.dumps(decision,indent=2)+'\n')
    paths=[j['result'] for j in manifest['jobs']]+[str(decisionpath.relative_to(ROOT))]
    with open('/tmp/aws-language-publication.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='main'
        handoff=ROOT/'experiments/HANDOFF.md';handoff.write_text('**Curie real online-credit confirmation completed.** '+json.dumps(gains)+'; all-seed point gate='+str(decision['positive_all_seeds'])+'. Same8192-event TRAIN fragment, not a domain benchmark or optimized efficiency claim. Completed result decision '+str(decisionpath.relative_to(ROOT))+'; actor/credit/optimizer/state snapshots preserved. Next: fresh chronological fragments and learned-state/route integration at depth4/8 after error analysis.\n\n'+handoff.read_text());paths.append('experiments/HANDOFF.md')
        subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True);subprocess.run(['git','commit','--only','-m','Publish three-seed real online credit confirmation','--',*paths],cwd=ROOT,check=True)
    subprocess.run(['bash','experiments/queue/run_safe.sh','experiments/queue/enabler/curie_real_credit_report_20261010T2205Z/curie_real_credit_report_20261010T2205Z.txt'],cwd=ROOT,env=env,check=True)
if __name__=='__main__':main()
