"""Finish admitted R1/B1 pilots, publish once and rebuild current report."""
import fcntl,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PREFIX='curie_averaged_credit_v3_20261010T1720Z'
MANIFEST='experiments/queue/enabler/curie_averaged_credit_finish_20261010T1730Z/manifest.json'

def main():
    deadline=time.monotonic()+7200
    state=ROOT/'.git/curie-recovery-preview'/PREFIX/'state.json'
    while True:
        data=json.loads(state.read_text()) if state.exists() else {}
        if data.get('status')=='completed':break
        if data.get('status')=='needs_review':raise RuntimeError(data)
        if time.monotonic()>deadline:raise TimeoutError('Admitted critic pipeline did not finish')
        time.sleep(20)
    manifest=json.loads((ROOT/'experiments/queue/enabler'/PREFIX/'manifest.json').read_text())
    records=[]
    for job in manifest['jobs']:
        result=json.loads((ROOT/job['result']).read_text())
        assert result['status']=='completed' and result['tag']==job['tag']
        assert all(result['source_sha256'].get(p)==sha for p,sha in job['result_sources'].items())
        if '_pilot_' in job['tag']:records.append(result)
    assert len(records)==3
    gates={}
    for history in ('isolated','connected'):
        for label in ('single','averaged'):
            arm=[next(v for v in r['metrics']['arms'] if v['history']==history and v['labels']==label) for r in records]
            gates[history+'/'+label]=dict(point_gate_all_seeds=all(v['mse_gain_vs_averaged_train_constant']>0 and v['utility_gain_vs_averaged_train_constant']>0 for v in arm),mse_gains=[v['mse_gain_vs_averaged_train_constant'] for v in arm],utility_gains=[v['utility_gain_vs_averaged_train_constant'] for v in arm])
    decision=dict(status='completed',tag=PREFIX+'_decision',battle='R1/B1',arms=gates,next_action='Review label/history effects and uncertainty before any depth fit; no automatic scaling',scope='Three synthetic frozen-actor pilots; point gate is calibration and utility above common TRAIN-mean action, not statistical confirmation.')
    path=ROOT/'experiments/results/credit'/(PREFIX+'_decision.json');assert not path.exists()
    with open('/tmp/aws-language-publication.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='main'
        path.write_text(json.dumps(decision,indent=2)+'\n');paths=[j['result'] for j in manifest['jobs']]+[str(path.relative_to(ROOT))]
        hand=ROOT/'experiments/HANDOFF.md';hand.write_text('**Curie averaged-credit pilots completed.** Registered crossed single/averaged labels and isolated/connected history have completed three seeds. Decision: '+json.dumps(gates)+'. Saved evidence precedes any new admission; no automatic deep fit. See R1 future-credit appendix and economical-event-credit design.\n\n'+hand.read_text());paths.append('experiments/HANDOFF.md')
        subprocess.run(['git','add','-f','--',*paths],cwd=ROOT,check=True);subprocess.run(['git','commit','--only','-m','Publish completed averaged future-credit diagnostic','--',*paths],cwd=ROOT,check=True)
    subprocess.run([str(ROOT/'.venv-docker/bin/python'),'scripts/run_curie_recovery.py','--manifest',MANIFEST],cwd=ROOT,check=True)
if __name__=='__main__':main()
