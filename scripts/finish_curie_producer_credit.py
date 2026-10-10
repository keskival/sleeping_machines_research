"""Source-bound bounded producer-credit pilot, publication and current report."""
import fcntl,hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];TAG='curie_feedback_metric_v2_20261010T2355Z'
def main():
    manifest=json.loads((ROOT/'experiments/queue/enabler'/TAG/'manifest.json').read_text());env=dict(os.environ,MEM_CAP_KB='8000000',MEM_CAP_RSS_KB='1500000',MIN_AVAIL_MB='9216',CGROUP_RESERVE_MB='2048',JOB_TIMEOUT_S='600')
    for job in manifest['jobs']:
        for source,digest in manifest['sources'].items():assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==digest,source
        assert hashlib.sha256((ROOT/job['queue']).read_bytes()).hexdigest()==job['queue_sha256']
        if not (ROOT/job['result']).exists():subprocess.run(['bash','experiments/queue/run_safe.sh',job['queue']],cwd=ROOT,env=env,check=True)
        result=json.loads((ROOT/job['result']).read_text());assert result['status']=='completed' and result['source_sha256']==manifest['sources']
    result=json.loads((ROOT/manifest['jobs'][-1]['result']).read_text());arm=result['metrics']['rows'][0];base=json.loads((ROOT/'experiments/results/credit/curie_real_boundary_stable_v1_20261010T2310Z_measure.json').read_text());by={r['arm']:r for r in base['metrics']['rows']};feedback=json.loads((ROOT/'experiments/results/credit/curie_feedback_credit_v1_20261010T2330Z_measure.json').read_text())['metrics']['rows'][0]
    decision=dict(status='completed',battle='B10/R1',seed=183,targets=arm['targets'],nll=arm['nll'],gain_over_untrained=by['untrained_audit']['nll']-arm['nll'],gain_over_bptt=by['bptt']['nll']-arm['nll'],gain_over_unweighted_feedback=feedback['nll']-arm['nll'],producer_error_ratio=sum(arm['audited_producer_mse'])/sum(arm['audited_zero_producer_mse']),metric_wall_s=arm['producer_metric_wall_s'],scope='One real TRAIN fragment; matched model and causal audit draws. Exact producer metric construction and all replay/local/optimizer work charged. No replicated utility or optimized efficiency claim.')
    path=ROOT/'experiments/results/credit'/f'{TAG}_decision.json';assert not path.exists();path.write_text(json.dumps(decision,indent=2)+'\n');paths=[j['result'] for j in manifest['jobs']]+[str(path.relative_to(ROOT))]
    with open('/tmp/aws-language-publication.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='main'
        h=ROOT/'experiments/HANDOFF.md';h.write_text('**Curie producer-weighted pilot completed.** '+json.dumps(decision)+'. Tutorials1–8 complete. Unseen-route support estimator passed; current native credit is still state-indexed. Next: use producer-weighted calibration/utility diagnosis before expanding data/depth; route-conditioned critic plus supported exploration needs native integration. No TEST.\n\n'+h.read_text());paths.append('experiments/HANDOFF.md')
        subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True);subprocess.run(['git','commit','--only','-m','Publish producer-weighted online credit pilot and current decision','--',*paths],cwd=ROOT,check=True)
    subprocess.run(['bash','experiments/queue/run_safe.sh','experiments/queue/enabler/curie_clock_tutorial_report_20261010T2320Z/curie_clock_tutorial_report_20261010T2320Z.txt'],cwd=ROOT,env=env,check=True)
if __name__=='__main__':main()
