"""R1: wait for the reserved ladder, then admit one cross-window acquisition."""
import hashlib
import json
import math
import subprocess
import time
import continue_curie_deep_language as original
from continue_curie_deep_recovery import stable_host

ROOT=original.ROOT
SMOKE='curie_crosswindow_d8_8m_s6_20261013_v1_smoke'
FIT='curie_crosswindow_d8_8m_s6_20261013_v1_resume'
BASELINE='curie_allocated_deep_d8_8m_20261013_s6_resume2_v1'


def main():
    # Bound waiting without changing or duplicating another controller's work.
    deadline=time.monotonic()+14*86400
    while any(subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0
              for session in ('curie_deep_recovery_20261013_v2','curie_market_depth_20261014_v1')):
        if time.monotonic()>deadline:raise RuntimeError('Current ladder exceeded controller wait envelope')
        time.sleep(20)
    tags=(BASELINE,'curie_allocated_deep_d16_8m_20261013_s6_v1','curie_allocated_deep_d8_32m_20261013_s6_v1')
    diagnostics=[]
    for tag in tags:
        original.load(tag)
        diagnostics.append(original.load(tag+'_quality','diagnostics'))
    assert all(d['forward_horizon_contract']['status']=='passed' for d in diagnostics)
    if not any(d['forward_horizon_contract']['maximum_after_boundary_feature_change']>1e-12 for d in diagnostics):
        raise RuntimeError('No measured cross-window feature effect; diagnose before extending credit')
    print('Completed native ladder and cross-window gates passed; admit one horizon candidate',flush=True)
    stable_host();original.run(SMOKE,5400)
    smoke_path=ROOT/f'experiments/results/token_language/{SMOKE}.partial.json'
    smoke=json.loads(smoke_path.read_text())
    assert smoke['status']=='checkpointed' and smoke['max_rss_kb']<2500000
    assert all(math.isfinite(row['dev_nll']) for row in smoke['curve'])
    norms=smoke['curve'][-1]['receiver_gradient_norms']['banks.input']
    assert len(norms)==8 and all(sum(v for head in layer for v in head)>0 for layer in norms)
    rate=smoke['train_wall_s_total']/smoke['presentations_total']
    timeout=math.ceil(2*rate*16384*512+3600)
    if timeout>604800:raise RuntimeError('Measured horizon fit exceeds seven-day envelope; optimize execution')
    stable_host();original.run(FIT,max(timeout,5400));candidate=original.load(FIT)
    # Source/setting identity and exact state/RNG restore are checked by driver.
    candidate['whole_fit_process_wall_s']=smoke['process_wall_s']+candidate['process_wall_s']
    candidate['whole_fit_work_scope']='Smoke process plus exact-resume process; successful-trajectory timer already carries smoke steps. No interruption tail in this ledger.'
    candidate['smoke_acquisition_sha256']=hashlib.sha256(smoke_path.read_bytes()).hexdigest()
    path=ROOT/f'experiments/results/token_language/{FIT}.json';tmp=path.with_suffix('.work.tmp')
    tmp.write_text(json.dumps(candidate,indent=2)+'\n');tmp.replace(path)
    original.run(FIT+'_quality',5400);scored=original.load(FIT+'_quality','diagnostics')
    baseline=diagnostics[0]
    result=dict(status='completed',battle='R1',baseline=BASELINE,candidate=FIT,
        final_long_history_dev_gain=baseline['long_history_dev']['nll']-scored['long_history_dev']['nll'],
        selected_long_history_dev_gain=baseline['selected_long_history_dev']['nll']-scored['selected_long_history_dev']['nll'],
        baseline_work_precision=baseline['process_work_precision'],candidate_work_precision=scored['process_work_precision'],
        candidate_credit_update_diagnostics=candidate['credit_update_diagnostics'],
        scope='Single-seed development comparison, identical data/depth/short-DEV selection criterion and long-history targets. Candidate adds the resource-smoke step64 DEV evaluation to regular cadence; report this selection opportunity. Longer conditional first-time-held route horizon; no public validation or matched-total-work win. Baseline wall is lower-bound when its interruption tail is unknown.')
    out=ROOT/f'experiments/results/diagnostics/{FIT}_decision.json'
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    paths=[str(smoke_path.relative_to(ROOT)),str(path.relative_to(ROOT)),
           f'experiments/results/diagnostics/{FIT}_quality.json',str(out.relative_to(ROOT))]
    subprocess.run(['git','add',*paths],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','Complete R1 cross-window native credit acquisition comparison'],cwd=ROOT,check=True)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
