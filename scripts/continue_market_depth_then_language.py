"""B10 eight-layer market development at a safe R1 completion boundary."""
import argparse
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path
import continue_curie_deep_language as original
from continue_curie_deep_recovery import stable_host
ROOT=original.ROOT
D16='curie_allocated_deep_d16_8m_20261013_s6_v1'
D32='curie_allocated_deep_d8_32m_20261013_s6_v1'
SMOKE='curie_b10_depth8_20261014_s0_v1_smoke'
FIT='curie_b10_depth8_20261014_s0_v1_fit'


def market_run(tag,timeout):
    env=dict(os.environ,MEM_CAP_KB='12000000',MEM_CAP_RSS_KB='3000000',MIN_AVAIL_MB='8192',
             CGROUP_RESERVE_MB='2048',JOB_TIMEOUT_S=str(timeout),WAIT='1',WAIT_TIMEOUT_S='120')
    subprocess.run(['bash','experiments/queue/run_safe.sh',f'experiments/queue/b10/{tag}.txt'],cwd=ROOT,env=env,check=True)


def completed(tag,kind='market'):
    result=json.loads((ROOT/f'experiments/results/{kind}/{tag}.json').read_text())
    if result['status']!='completed':raise RuntimeError('Incomplete result')
    return result


def publish(paths,message):
    subprocess.run(['git','add',*paths],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m',message],cwd=ROOT,check=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--previous-controller-pid',type=int,required=True)
    args=p.parse_args();pid=args.previous_controller_pid
    command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ')
    if b'continue_curie_deep_recovery_v2.py' not in command:
        raise RuntimeError('Refuse to signal anything except the identified old coordinator')
    # Stop only the coordinator; its setsid run_safe training child continues.
    os.kill(pid,signal.SIGSTOP)
    print('Old coordinator stopped; active D16 trainer continues to completion',flush=True)
    deadline=time.monotonic()+7*86400
    while True:
        path=ROOT/f'experiments/results/token_language/{D16}.json'
        ready=path.exists() and json.loads(path.read_text()).get('status')=='completed'
        free=subprocess.run(['flock','-n','/tmp/experiments-runner.lock','true'],capture_output=True).returncode==0
        if ready and free:break
        if time.monotonic()>deadline:raise RuntimeError('D16 completion boundary not reached; inspect preserved jobs')
        time.sleep(20)
    # Child and ordinary host lock are finished; retire the stopped coordinator.
    os.kill(pid,signal.SIGKILL)
    original.run(D16+'_quality',5400);original.load(D16+'_quality','diagnostics')
    publish([f'experiments/results/token_language/{D16}.json',f'experiments/results/diagnostics/{D16}_quality.json'],
            'Complete admitted D16 language stage before market depth handoff')
    market_error=None
    try:market_stage()
    except Exception as error:
        market_error=repr(error)
        print(json.dumps({'market_stage':'failed','error':market_error,'next':'preserve reserved32M language stage'}),flush=True)
        failure=ROOT/f'experiments/results/market/{FIT}_failure.json'
        with failure.open('x') as f:json.dump(dict(status='failed',battle='B10',error=market_error,smoke=SMOKE,fit=FIT),f,indent=2);f.write('\n')
    # Preserve the reserved larger-data language test after this market fit.
    packet=original.load('curie_allocated_deep_d8_smoke_20261013_s6_v1')
    timeout=math.ceil(2*(packet['train_wall_s_total']/packet['presentations_total'])*65536*512+3600)
    if timeout>604800:raise RuntimeError('Measured language fit exceeds7days')
    stable_host();original.run(D32,timeout);original.load(D32)
    original.run(D32+'_quality',5400);original.load(D32+'_quality','diagnostics')
    publish([f'experiments/results/token_language/{D32}.json',f'experiments/results/diagnostics/{D32}_quality.json'],
            'Complete reserved32M native language acquisition after market stage')


def market_stage():
    stable_host();market_run(SMOKE,5400);smoke=completed(SMOKE)
    assert smoke['evidence']=='numerical/resource smoke' and smoke['max_rss_kb']<2800000
    assert len(smoke['stage_gradient_norm_totals'])==8 and all(x>0 and math.isfinite(x) for x in smoke['stage_gradient_norm_totals'])
    rate=smoke['training_wall_s']/smoke['training_targets']
    timeout=math.ceil(2*rate*30*smoke['train_windows']*1023+7200)
    if timeout>604800:raise RuntimeError('Measured market fit exceeds7days; optimize execution without shrinking depth')
    stable_host();market_run(FIT,max(timeout,5400));fit=completed(FIT)
    if fit['completion_reason']=='wall_limit':raise RuntimeError('Market fit stopped at resource deadline; retain checkpoint and diagnose')
    native=completed('curie_b10_race_dev_s0_20261009T1800Z')
    reference=completed('curie_b10_hawkes_dev_s0_20261009T1800Z')
    result=dict(status='completed',battle='B10',candidate=FIT,layers=8,
        native_reference=native['tag'],hawkes_reference=reference['tag'],
        native_gain=fit['best_val_ll_per_event']-native['best_val_ll_per_event'],
        hawkes_gain=fit['best_val_ll_per_event']-reference['best_val_ll_per_event'],
        training_targets=fit['training_targets'],training_wall_s=fit['training_wall_s'],parameters=fit['parameters'],
        scope='Single-seed development, matching chronological windows and1us interval score; deeper clock-race member. No sealed TEST, volatility gate or trading-profit claim. Full BPTT/all-clock work, no sparse hidden-routing or learned-credit scaling claim.')
    out=ROOT/f'experiments/results/market/{FIT}_decision.json'
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    publish([f'experiments/results/market/{SMOKE}.json',f'experiments/results/market/{FIT}.json',str(out.relative_to(ROOT))],
            'Complete native eight-layer market development comparison')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
