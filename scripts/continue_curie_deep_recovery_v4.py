"""Curie R1/B10 chain after the 15 October restart, superseding v3 at the D16 job boundary.

1. Wait for the running exact-resume D16 job (started by v3) and the free host lock.
2. Charge the interruption, score D16 quality, publish.
3. R1 addressed episodic memory pair (smokes, then identical-stream 2M arms), decision, publish.
4. Admitted B10 eight-layer market stage (continue_market_depth_v2.market_stage), unchanged.
"""
import json
import math
import subprocess
import time
import continue_curie_deep_language as original
import continue_market_depth_v2 as market
from continue_curie_deep_recovery import stable_host

ROOT = original.ROOT
OLD = 'curie_allocated_deep_d16_8m_20261013_s6_v1'
NEW = 'curie_allocated_deep_d16_8m_20261013_s6_resume1_v1'
PAIR = {arm: (f'curie_episodic_memory_smoke_{arm}_20261015_v1', f'curie_episodic_memory_2m_{arm}_20261015_v1') for arm in ('episodic', 'none')}
DECISION = 'curie_episodic_memory_2m_decision_20261015_v1'


def finish_d16():
    path = ROOT / f'experiments/results/token_language/{NEW}.json'
    while True:
        ready = path.exists() and json.loads(path.read_text()).get('status') == 'completed'
        free = subprocess.run(['flock', '-n', '/tmp/experiments-runner.lock', 'true'], capture_output=True).returncode == 0
        if ready and free: break
        if not ready and free and not subprocess.run(['pgrep', '-f', f'run_safe.sh experiments/queue/r1/{NEW}.txt'], capture_output=True).stdout:
            raise RuntimeError('D16 resume runner ended without a completed result; inspect its log')
        time.sleep(30)
    receipt = json.loads((ROOT / f'experiments/results/token_language/{OLD}_interruption.json').read_text())
    packet = json.loads(path.read_text())
    if 'resource_interruptions' not in packet:
        packet['resource_interruptions'] = [receipt]
        packet['whole_fit_process_wall_lower_bound_s'] = receipt['prior_runner_observed_interval_lower_bound_s'] + packet['process_wall_s']
        packet['whole_fit_work_scope'] = 'Observed interrupted runner interval plus resumed process wall. Lower bound; successful-trajectory training time excludes discarded work after the step-15360 checkpoint.'
        tmp = path.with_suffix('.work.tmp'); tmp.write_text(json.dumps(packet, indent=2) + '\n'); tmp.replace(path)
    original.run(NEW + '_quality', 5400)
    original.load(NEW + '_quality', 'diagnostics')
    market.publish([f'experiments/results/token_language/{NEW}.json', f'experiments/results/diagnostics/{NEW}_quality.json',
                    f'experiments/results/token_language/{OLD}_interruption.json'],
                   'Complete R1 depth16 8M acquisition (exact resume) and DEV diagnostics')


def episodic_pair():
    results = {}
    for arm in ('episodic', 'none'):
        smoke, fit = PAIR[arm]
        stable_host(); original.run(smoke, 3600); s = original.load(smoke)
        assert s['max_rss_kb'] < 2_500_000 and math.isfinite(s['prequential_nll'])
        timeout = math.ceil(2 * s['training_s'] / 64 * (4096 - 64) + 3600)
        if timeout > 86400: raise RuntimeError('Measured episodic 2M arm exceeds one day; optimize execution before fitting')
        stable_host(); original.run(fit, max(timeout, 5400)); results[arm] = original.load(fit)
    e, c = results['episodic'], results['none']
    decision = dict(status='completed', battle='R1', episodic=PAIR['episodic'][1], control=PAIR['none'][1],
                    parent_nll=5.978387557528713, episodic_nll=e['scored']['nll'], control_nll=c['scored']['nll'],
                    episodic_gain_over_control=c['scored']['nll'] - e['scored']['nll'],
                    episodic_native_readout_nll=e['native_readout_scored']['nll'],
                    episodic_prequential=e['prequential_nll'], control_prequential=c['prequential_nll'],
                    training_s=dict(episodic=e['training_s'], control=c['training_s']), memory_work=e['memory_work'],
                    smoke_training_s={arm: original.load(PAIR[arm][0])['training_s'] for arm in PAIR},
                    rule='Adopt the episodic store for R1 continuations if it beats the identical-stream control by >=0.03 nats on the scored DEV protocol; otherwise diagnose read/gate/credit before any larger fit.',
                    scope='Single seed, identical restored learner and fresh-TRAIN stream, 8192 DEV targets/1024 windows; no public validation.')
    decision['adopt'] = decision['episodic_gain_over_control'] >= 0.03
    out = ROOT / f'experiments/results/diagnostics/{DECISION}.json'
    with out.open('x') as f: json.dump(decision, f, indent=2); f.write('\n')
    paths = [str(out.relative_to(ROOT))]
    for smoke, fit in PAIR.values(): paths += [f'experiments/results/token_language/{smoke}.json', f'experiments/results/token_language/{fit}.json']
    market.publish(paths, 'Complete R1 addressed episodic memory 2M pair and decision')
    print(json.dumps(decision), flush=True)


def main():
    finish_d16()
    try:
        episodic_pair()
    except Exception as error:
        print(json.dumps({'episodic_pair': 'failed', 'error': repr(error)}), flush=True)
    try:
        market.market_stage()
    except Exception as error:
        print(json.dumps({'market_stage': 'failed', 'error': repr(error), 'next': 'stop; inspect completed repair evidence'}), flush=True)
        failure = ROOT / f'experiments/results/market/{market.FIT}_failure.json'
        with failure.open('x') as f:
            json.dump(dict(status='failed', battle='B10', error=repr(error), smoke=market.SMOKE, fit=market.FIT), f, indent=2); f.write('\n')
    print('Queue complete', flush=True)


if __name__ == '__main__':
    main()
