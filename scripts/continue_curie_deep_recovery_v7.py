"""Curie chain v7 (supersedes v6 before any v6 job started; the episodic control arm continues).

1. Wait for the running episodic control arm and the free host lock; write the pair decision, publish.
2. R1 depth ladder: residual D2, serial D8, residual D8, residual D16, then residual+per-stage reads D2/D8/D16, at 1M fresh targets; long-history scoring; publish.
3. R1 keyed member 16.8M fresh-token 1024-lane fit (v5 stage).
4. Admitted B10 eight-layer market stage.
"""
import json
import subprocess
import time
import continue_curie_deep_language as original
import continue_market_depth_v2 as market
import continue_curie_deep_recovery_v4 as v4
import continue_curie_deep_recovery_v5 as v5
from continue_curie_deep_recovery import stable_host

ROOT = original.ROOT
LADDER = (('d2_residual', 3600), ('d8_serial', 10800), ('d8_residual', 10800), ('d16_residual', 21600),
          ('d2_residual_reads', 5400), ('d8_residual_reads', 14400), ('d16_residual_reads', 28800))


def finish_pair():
    control = v4.PAIR['none'][1]; path = ROOT / f'experiments/results/token_language/{control}.json'
    while True:
        ready = path.exists() and json.loads(path.read_text()).get('status') == 'completed'
        free = subprocess.run(['flock', '-n', '/tmp/experiments-runner.lock', 'true'], capture_output=True).returncode == 0
        if ready and free: break
        if not ready and free and not subprocess.run(['pgrep', '-f', f'run_safe.sh experiments/queue/r1/{control}.txt'], capture_output=True).stdout:
            raise RuntimeError('Control arm runner ended without a completed result')
        time.sleep(30)
    e, c = original.load(v4.PAIR['episodic'][1]), original.load(control)
    decision = dict(status='completed', battle='R1', episodic=v4.PAIR['episodic'][1], control=control,
                    parent_nll=5.978387557528713, episodic_nll=e['scored']['nll'], control_nll=c['scored']['nll'],
                    episodic_gain_over_control=c['scored']['nll'] - e['scored']['nll'], episodic_native_readout_nll=e['native_readout_scored']['nll'],
                    episodic_prequential=e['prequential_nll'], control_prequential=c['prequential_nll'],
                    training_s=dict(episodic=e['training_s'], control=c['training_s']), memory_work=e['memory_work'],
                    smoke_training_s={arm: original.load(v4.PAIR[arm][0])['training_s'] for arm in v4.PAIR},
                    rule='Adopt the episodic store for R1 continuations if it beats the identical-stream control by >=0.03 nats on the scored DEV protocol.',
                    scope='Single seed, identical restored learner and fresh-TRAIN stream, 8192 DEV targets/1024 windows; no public validation.')
    decision['adopt'] = decision['episodic_gain_over_control'] >= 0.03
    out = ROOT / f'experiments/results/diagnostics/{v4.DECISION}.json'
    with out.open('x') as f: json.dump(decision, f, indent=2); f.write('\n')
    paths = [str(out.relative_to(ROOT))]
    for smoke, fit in v4.PAIR.values(): paths += [f'experiments/results/token_language/{smoke}.json', f'experiments/results/token_language/{fit}.json']
    market.publish(paths, 'Complete R1 addressed episodic memory 2M pair and decision')
    print(json.dumps(decision), flush=True)


def depth_ladder():
    for arm, timeout in LADDER:
        tag = f'curie_depth_ladder_{arm}_1m_20261015_v1'
        try:
            stable_host(); original.run(tag, timeout); original.load(tag)
        except Exception as error:
            print(json.dumps({'ladder_arm': tag, 'failed': repr(error)}), flush=True)
    tag = 'curie_depth_ladder_1m_decision_20261015_v1'
    subprocess.run(['/workspace/.venv-docker/bin/python', 'analysis/score_depth_ladder.py', '--tag', tag], cwd=ROOT, check=True)
    paths = [f'experiments/results/diagnostics/{tag}.json'] + [f'experiments/results/token_language/curie_depth_ladder_{arm}_1m_20261015_v1.json'
             for arm, _ in LADDER if (ROOT / f'experiments/results/token_language/curie_depth_ladder_{arm}_1m_20261015_v1.json').exists()]
    market.publish(paths, 'Complete R1 residual/serial depth ladder at 1M targets')


def main():
    for stage in (finish_pair, depth_ladder, v5.keyed_scaling):
        try: stage()
        except Exception as error: print(json.dumps({stage.__name__: 'failed', 'error': repr(error)}), flush=True)
    try:
        market.market_stage()
    except Exception as error:
        print(json.dumps({'market_stage': 'failed', 'error': repr(error)}), flush=True)
        failure = ROOT / f'experiments/results/market/{market.FIT}_failure.json'
        with failure.open('x') as f: json.dump(dict(status='failed', battle='B10', error=repr(error), smoke=market.SMOKE, fit=market.FIT), f, indent=2); f.write('\n')
    print('Queue complete', flush=True)


if __name__ == '__main__':
    main()
