"""Curie chain v8 (supersedes v7 mid-ladder; the running ladder job continues). Resumable stages.

1. Remaining R1 depth-ladder arms (finished arms skipped, a running arm awaited), then ladder scoring and publication.
2. R1 keyed-member depth: L4 and L8 at the exact 1M L2 recipe; flagship-protocol placement and history reliance; publish.
3. R1 keyed member 16.8M fresh-token 1024-lane fit (v5 stage).
4. Admitted B10 eight-layer market stage.
"""
import json
import subprocess
import time
import continue_curie_deep_language as original
import continue_market_depth_v2 as market
import continue_curie_deep_recovery_v5 as v5
import continue_curie_deep_recovery_v7 as v7
from continue_curie_deep_recovery import stable_host

ROOT = original.ROOT
PY = '/workspace/.venv-docker/bin/python'
KEYED_DEPTH = (('curie_r1g2_keyed1_1M_l4_s0_20261015_v1', 21600), ('curie_r1g2_keyed1_1M_l8_s0_20261015_v1', 36000))


def completed(tag):
    path = ROOT / f'experiments/results/token_language/{tag}.json'
    return path.exists() and json.loads(path.read_text()).get('status') == 'completed'


def run_or_await(tag, timeout):
    if completed(tag): return
    if subprocess.run(['pgrep', '-f', f'run_safe.sh experiments/queue/r1/{tag}.txt'], capture_output=True).stdout:
        while subprocess.run(['pgrep', '-f', f'run_safe.sh experiments/queue/r1/{tag}.txt'], capture_output=True).stdout: time.sleep(30)
        if not completed(tag): raise RuntimeError(f'{tag} ended without a completed result')
        return
    stable_host(); original.run(tag, timeout); original.load(tag)


def depth_ladder():
    for arm, timeout in v7.LADDER:
        tag = f'curie_depth_ladder_{arm}_1m_20261015_v1'
        try: run_or_await(tag, timeout)
        except Exception as error: print(json.dumps({'ladder_arm': tag, 'failed': repr(error)}), flush=True)
    tag = 'curie_depth_ladder_1m_decision_20261015_v1'
    subprocess.run([PY, 'analysis/score_depth_ladder.py', '--tag', tag], cwd=ROOT, check=True)
    paths = [f'experiments/results/diagnostics/{tag}.json'] + [f'experiments/results/token_language/curie_depth_ladder_{arm}_1m_20261015_v1.json'
             for arm, _ in v7.LADDER if completed(f'curie_depth_ladder_{arm}_1m_20261015_v1')]
    market.publish(paths, 'Complete R1 residual/serial/read depth ladder at 1M targets')


def keyed_depth():
    for tag, timeout in KEYED_DEPTH:
        try: run_or_await(tag, timeout)
        except Exception as error: print(json.dumps({'keyed_depth': tag, 'failed': repr(error)}), flush=True)
    done = [t for t, _ in KEYED_DEPTH if completed(t)]
    if not done: raise RuntimeError('No keyed depth arm completed')
    placement = 'curie_keyed_depth_flagship_protocol_20261015_v1'
    subprocess.run([PY, 'analysis/cross_member_language_protocol.py', '--tag', placement, '--runs', 'curie_r1g2_keyed1_1M_s0_20261007T1800Z', *done], cwd=ROOT, check=True)
    paths = [f'experiments/results/diagnostics/{placement}.json']
    for t in ['curie_r1g2_keyed1_1M_s0_20261007T1800Z', *done]:
        h = f'curie_keyed_history_reliance_{t.split("_keyed1_")[1]}'
        subprocess.run([PY, 'analysis/keyed_history_reliance.py', '--tag', h, '--run', t], cwd=ROOT, check=True)
        paths.append(f'experiments/results/diagnostics/{h}.json')
    market.publish(paths + [f'experiments/results/token_language/{t}.json' for t in done], 'Complete R1 keyed-member depth ladder at 1M tokens')


def main():
    for stage in (depth_ladder, keyed_depth, v5.keyed_scaling):
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
