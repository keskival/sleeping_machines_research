"""Curie chain v9 (supersedes v8 mid-ladder; the running ladder job continues). Resumable stages.

1. Remaining R1 depth-ladder arms (finished arms skipped, a running arm awaited), ladder scoring, per-depth usage probe,
   publication.
2. R1 depth per fresh token: plan-B deep keyed member L2 and L8, one pass over 4M fresh tokens (r1_deep_keyed_stream.py),
   publication. Supersedes v8's keyed L4/L8 on the original two-layer driver at the multi-epoch 1M recipe: that member
   is data-limited at 1M (best epoch 2), its post-norm layers have no per-layer reads, and AWS's plan-B 1M ladder
   already covers the multi-epoch regime with the stronger construction.
3. R1 keyed member 16.8M fresh-token 1024-lane fit (v5 stage).
4. Admitted B10 eight-layer market stage.
"""
import json
import subprocess
import continue_market_depth_v2 as market
import continue_curie_deep_recovery_v5 as v5
import continue_curie_deep_recovery_v7 as v7
from continue_curie_deep_recovery_v8 import ROOT, PY, completed, run_or_await

STREAM = (('curie_r1_deep_keyed_stream_L2_4M_s0_20261015_v1', 21600), ('curie_r1_deep_keyed_stream_L8_4M_s0_20261015_v1', 50400))


def depth_ladder():
    for arm, timeout in v7.LADDER:
        tag = f'curie_depth_ladder_{arm}_1m_20261015_v1'
        try: run_or_await(tag, timeout)
        except Exception as error: print(json.dumps({'ladder_arm': tag, 'failed': repr(error)}), flush=True)
    tag = 'curie_depth_ladder_1m_decision_20261015_v1'
    subprocess.run([PY, 'analysis/score_depth_ladder.py', '--tag', tag], cwd=ROOT, check=True)
    arms = [f'curie_depth_ladder_{arm}_1m_20261015_v1' for arm, _ in v7.LADDER if completed(f'curie_depth_ladder_{arm}_1m_20261015_v1')]
    probe = 'curie_depth_usage_probe_ladder_20261015_v1'
    subprocess.run([PY, 'analysis/depth_usage_probe.py', '--tag', probe, '--arms', *arms], cwd=ROOT, check=True)
    paths = [f'experiments/results/diagnostics/{tag}.json', f'experiments/results/diagnostics/{probe}.json'] + [f'experiments/results/token_language/{t}.json' for t in arms]
    market.publish(paths, 'Complete R1 residual/serial/read depth ladder at 1M targets with per-depth usage probe')


def stream_depth():
    for tag, timeout in STREAM:
        try: run_or_await(tag, timeout)
        except Exception as error: print(json.dumps({'stream_depth': tag, 'failed': repr(error)}), flush=True)
    done = [t for t, _ in STREAM if completed(t)]
    if not done: raise RuntimeError('No stream depth arm completed')
    market.publish([f'experiments/results/token_language/{t}.json' for t in done], 'Complete R1 depth per fresh token: deep keyed L2/L8, one pass over 4M tokens')


def main():
    for stage in (depth_ladder, stream_depth, v5.keyed_scaling):
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
