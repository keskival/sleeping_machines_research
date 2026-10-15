"""Curie chain v11 (supersedes v10: part-whole jobs re-versioned v2 after the evaluation-memory fix; the killed v1 fit never reached its first checkpoint). Resumable stages.

User direction 15 October: design depth from the data's part-whole structure (theory/PART_WHOLE_COMPOSITION.md).
1. One-pass 4M deep keyed L2 (running) = the base of the part-whole hierarchy.
2. Part-whole hierarchy: smoke (cost gate) then the 4M one-pass fit on the same tokens/seed/schedule.
3. One-pass 4M deep keyed L8 (stacking reference).
4. Decision analysis (P6a-c) and publication.
5. Part-whole without level-matched future targets (P6d), decision update, publication.
6. Keyed member 16.8M 1024-lane fit (v5 stage), 7. B10 eight-layer market stage.
"""
import json
import subprocess
import continue_market_depth_v2 as market
import continue_curie_deep_recovery_v5 as v5
from continue_curie_deep_recovery_v8 import ROOT, PY, completed, run_or_await
import continue_curie_deep_language as original
from continue_curie_deep_recovery import stable_host

L2, L8 = 'curie_r1_deep_keyed_stream_L2_4M_s0_20261015_v1', 'curie_r1_deep_keyed_stream_L8_4M_s0_20261015_v1'
SMOKE, FIT, NOFUT = 'curie_r1_part_whole_4M_smoke_20261015_v2', 'curie_r1_part_whole_4M_s0_20261015_v2', 'curie_r1_part_whole_nofuture_4M_s0_20261015_v2'


def result(tag):
    path = ROOT / f'experiments/results/token_language/{tag}.json'
    return json.loads(path.read_text()) if path.exists() else None


def part_whole():
    if not (result(SMOKE) or {}).get('status') == 'smoke_completed':
        stable_host(); original.run(SMOKE, 3600)
    smoke = result(SMOKE); print(json.dumps({'part_whole_smoke': {k: smoke[k] for k in ('s_per_step', 'max_rss_kb')}}), flush=True)
    if smoke['max_rss_kb'] > 2_800_000 or smoke['s_per_step'] > 12: raise RuntimeError('Part-whole smoke exceeds the admitted cost envelope')
    run_or_await(FIT, 64800)


def decide(tag, runs, message, extra=()):
    subprocess.run([PY, 'analysis/part_whole_decision.py', '--tag', tag, '--runs', *runs], cwd=ROOT, check=True)
    paths = [f'experiments/results/diagnostics/{tag}.json'] + [f'experiments/results/token_language/{t}.json' for t in (*runs, *extra) if result(t)]
    market.publish(paths, message)


def main():
    stages = [('l2', lambda: run_or_await(L2, 21600)), ('part_whole', part_whole), ('l8', lambda: run_or_await(L8, 50400)),
              ('decision', lambda: decide('curie_part_whole_decision_20261015_v2', [L2, FIT, L8], 'R1 part-whole hierarchy vs one-pass deep keyed L2/L8 (P6a-c)', (SMOKE,))),
              ('nofuture', lambda: run_or_await(NOFUT, 64800)),
              ('decision_p6d', lambda: decide('curie_part_whole_decision_p6d_20261015_v2', [L2, FIT, NOFUT, L8], 'R1 part-whole hierarchy: level-matched future targets ablation (P6d)')),
              ('keyed_scaling', v5.keyed_scaling)]
    for name, stage in stages:
        try: stage()
        except Exception as error: print(json.dumps({name: 'failed', 'error': repr(error)}), flush=True)
    try:
        market.market_stage()
    except Exception as error:
        print(json.dumps({'market_stage': 'failed', 'error': repr(error)}), flush=True)
        failure = ROOT / f'experiments/results/market/{market.FIT}_failure.json'
        with failure.open('x') as f: json.dump(dict(status='failed', battle='B10', error=repr(error), smoke=market.SMOKE, fit=market.FIT), f, indent=2); f.write('\n')
    print('Queue complete', flush=True)


if __name__ == '__main__':
    main()
