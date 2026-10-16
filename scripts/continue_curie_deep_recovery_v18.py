"""Curie chain v18 (supersedes v17 while 16M L8 runs; that job is awaited unchanged). Resumable stages.

Route (16 Oct 10:40, user: train our own models against published benchmarks first): the published word-level PTB
battle opens right after 16M L8, then the 16M L2 partner (P5d), then the v17 order unchanged.
1. 16M one-pass L8 (running) → PTB driver smoke → PTB d256 L8 development fit (validation only) → validation scoring
   (frozen + causal online adaptation) → publication.
2. 16M one-pass L2 → L4 exact resume → part-whole levels on L8 → publication (v17 depth_data).
3. v15 stages unchanged.
"""
import json
from pathlib import Path
import continue_market_depth_v2 as market
from continue_curie_deep_recovery_v8 import completed, run_or_await
import continue_curie_deep_language as original
from continue_curie_deep_recovery import stable_host
import continue_curie_deep_recovery_v11 as v11
import continue_curie_deep_recovery_v12 as v12
import continue_curie_deep_recovery_v14 as v14
import continue_curie_deep_recovery_v15 as v15
import continue_curie_deep_recovery_v16 as v16

ROOT = Path(__file__).resolve().parents[1]
PTB_SMOKE, PTB_FIT, PTB_VALID = 'curie_ptb_native_smoke_20261016_v1', 'curie_ptb_native_d256_L8_s0_20261016_v1', 'curie_ptb_native_d256_L8_s0_valid_score_20261016_v1'


def word_done(tag):
    path = ROOT/f'experiments/results/word_language/{tag}.json'
    return path.exists() and json.loads(path.read_text()).get('status') == 'completed'


def ptb():
    log = ROOT/f'experiments/queue/logs/{PTB_SMOKE}.log'
    lines = [l for l in log.read_text().splitlines() if l.startswith('{"smoke_steps"')] if log.exists() else []
    if not lines:
        stable_host(); original.run(PTB_SMOKE, 3600)
        lines = [l for l in log.read_text().splitlines() if l.startswith('{"smoke_steps"')]
    s = json.loads(lines[-1]); print(json.dumps({'ptb_smoke': s}), flush=True)
    if s['max_rss_kb'] > 2_600_000 or s['tokens_per_s'] < 300: raise RuntimeError('PTB d256 L8 smoke exceeds the admitted cost envelope')
    if not word_done(PTB_FIT): stable_host(); original.run(PTB_FIT, 57600)
    if not word_done(PTB_VALID): stable_host(); original.run(PTB_VALID, 14400)
    market.publish([f'experiments/results/word_language/{t}.json' for t in (PTB_FIT, PTB_VALID)] + ['experiments/results/diagnostics/curie_ptb_native_contracts_20261016_v2.json'],
                   'R1 PTB development round 1: native d256 L8 deep keyed member, validation frozen and causal online adaptation')


def depth_data():
    for tag, timeout in ((v16.L8_16, 64800),):
        try: run_or_await(tag, timeout)
        except Exception as error: print(json.dumps({'depth_16m': tag, 'failed': repr(error)}), flush=True)
    try: ptb()
    except Exception as error: print(json.dumps({'ptb': 'failed', 'error': repr(error)}), flush=True)
    for tag, timeout in ((v16.L2_16, 43200), (v14.L4, 36000), (v14.PW8, 64800)):
        try: run_or_await(tag, timeout)
        except Exception as error: print(json.dumps({'depth_ladder': tag, 'failed': repr(error)}), flush=True)
    done = [t for t in (v11.L2, v14.L4, v11.L8, v16.L8_16, v16.L2_16, v14.PW8) if completed(t)]
    extra = [f'experiments/results/diagnostics/{v14.BENCH}.json'] if v12.diag(v14.BENCH) else []
    market.publish([f'experiments/results/token_language/{t}.json' for t in done] + extra, 'R1 depth x data: one-pass deep keyed ladder at 4M and 16M fresh tokens, part-whole levels on L8')


def main():
    stages = [('depth_data', depth_data), ('contracts', v12.contracts), ('base_l1024', lambda: run_or_await(v12.BASE, 28800)),
              ('type_closed', v15.type_closed), ('concept_tree', v12.concept_tree), ('concept_no_tree', lambda: run_or_await(v12.CN, 64800)),
              ('recursive', lambda: run_or_await(v15.REC, 64800)), ('concept_decision', v12.concept_decision), ('publish_new_arms', v15.publish_new_arms),
              ('nofuture', lambda: run_or_await(v11.NOFUT, 64800)),
              ('decision_p6d', lambda: v11.decide('curie_part_whole_decision_p6d_20261015_v2', [v11.L2, v11.FIT, v11.NOFUT, v11.L8], 'R1 part-whole hierarchy: level-matched future targets ablation (P6d)')),
              ('keyed_scaling', v15.v5.keyed_scaling)]
    for name, stage in stages:
        try: stage()
        except Exception as error: print(json.dumps({name: 'failed', 'error': repr(error)}), flush=True)
    try:
        market.market_stage()
    except Exception as error:
        print(json.dumps({'market_stage': 'failed', 'error': repr(error)}), flush=True)
    print('Queue complete', flush=True)


if __name__ == '__main__':
    main()
