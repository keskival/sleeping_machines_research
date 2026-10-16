"""Curie chain v12 (supersedes v11 while the part-whole v2 fit runs; that fit is awaited unchanged). Resumable stages.

User direction 15 October: boost higher-level concept learning (slower levels, larger receptive fields, attention over
part-whole trees). theory/PART_WHOLE_COMPOSITION.md sections 5-6.
1. Part-whole v2 fit (running) → decision v2 (P6a-c against the one-pass L2 base) → publication.
2. Concept-hierarchy contracts at 1024 tokens incl. scoring-shape memory.
3. Plan-B L2 base on 1024-token lanes.
4. Concept hierarchy + tree read: smoke (cost gate) → fit.
5. Concept hierarchy without the tree read.
6. Document-concept curves for every one-pass model → concept decision (P7a-c) → publication.
7. One-pass L8, part-whole no-future ablation (P6d), keyed 16.8M, B10 eight-layer stage.
"""
import json
import subprocess
import continue_market_depth_v2 as market
import continue_curie_deep_recovery_v5 as v5
from continue_curie_deep_recovery_v8 import ROOT, PY, completed, run_or_await
import continue_curie_deep_language as original
from continue_curie_deep_recovery import stable_host
import continue_curie_deep_recovery_v11 as v11

CONTRACTS = 'curie_concept_hierarchy_contracts_20261015_v1'
BASE = 'curie_r1_deep_keyed_stream_L2_4M_l1024_s0_20261015_v1'
CT_SMOKE, CT, CN = 'curie_r1_concept_tree_4M_l1024_smoke_20261015_v1', 'curie_r1_concept_tree_4M_l1024_s0_20261015_v1', 'curie_r1_concept_4M_l1024_s0_20261015_v1'
CURVE, DECISION = 'curie_document_concept_curve_20261015_v2', 'curie_concept_decision_20261015_v1'


def diag(tag):
    path = ROOT / f'experiments/results/diagnostics/{tag}.json'
    return json.loads(path.read_text()) if path.exists() else None


def part_whole_v2():
    run_or_await(v11.FIT, 64800)
    v11.decide('curie_part_whole_decision_20261015_v2', [v11.L2, v11.FIT, v11.L8], 'R1 part-whole hierarchy v2 vs one-pass deep keyed L2 (P6a-c)', (v11.SMOKE,))


def contracts():
    if not diag(CONTRACTS): stable_host(); original.run(CONTRACTS, 7200)
    rec = diag(CONTRACTS); print(json.dumps({'concept_contracts': rec['status'], 'checks': rec['checks'], 'shapes': rec['measurements'].get('scoring_shapes')}), flush=True)
    if rec['status'] != 'completed': raise RuntimeError('Concept-hierarchy contracts failed')


def concept_tree():
    if not (v11.result(CT_SMOKE) or {}).get('status') == 'smoke_completed':
        stable_host(); original.run(CT_SMOKE, 7200)
    smoke = v11.result(CT_SMOKE); print(json.dumps({'concept_tree_smoke': {k: smoke[k] for k in ('s_per_step', 'max_rss_kb')}}), flush=True)
    if smoke['max_rss_kb'] > 2_800_000 or smoke['s_per_step'] > 12: raise RuntimeError('Concept+tree smoke exceeds the admitted cost envelope')
    run_or_await(CT, 64800)


def concept_decision():
    runs = [t for t in (BASE, CT, CN, v11.L2, v11.FIT) if completed(t)]
    if not diag(CURVE): subprocess.run([PY, 'analysis/document_concept_curve.py', '--tag', CURVE, '--runs', *runs], cwd=ROOT, check=True)
    subprocess.run([PY, 'analysis/concept_decision.py', '--tag', DECISION, '--base', BASE, '--tree', CT, '--no-tree', CN, '--curve', CURVE], cwd=ROOT, check=True)
    paths = [f'experiments/results/diagnostics/{t}.json' for t in (CONTRACTS, CURVE, DECISION)] + [f'experiments/results/token_language/{t}.json' for t in (BASE, CT_SMOKE, CT, CN) if v11.result(t)]
    market.publish(paths, 'R1 concept hierarchy (five observed levels, concept memories, tree read) vs plan-B L2 on 1024-token lanes (P7a-c)')


def main():
    stages = [('part_whole_v2', part_whole_v2), ('contracts', contracts), ('base_l1024', lambda: run_or_await(BASE, 28800)), ('concept_tree', concept_tree),
              ('concept_no_tree', lambda: run_or_await(CN, 64800)), ('concept_decision', concept_decision),
              ('l8', lambda: run_or_await(v11.L8, 50400)), ('nofuture', lambda: run_or_await(v11.NOFUT, 64800)),
              ('decision_p6d', lambda: v11.decide('curie_part_whole_decision_p6d_20261015_v2', [v11.L2, v11.FIT, v11.NOFUT, v11.L8], 'R1 part-whole hierarchy: level-matched future targets ablation (P6d)')),
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
