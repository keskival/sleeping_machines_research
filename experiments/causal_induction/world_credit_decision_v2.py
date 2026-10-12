"""Completed common-acquisition credit comparison; no fitting or tuning."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
from public_stream import records, persistence
from native_sensor import ROOT
from world_credit_pipeline import verify_result, ACQUISITION, RESULTS


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); a = ap.parse_args()
    out = RESULTS/(a.tag+'.json'); assert not out.exists(); begin = time.monotonic()
    source = verify_result(ACQUISITION)
    arms = {arm: verify_result(f'curie_world_credit_20261012_v2_{arm}_reuse')
            for arm in ['exact', 'learned', 'prior']}
    expected_fingerprint = source['checks']['learner_fingerprint']
    for arm, result in arms.items():
        assert result['arm'] == arm and result['targets'] == 1024
        assert all(row['incoming_learner_fingerprint'] == expected_fingerprint for row in result['rows'])
        for name, digest in result['result_parent_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    assert len({r['checks']['source_checkpoint_sha256'] for r in arms.values()}) == 1
    cases = arms['exact']['rows']
    for result in arms.values():
        assert [r['path'] for r in result['rows']] == [r['path'] for r in cases]
        for row, exact in zip(result['rows'], cases):
            labels = [(r['query_id'], r['issued_time'], r['feedback_time'], r['target']) for r in row['issued_trace']]
            assert labels == [(r['query_id'], r['issued_time'], r['feedback_time'], r['target']) for r in exact['issued_trace']]
            assert all(r['issued_time'] < r['feedback_time'] for r in row['issued_trace'])
    # Cheap causally updated public-input diagnostics, not new dense references.
    positives, total, persistence_prior = 1., 2., (1., 2.)
    for row in source['rows']:
        path = ROOT/row['path']
        _, persistence_prior = persistence(path, persistence_prior, True)
        for r in records(path):
            if r['kind'] == 'feedback': positives += r['target']; total += 1
    diagnostic_rows = []
    for case in cases:
        correct_prior = persistence_prior
        persist, _ = persistence(ROOT/case['path'], correct_prior, True)
        successes, trials = positives, total
        frequency_scores = []
        issued = {}
        for r in records(ROOT/case['path']):
            if r['kind'] == 'query': issued[r['query_id']] = successes/trials
            elif r['kind'] == 'feedback':
                p = issued.pop(r['query_id']); y = r['target']
                frequency_scores.append(-math.log(max(1e-12, p if y else 1-p)))
                successes += y; trials += 1
        diagnostic_rows.append(dict(path=case['path'], persistence_nll=persist['nll'],
                                    online_global_frequency_nll=float(np.mean(frequency_scores)),
                                    neutral_nll=math.log(2), targets=case['targets']))
    gains = {control: arms[control]['pooled_nll']-arms['learned']['pooled_nll']
             for control in ['exact', 'prior']}
    depth_gains = {}
    for depth in [4, 8]:
        means = {arm: np.mean([row['nll'] for row in result['rows'] if row['depth'] == depth])
                 for arm, result in arms.items()}
        depth_gains[str(depth)] = {control: float(means[control]-means['learned'])
                                  for control in ['exact', 'prior']}
    gate = all(g > 0 for g in gains.values()) and all(g > 0 for group in depth_gains.values() for g in group.values()) and arms['learned']['pooled_world_nll'] < math.log(2)
    comparison = []
    for arm, result in arms.items():
        complete_wall = source['whole_pipeline_wall_s']+result['whole_pipeline_wall_s']
        comparison.append(dict(arm=arm, pooled_nll=result['pooled_nll'], world_nll=result['pooled_world_nll'],
                               acquisition_wall_s=result['source_acquisition_wall_s'],
                               reuse_wall_s=result['fitting_wall_s'], whole_learning_wall_s=result['whole_learning_wall_s'],
                               complete_acquisition_and_reuse_stage_wall_s=complete_wall,
                               acquisition_target_presentations=4096, reuse_target_presentations=1024,
                               whole_ms_per_presentation=1000*complete_wall/5120,
                               reuse_ms_per_target=1000*result['fitting_wall_s']/1024,
                               average_single_deployment_wall_s=source['whole_pipeline_wall_s']+result['whole_pipeline_wall_s']/8))
    case_gains = []
    for i, case in enumerate(cases):
        case_gains.append(dict(path=case['path'], depth=case['depth'], width=case['width'], variant=case['variant'],
                               gains={control: arms[control]['rows'][i]['nll']-arms['learned']['rows'][i]['nll'] for control in ['exact', 'prior']},
                               second_half_gains={control: arms[control]['rows'][i]['second_half_nll']-arms['learned']['rows'][i]['second_half_nll'] for control in ['exact', 'prior']}))
    parents = [RESULTS/(ACQUISITION+'.json')]+[RESULTS/f'curie_world_credit_20261012_v2_{arm}_reuse.json' for arm in arms]
    parents += [ROOT/'experiments/causal_induction/WORLD_CREDIT_DECISION.md']
    sources = {str(Path(__file__).relative_to(ROOT)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for result in [source, *arms.values()]:
        for name, digest in result['source_sha256'].items():
            assert name not in sources or sources[name] == digest; sources[name] = digest
    result = dict(status='completed', battle='B5/R1', tag=a.tag, development_gate=gate,
                  rows=comparison, quality_gains=gains, depth_quality_gains=depth_gains,
                  cases=case_gains, causal_count_diagnostics=diagnostic_rows,
                  learned_world_gain_vs_frequency=float(np.mean([x['online_global_frequency_nll'] for x in diagnostic_rows]))-arms['learned']['pooled_world_nll'],
                  learned_actor_gain_vs_persistence=float(np.mean([x['persistence_nll'] for x in diagnostic_rows]))-arms['learned']['pooled_nll'],
                  parent_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in parents},
                  source_sha256=sources, whole_analysis_wall_s=time.monotonic()-begin,
                  scope='One exploratory actor seed, all8declared nonlinear cases/1024issued-before-feedback targets per arm. Common native world acquisition and complete charged work; exact and prior retain world learning/shared memory. Generating depth4/8 and factors8/16, not neural-depth scaling. Cheap prequential frequency/persistence diagnostics use public source/target observations only. No publicTEST, new dense reference, calibrated uncertainty/causal-response embedding or quality-matched efficiency claim.')
    out.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
