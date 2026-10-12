"""Aggregate completed independent v2 deployments without fitting."""
import argparse
import json
import time
from pathlib import Path
from native_sensor import ROOT
from world_credit_pipeline import verify_result, sha, ACQUISITION, RESULTS


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True)
    ap.add_argument('--arm', choices=['exact', 'learned', 'prior'], required=True); a = ap.parse_args()
    begin = time.monotonic(); out = RESULTS/(a.tag+'.json'); assert not out.exists()
    source = verify_result(ACQUISITION)
    cases = [verify_result(f'curie_world_credit_20261012_v2_{a.arm}_case{i}') for i in range(8)]
    rows = []; sources = {}; parents = {}
    for i, case in enumerate(cases):
        assert case['arm'] == a.arm and case['case'] == i and case['targets'] == 128
        assert case['rows'][0]['incoming_learner_fingerprint'] == source['checks']['learner_fingerprint']
        assert case['source_checkpoint_sha256'] == source['checks']['checkpoint_sha256']
        rows.extend(case['rows'])
        parents[f'experiments/results/causal_induction/{case["tag"]}.json'] = sha(RESULTS/(case['tag']+'.json'))
        for name, digest in case['source_sha256'].items():
            assert name not in sources or sources[name] == digest; sources[name] = digest
    sources[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    targets = sum(r['targets'] for r in rows)
    source_wall = source['confidence_acquisition_wall_s']+sum(r['wall_s'] for r in source['rows'])
    fit_wall = sum(r['wall_s'] for r in rows)
    result = dict(status='completed', battle='B5/R1', tag=a.tag, mode='reuse', arm=a.arm,
                  checks=dict(identical_common_source=True, source_checkpoint_sha256=source['checks']['checkpoint_sha256'],
                              acquisition_result_sha256=sha(RESULTS/(ACQUISITION+'.json'))),
                  rows=rows, targets=targets,
                  pooled_nll=sum(r['nll']*r['targets'] for r in rows)/targets,
                  pooled_world_nll=sum(r['world_nll']*r['targets'] for r in rows)/targets,
                  confidence_acquisition_targets=0, confidence_acquisition_wall_s=0.,
                  fitting_wall_s=fit_wall, source_acquisition_wall_s=source_wall,
                  whole_learning_wall_s=source_wall+fit_wall,
                  whole_pipeline_wall_s=sum(c['whole_pipeline_wall_s'] for c in cases)+time.monotonic()-begin,
                  peak_rss_kb=max(c['peak_rss_kb'] for c in cases),
                  parent_sha256=source['parent_sha256'], result_parent_sha256=parents,
                  data_records=[r for c in cases for r in c['data_records']],
                  source_sha256=sources,
                  scope='Completed independently guarded case deployments, original v1 math/source/optimizer/seeds/cases and gate unchanged. Source acquisition shared once per8deployment cohort,1024prequential targets. All native outcome/memory/alternative work matched; exact/learned/prior differ source-choice teaching. Per-case preprocessing/load/model-copy cost charged in stage wall. V1 failed execution preserved separately as development overhead; no efficiency or neural-depth claim.')
    # Check the completed repair against preserved deterministic v1 partial scores.
    if a.arm == 'exact':
        failure = ROOT/'experiments/results/causal_induction/curie_world_credit_20261012_v1_execution_stop.json'
        stopped = json.loads(failure.read_text()); assert stopped['status'] == 'stopped'
        errors = []
        for old in stopped['completed_case_metrics']:
            new = rows[old['episode']]
            errors.append(max(abs(new['nll']-old['nll']), abs(new['world_nll']-old['world_nll'])))
        assert len(errors) == 7 and max(errors) < 1e-12
        result['checks']['v1_partial_case_parity_error'] = max(errors)
        result['result_parent_sha256'][str(failure.relative_to(ROOT))] = sha(failure)
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(tag=a.tag, nll=result['pooled_nll'], world_nll=result['pooled_world_nll'],
                          whole_fit_wall_s=result['whole_learning_wall_s'], peak_rss_kb=result['peak_rss_kb'])), flush=True)


if __name__ == '__main__': main()
