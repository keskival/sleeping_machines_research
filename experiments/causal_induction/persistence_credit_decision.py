"""Completed strong conditional-credit comparison; original gate unchanged."""
import json
import time
from pathlib import Path
from world_credit_pipeline import ROOT, RESULTS, ACQUISITION, verify_result, sha

TAG = 'curie_persistence_credit_20261012_v1_decision'
def main():
    begin = time.monotonic(); out = RESULTS/(TAG+'.json'); assert not out.exists()
    source = verify_result(ACQUISITION)
    cases = [verify_result(f'curie_persistence_credit_20261012_v1_case{i}') for i in range(8)]
    references = {arm:verify_result(f'curie_world_credit_20261012_v2_{arm}_reuse') for arm in ['exact','learned','prior']}
    rows=[]; parents={}; sources={}
    for i,case in enumerate(cases):
        assert case['status']=='completed' and case['case']==i and not case['smoke'] and case['targets']==128
        row=case['rows'][0]
        assert row['incoming_learner_fingerprint']==source['checks']['learner_fingerprint']
        assert case['source_checkpoint_sha256']==source['checks']['checkpoint_sha256']
        assert row['persistence_prediction_parity_error']==0.
        for arm,ref in references.items():
            other=ref['rows'][i]
            assert all(row[k]==other[k] for k in ['path','targets','depth','width','variant','incoming_learner_fingerprint'])
            assert [(r['query_id'],r['issued_time'],r['feedback_time'],r['target']) for r in row['issued_trace']]==[(r['query_id'],r['issued_time'],r['feedback_time'],r['target']) for r in other['issued_trace']]
        rows.append(row)
        parents[str((RESULTS/(case['tag']+'.json')).relative_to(ROOT))]=sha(RESULTS/(case['tag']+'.json'))
        for name,digest in case['source_sha256'].items():
            assert name not in sources or sources[name]==digest; sources[name]=digest
    for ref in references.values():
        parents[str((RESULTS/(ref['tag']+'.json')).relative_to(ROOT))]=sha(RESULTS/(ref['tag']+'.json'))
    sources[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    nll=sum(r['nll']*r['targets'] for r in rows)/1024
    gains={arm:nll-ref['pooled_nll'] for arm,ref in references.items()}
    depth_gains={str(d):sum(r['nll']-other['nll'] for r,other in zip(rows,references['learned']['rows']) if r['depth']==d)/4 for d in [4,8]}
    source_fit=source['confidence_acquisition_wall_s']+sum(r['wall_s'] for r in source['rows'])
    result=dict(status='completed',battle='B5/R1',tag=TAG,rows=rows,targets=1024,
        persistence_credit_nll=nll, learned_credit_nll=references['learned']['pooled_nll'],
        quality_gains_vs_persistence=gains, learned_depth_gains_vs_persistence=depth_gains,
        strong_control_gate=gains['learned']>0 and all(v>0 for v in depth_gains.values()),
        reuse_fitting_wall_s=sum(r['wall_s'] for r in rows), source_acquisition_wall_s=source_fit,
        complete_acquisition_and_reuse_stage_wall_s=source['whole_pipeline_wall_s']+sum(c['whole_pipeline_wall_s'] for c in cases),
        peak_rss_kb=max(c['peak_rss_kb'] for c in cases),
        parent_sha256=parents,source_sha256=sources,
        whole_analysis_wall_s=time.monotonic()-begin,
        scope='Additional pre-registered strong causal credit control, eight independent matched source deployments/one actor seed. Positive gains mean the named arm improves on persistence credit. Original v2 primary gate stays failed. All native critic/alternative/optimizer work retained and charged; no minimal-cost baseline, public test, family-level or neural-depth claim.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['tag','persistence_credit_nll','quality_gains_vs_persistence','learned_depth_gains_vs_persistence','strong_control_gate']}),flush=True)
if __name__=='__main__': main()
