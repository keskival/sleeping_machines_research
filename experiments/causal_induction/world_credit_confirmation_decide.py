"""Four-control prospective confirmation; diagnose every completed stage."""
import argparse
import json
import math
import time
from pathlib import Path
from world_credit_confirmation_verify import verify_result
from world_credit_pipeline import ROOT,RESULTS,ACQUISITION,sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--seed',type=int,choices=[194,195,196],required=True);a=ap.parse_args()
 begin=time.monotonic();out=RESULTS/(a.tag+'.json');assert not out.exists()
 source_tag=ACQUISITION if a.seed==194 else f'curie_world_credit_confirm_20261012_v1_s{a.seed}_acquisition';source=verify_result(source_tag)
 parents={str((RESULTS/(source_tag+'.json')).relative_to(ROOT)):sha(RESULTS/(source_tag+'.json'))};sources=dict(source['source_sha256']);arms={}
 for arm in ['exact','learned','prior','persistence']:
  cases=[verify_result(f'curie_world_credit_confirm_20261012_v1_s{a.seed}_{arm}_case{i}') for i in range(8)];rows=[]
  for i,c in enumerate(cases):
   assert c['seed']==a.seed and c['arm']==arm and c['case']==i and c['targets']==128
   assert c['source_checkpoint_sha256']==source['checks']['checkpoint_sha256']
   assert c['rows'][0]['incoming_learner_fingerprint']==source['checks']['learner_fingerprint']
   if arm=='persistence':assert c['rows'][0]['persistence_prediction_parity_error']==0.
   rows.extend(c['rows']);parents[str((RESULTS/(c['tag']+'.json')).relative_to(ROOT))]=sha(RESULTS/(c['tag']+'.json'))
   for n,h in c['source_sha256'].items():assert n not in sources or sources[n]==h;sources[n]=h
  full=source['whole_pipeline_wall_s']+sum(c['whole_pipeline_wall_s'] for c in cases)
  arms[arm]=dict(arm=arm,rows=rows,pooled_nll=sum(r['nll'] for r in rows)/8,world_nll=sum(r['world_nll'] for r in rows)/8,
   complete_acquisition_and_reuse_stage_wall_s=full,whole_learning_wall_s=source['whole_learning_wall_s']+sum(r['wall_s'] for r in rows),
   reuse_ms_per_target=1000*sum(r['wall_s'] for r in rows)/1024,peak_rss_kb=max(c['peak_rss_kb'] for c in cases))
 gains={arm:arms[arm]['pooled_nll']-arms['learned']['pooled_nll'] for arm in ['exact','prior','persistence']}
 depths={str(d):{arm:sum(b['nll']-r['nll'] for r,b in zip(arms['learned']['rows'],arms[arm]['rows']) if r['depth']==d)/4 for arm in ['exact','prior','persistence']} for d in [4,8]}
 for arm in ['exact','prior','persistence']:
  for x,y in zip(arms['learned']['rows'],arms[arm]['rows']):
   assert all(x[k]==y[k] for k in ['path','depth','width','variant','incoming_learner_fingerprint'])
   assert [(r['issued_time'],r['feedback_time'],r['query_id'],r['target']) for r in x['issued_trace']]==[(r['issued_time'],r['feedback_time'],r['query_id'],r['target']) for r in y['issued_trace']]
 # Proper losses/calibration are descriptive issued-before-feedback diagnostics.
 calibration=[]
 for arm,data in arms.items():
  trace=[t for r in data['rows'] for t in r['issued_trace']]
  calibration.append(dict(arm=arm,actor_nll=data['pooled_nll'],world_nll=data['world_nll'],
   world_actor_nll_gap=data['world_nll']-data['pooled_nll'],
   actor_brier=sum((t['probability']-t['target'])**2 for t in trace)/1024,
   world_brier=sum((t['world_probability']-t['target'])**2 for t in trace)/1024))
 sources[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
 gate=all(g>0 for g in gains.values()) and all(g>0 for r in depths.values() for g in r.values()) and arms['learned']['world_nll']<math.log(2)
 result=dict(status='completed',battle='B5/R1',tag=a.tag,seed=a.seed,rows=list(arms.values()),quality_gains=gains,depth_quality_gains=depths,
  confirmation_gate=gate,causal_calibration_diagnostics=calibration,source_sha256=sources,parent_sha256=parents,
  whole_analysis_wall_s=time.monotonic()-begin,
  next_decision=('unchanged fresh-source seed'+str(a.seed+1) if gate and a.seed<196 else 'three-stage confirmation complete' if gate else 'stop repeat acquisitions; diagnose learned outcome information versus generic smoothing and native forward-response correction'),
  scope='Fixed fresh-system four-arm confirmation stage; all8systems/1024prequential targets per arm, generating depths4/8 and factors8/16. Same source model/Adam fingerprint within each stage. Source acquisition charged once per8deployment cohort, whole denominator5120presentations. No case/actor selection, publicTEST, neural-depth or efficiency claim. All three registered stages must pass for confirmation; an early failure stops later acquisitions.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['seed','quality_gains','depth_quality_gains','confirmation_gate','next_decision']}),flush=True)
if __name__=='__main__':main()
