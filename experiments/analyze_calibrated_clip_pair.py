"""Completed paired R1 verdict with inherited and auxiliary cost boundaries."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();out=ROOT/f'experiments/results/diagnostics/{a.tag}.json';assert not out.exists();inputs={};rows=[];protocol=None
 for policy in ['global','calibrated']:
  prefix=f'aws_calibrated_clip_pair_{policy}_20261014_s6_v1';names=[f'experiments/results/token_language/{prefix}_2m.json',f'experiments/results/token_language/{prefix}_smoke.json',f'experiments/results/diagnostics/{prefix}_2m_quality.json'];records=[]
  for name in names:
   path=ROOT/name;r=json.loads(path.read_text());assert r['status']=='completed';inputs[name]=sha(path)
   for file,digest in {**r['source_sha256'],**r.get('input_sha256',{})}.items():assert sha(ROOT/file)==digest,file
   records.append(r)
  fit,smoke,quality=records;scored=quality['rows'][-1]['scored'];observed={k:scored[k] for k in ('targets','target_start','target_end_inclusive','window','lanes','eos_state_reset')}
  if protocol is None:protocol=observed
  assert observed==protocol
  qwork=sum(r['diagnostic_s'] for r in quality['rows'])
  incremental_training=fit['added_training_s']+smoke['added_training_s']
  full_process_proxy=fit['process_wall_s']+smoke['process_wall_s']+qwork+fit['calibration_wall_s']
  rows.append(dict(policy=policy,nll=scored['nll'],parent=fit['parent'],added_updates=fit['added_updates'],added_targets=fit['added_targets'],total_actor_targets=fit['presentations_total'],inherited_training_s=fit['inherited_training_s'],incremental_training_including_smoke_s=incremental_training,calibration_s=fit['calibration_wall_s'],evaluation_scoring_s=qwork,incremental_process_and_scoring_proxy_s=full_process_proxy,whole_fit_and_auxiliary_proxy_s=fit['inherited_training_s']+full_process_proxy,calibration_factual_targets=fit['calibration_factual_targets'],max_rss_kb=fit['max_rss_kb']))
 g,c=rows;assert (g['parent'],g['added_updates'],g['added_targets'],g['total_actor_targets'])==(c['parent'],c['added_updates'],c['added_targets'],c['total_actor_targets'])
 gain=g['nll']-c['nll'];ratio=c['incremental_process_and_scoring_proxy_s']/g['incremental_process_and_scoring_proxy_s']
 if gain>=.01 and ratio<=1.01:decision='Advance calibratedlocal: quality/work gate passes; confirm and integrate useful depth/consequence credit'
 elif gain>=.01:decision='Retain calibratedlocal quality improvement; optimize measured overhead before efficiency claim'
 elif gain>=-.01:decision='Quality retained within0.01; locality engineering point, no competitivequalitywin. Measure useful async execution before more fitting'
 else:decision='Calibratedlocal quality fails; keep global reference and diagnose direction/moment/tail response before next fit'
 d=dict(status='completed',battle='R1',rows=rows,protocol=protocol,nll_gain=gain,incremental_work_ratio=ratio,decision=decision,input_sha256=inputs,source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))},scope='One paired2Mcontinuation/seed6 from identical acquired8MbankD8learner, same actual data/order/teaching. Calibrated policy also uses16kTRAINcalibration exposures; work charged once. Process/scoring proxy excludes unspecified transfer/publication time, uses recorded diagnostics rather than a full profiler, and is not FLOPs/energy. Norm retention alone is not quality evidence; no publicbenchmark or hardwareclaim.')
 with out.open('x') as f:json.dump(d,f,indent=2)
 print(json.dumps(dict(gain=gain,work_ratio=ratio,decision=decision)))
if __name__=='__main__':main()
