"""R1 aggregate completed matched DEV evidence; no inference or training runtime."""
import argparse,hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
 out=ROOT/f'experiments/results/diagnostics/{a.tag}.json'
 if out.exists():raise FileExistsError(out)
 inputs={};sources={};rows={};protocol=None
 def record(path):
  r=json.loads(path.read_text());assert r['status']=='completed';inputs[str(path.relative_to(ROOT))]=sha(path)
  for name,digest in {**r.get('source_sha256',{}),**r.get('input_sha256',{})}.items():assert sha(ROOT/name)==digest,name
  sources.update(r.get('source_sha256',{}));return r
 paired=record(ROOT/'experiments/results/diagnostics/aws_multidepth_8m_error_buckets_20261013_v1.json')
 pairs=[('aws_bernoulli_teacher_8m_20261013_s6_v2',paired['rows'][0]),('aws_multidepth_readout_8m_20261013_s6_v1',paired['rows'][1])]
 for path in sorted((ROOT/'experiments/results/diagnostics').glob('aws_*d*_8m_quality_*.json')):
  r=record(path)
  for row in r['rows']:pairs.append((row['variant'],row))
 for tag,row in pairs:
  scored=row['scored'];p={k:scored[k] for k in ('targets','target_start','target_end_inclusive','window','lanes','eos_state_reset')}
  if protocol is None:protocol=p
  assert p==protocol,'Never combine scoring protocols'
  r=record(ROOT/f'experiments/results/token_language/{tag}.json')
  assert r['presentations_total']==8388352 and r['args']['seed']==6
  entry=dict(tag=tag,depth=r['args']['depth'],acquisition=r['presentations_total'],parameters=r['parameters'],paired_dev_nll=scored['nll'],full_training_s=r['train_wall_s_total'],fit_ms_per_target=1000*r['train_wall_s_total']/r['presentations_total'],selected_writes_per_target=r['args']['depth']*r['args']['heads'],scored_keys_per_target=r['args']['depth']*r['args']['heads']*r['args']['pool'],buckets=row['buckets'])
  if tag in rows:assert abs(rows[tag]['paired_dev_nll']-entry['paired_dev_nll'])<1e-12
  rows[tag]=entry
 baseline=rows['aws_bernoulli_teacher_8m_20261013_s6_v2']
 for row in rows.values():
  row['gain_vs_baseline']=baseline['paired_dev_nll']-row['paired_dev_nll']
  row['training_time_ratio']=row['full_training_s']/baseline['full_training_s']
  row['quality_time_frontier']=not any(other['paired_dev_nll']<=row['paired_dev_nll'] and other['full_training_s']<=row['full_training_s'] and (other['paired_dev_nll']<row['paired_dev_nll'] or other['full_training_s']<row['full_training_s']) for other in rows.values())
 d=dict(status='completed',battle='R1',tag=a.tag,rows=sorted(rows.values(),key=lambda r:r['paired_dev_nll']),protocol=protocol,input_sha256=inputs,source_sha256=dict(sources,**{str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}),scope='Re-analysis of completed final weights, same8M acquisition/seed6/8192DEV/1024history/alltargets/noEOSreset; no new scoring, fitting, selection or public validation. Pareto frontier uses measured complete CPU training time, not FLOPs, ASIC energy or scaled Transformer comparisons. Eval cadence256vs1024 and readout sizes differ where recorded in source args.')
 out.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps([{k:v for k,v in r.items() if k!='buckets'} for r in d['rows']],indent=2))
if __name__=='__main__':main()
