"""Source-bound unchanged native-credit decision, never infer pending results."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();out=ROOT/f'experiments/results/causal_induction/{a.tag}.json';assert not out.exists();rows=[];parents={}
 for seed in [194,195,196]:
  p=ROOT/f'experiments/results/causal_induction/curie_causal_race_credit_confirm_20261012_v1_s{seed}.json';r=json.loads(p.read_text());assert r['status']=='completed'
  for name,digest in r['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
  parents[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest();rows.append(dict(seed=seed,gains=r['quality_gains'],all_positive=r['all_quality_positive'],arms=[{k:x[k] for k in ['arm','mean_dev_nll','whole_fit_wall_s']} for x in r['rows']]))
 mean={arm:sum(x['gains'][arm] for x in rows)/3 for arm in ['exact','zero25','reset_memory']};gate=all(x['all_positive'] for x in rows if x['seed'] in [195,196]) and all(x>0 for x in mean.values());result=dict(status='completed',battle='B5/R1',tag=a.tag,rows=rows,mean_quality_gains=mean,confirmation_gate=gate,next_decision='admit bounded native learner-transfer smoke, then fit only after actual RSS/time admission' if gate else 'hold broader transfer; diagnose native teacher with completed centered contract, preserve fixed cohort',parent_sha256=parents,source_sha256={str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},scope='Private four fresh graphs with three actors, fixed confirmation rule. Per-seed/mean quality and whole fitting CPU; no public benchmark/efficiency or neural-depth claim. Original selection scores excluded from cohort decision.');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
