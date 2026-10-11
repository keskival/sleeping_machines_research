"""B1 saved-curve audit; no training, dataset access, or TEST scoring."""
import argparse, hashlib, json, statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
 rows=[];bindings={}
 for dataset,stem,arms,budget in [('taxi','online3_taxi', ['bptt','online_deep','online_local'],20),('amazon','online5_amazon',['bptt','trace'],100)]:
  for seed in range(3):
   curves={}
   for arm in arms:
    tag=f'{stem}_{arm}_s{seed}_r2' if dataset=='taxi' else f'{stem}_{arm}_decay_e100_s{seed}'
    p=ROOT/'experiments/results/credit'/f'{tag}.json'
    if not p.exists():continue
    x=json.loads(p.read_text())
    if x.get('status')!='completed':continue
    bindings[str(p.relative_to(ROOT))]=digest(p)
    elapsed=0;curve=[]
    for h in x['history']:
     elapsed+=h['epoch_s'];curve.append(dict(passes=h['epoch']+1,dev_ll=h['dev_ll'],epoch_wall_cumulative_s=elapsed,updates=h['updates']))
    curves[arm]=curve
   if 'bptt' not in curves:continue
   ref=curves['bptt'];target=ref[-1]['dev_ll']
   summary={}
   for arm,c in curves.items():
    hit=next((v for v in c if v['dev_ll']>=target),None)
    at_ref_wall=[v for v in c if v['epoch_wall_cumulative_s']<=ref[-1]['epoch_wall_cumulative_s']]
    summary[arm]=dict(completed_passes=c[-1]['passes'],final_dev_ll=c[-1]['dev_ll'],first_reference_final_quality=hit,quality_at_reference_epoch_wall=at_ref_wall[-1] if at_ref_wall else None,final_epoch_wall_s=c[-1]['epoch_wall_cumulative_s'],final_updates=c[-1]['updates'])
   candidate='online_deep' if dataset=='taxi' else 'trace'
   paired=[]
   if candidate in curves:
    paired=[dict(passes=u['passes'],gain=v['dev_ll']-u['dev_ll']) for u,v in zip(ref,curves[candidate])]
   rows.append(dict(dataset=dataset,seed=seed,reference_target_posthoc=target,arms=summary,paired_pass_gains=paired))
 out=dict(status='completed',battle='B1 learning-horizon audit',tag=a.tag,source_sha256={str(Path(__file__).resolve().relative_to(ROOT)):digest(Path(__file__))},input_sha256=bindings,rows=rows,scope='Post-hoc descriptive DEV audit of completed native controls. Epoch wall includes only recorded epoch time, not setup/pretraining. Passes repeat fixed TRAIN, not unique-data scaling. Update counts differ. No learned-critic or classical-reference scaling claim; pending files excluded.')
 p=ROOT/'experiments/results/credit'/f'{a.tag}.json'
 if p.exists():raise RuntimeError('result already exists')
 p.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps([dict(dataset=r['dataset'],seed=r['seed'],arms=r['arms']) for r in rows],indent=2))
if __name__=='__main__':main()
