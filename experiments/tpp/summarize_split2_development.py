"""B4 completed prospective split2 core fit decision, no TEST access."""
import argparse,json,hashlib,math,time,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'experiments/results/tpp_b4'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();start=time.monotonic();paths=[BASE/'aws_b4_split2_logspread_develop_20261011.json',BASE/'b4g_github_s2.json'];records=[]
 for path in paths:
  r=json.loads(path.read_text())
  for s,h in r['source_sha256'].items():assert digest(ROOT/s)==h
  records.append(r)
 repair,old=records;assert repair['split']==old['split']==2 and 'test' not in repair
 rows=[dict(epoch=x['epoch'],repair=x['val'],guarded_total=y['val_total'],total_gain=y['val_total']-x['val']['total']) for x,y in zip(repair['history'],old['history'])]
 complete=repair['status']=='completed' and repair['skipped_updates']==0
 selected=min(repair['history'],key=lambda x:x['val']['total'])['val'] if repair['history'] else None
 gain=old['val']['total']-selected['total'] if selected else None
 gate=complete and gain>=.5 and all(math.isfinite(x['val'][k]) for x in repair['history'] for k in ['L_T','L_M','total'])
 out=dict(status='completed',tag=a.tag,battle='B4 prospective split2 development',repaired_fit_completed=complete,quality_gate=gate,selected_repaired_dev=selected,guarded_dev=old['val'],selected_total_gain=gain,epochs=repair['epochs_run'],skipped_updates=repair['skipped_updates'],common_epoch_comparison=rows,fitting_wall_s=repair['fitting_wall_s'],evaluation_wall_s=repair['evaluation_wall_s'],paid_pilot_wall_s=repair['additional_paid_pilot_wall_s'],completed_train_targets=repair['completed_train_targets'],inference=repair['inference'],fit_max_rss_kb=repair['max_rss_kb'],source_sha256={'experiments/tpp/summarize_split2_development.py':digest(Path(__file__))},input_sha256={str(p.relative_to(ROOT)):digest(p) for p in paths},wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,rule='Before result: zero skipped updates/finite completed fit and selected DEV total gain>=0.5 against saved guarded split2 before further mark fitting.',scope='One fresh seed0/new Github partition, same cohort; no TEST or benchmark replacement.100pass/original30patience/lr/batch preserved. Fresh restart repeats paid5pass pilot; pilot/fitting/evaluation/inference costs separate. Full fitting FLOPs unmeasured; no sparse/depth learning efficiency claim.')
 with (BASE/f'{a.tag}.json').open('x') as f:json.dump(out,f,indent=2)
 print(json.dumps(out))
if __name__=='__main__':main()
