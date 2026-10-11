"""B4 prospective partition pilot: numerical/quality decision, no TEST."""
import argparse,json,hashlib,math,time,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'experiments/results/tpp_b4'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();start=time.monotonic();paths=[BASE/'aws_b4_split2_logspread_pilot_20261011.json',BASE/'b4g_github_s2.json'];records=[]
 for path in paths:
  r=json.loads(path.read_text());assert r['status']=='completed'
  for s,h in r['source_sha256'].items():assert digest(ROOT/s)==h
  records.append(r)
 new,old=records;assert new['split']==old['split']==2 and 'test' not in new
 common=[dict(epoch=x['epoch'],repair_total=x['val_total'],guarded_total=y['val_total'],gain=y['val_total']-x['val_total']) for x,y in zip(new['history'],old['history'])]
 finite=all(math.isfinite(x[k]) for x in new['history'] for k in ['val_L_T','val_L_M','val_total']);gate=new['epochs_run']==5 and new['skipped_updates']==0 and finite
 out=dict(status='completed',tag=a.tag,battle='B4 prospective split2 development',numerical_pilot_gate=gate,epochs=new['epochs_run'],skipped_updates=new['skipped_updates'],validation=new['val'],paired_epoch_comparison=common,pilot_wall_s=new['wall_s'],mean_common_epoch_gain=sum(x['gain'] for x in common)/len(common),source_sha256={'experiments/tpp/summarize_split2_logspread_pilot.py':digest(Path(__file__))},input_sha256={str(p.relative_to(ROOT)):digest(p) for p in paths},wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Fresh Github split2 seed0, five TRAIN/DEV passes after full-gradient gate; unchanged numeric repair and original data-derived clock settings. Same cohort/new partition, not independent dataset or extra unique data. Read existing saved score metadata only, no new TEST. Pilot wall includes validation, RSS watchdog cap2.5GB; pilot peak RSS not supplied by adapter. Longer fitting not automatically admitted; numerical gate and early quality inform its next plan.')
 with (BASE/f'{a.tag}.json').open('x') as f:json.dump(out,f,indent=2)
 print(json.dumps(out))
if __name__=='__main__':main()
