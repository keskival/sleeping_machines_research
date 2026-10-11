"""B4 fixed split2 mark-memory development gate; no TEST."""
import argparse,json,hashlib,statistics,time,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'experiments/results/tpp_b4'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();start=time.monotonic();paths=[BASE/'aws_b4_split2_logspread_develop_20261011.json',BASE/'aws_b4_split2_marks_develop_20261011.json'];records=[]
 for path in paths:
  r=json.loads(path.read_text())
  for s,h in r['source_sha256'].items():assert digest(ROOT/s)==h
  records.append(r)
 base,mark=records;assert base['split']==mark['split']==2 and base['data_sha256']==mark['data_sha256']
 complete=all(r['status']=='completed' and r['skipped_updates']==0 for r in records)
 selected=[min(r['history'],key=lambda x:x['val']['total'])['val'] if r['history'] else None for r in records]
 common=[dict(epoch=b['epoch'],mark_gain=b['val']['L_M']-m['val']['L_M'],time_gain=b['val']['L_T']-m['val']['L_T'],total_gain=b['val']['total']-m['val']['total']) for b,m in zip(base['history'],mark['history'])];window=common[-20:];mean=statistics.mean(r['mark_gain'] for r in window) if window else None
 gate=complete and len(window)==20 and mean>=1.
 out=dict(status='completed',tag=a.tag,battle='B4 prospective split2 mark-memory development',gate=gate,fits_completed_without_skips=complete,selected_repaired=selected[0],selected_marks=selected[1],selected_mark_gain=selected[0]['L_M']-selected[1]['L_M'] if all(selected) else None,selected_total_gain=selected[0]['total']-selected[1]['total'] if all(selected) else None,last20_common_mark_gain=mean,common_epoch_comparison=common,accounting=[dict(arm=name,epochs=r['epochs_run'],targets=r['completed_train_targets'],fit_s=r['fitting_wall_s'],evaluation_s=r['evaluation_wall_s'],inference=r['inference'],peak_rss_kb=r['max_rss_kb']) for name,r in zip(['repaired core','mark memory'],records)],paid_pilot_wall_s=base['additional_paid_pilot_wall_s'],source_sha256={'experiments/tpp/summarize_split2_marks.py':digest(Path(__file__))},input_sha256={str(p.relative_to(ROOT)):digest(p) for p in paths},wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,rule='Unchanged joint stability gate:20 common final epochs mean mark gain>=1.0, finite completed fits and zero skipped updates. Selected score does not replace stability gate.',scope='One seed0/fresh Github split2 partition, same cohort. Separate original-budget models/data-order recipes; no TEST or new benchmark claim. Complete predictor may supply diagnostic TRAIN soft targets only after separately registered calibration/readout decision. Full fitting FLOPs unmeasured.')
 with (BASE/f'{a.tag}.json').open('x') as f:json.dump(out,f,indent=2)
 print(json.dumps(out))
if __name__=='__main__':main()
