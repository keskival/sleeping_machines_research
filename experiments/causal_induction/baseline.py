"""B5/R1 structural diagnostic: public-stream persistence/calibration reference."""
import argparse,hashlib,json,resource,time
from pathlib import Path
from public_stream import persistence,records
ROOT=Path(__file__).resolve().parents[2]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();out=ROOT/f'experiments/results/causal_induction/{a.tag}.json';assert not out.exists();begin=time.monotonic();parent=ROOT/'experiments/results/causal_induction/curie_causal_induction_20261012_v1_prepare.json';r=json.loads(parent.read_text());assert r['status']=='completed';prior=(1.,2.);train=[];rows=[]
 for item in r['records']:
  path=ROOT/item['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
  if item['split']=='train':metrics,prior=persistence(path,prior,True);train.append({k:v for k,v in metrics.items() if k!='issued_predictions'})
 for item in r['records']:
  if item['split']=='train':continue
  path=ROOT/item['path']
  for adapt in [False,True]:
   metrics,_=persistence(path,prior,adapt);metrics.pop('issued_predictions');rows.append(dict(split=item['split'],variant=item['variant'],path=item['path'],arm='online_calibration' if adapt else 'frozen_calibration',**metrics))
 first=ROOT/next(x['path'] for x in r['records'] if x['split']=='train');stream=list(records(first));assert all('target' not in x for x in stream if x['kind']=='query');issued=False
 for x in stream:
  if x['kind']=='query':issued=True;break
 assert issued
 sources=[Path(__file__),ROOT/'experiments/causal_induction/public_stream.py'];out.write_text(json.dumps(dict(status='completed',battle='B5/R1',tag=a.tag,prior=prior,train=train,rows=rows,whole_wall_s=time.monotonic()-begin,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},scope='Last publicly observed bit forecast with acquired/online confidence calibration. Predictions stored at request time and scored only on later feedback; no hidden-parent/gate/oracle files read. Strong information-matched memory baseline, not sparse neural model or learned causal-structure evidence. Heldout graph adaptation copies acquired global calibration independently per episode.'),indent=2)+'\n');print(json.dumps(dict(rows=len(rows),prior=prior,wall_s=time.monotonic()-begin)),flush=True)
if __name__=='__main__':main()
