"""Read-only R1 checkpoint health monitor; no forecasts,backward or updates."""
import argparse,gc,hashlib,io,json,math,time,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]


def atomic(path,value):
 tmp=path.with_suffix('.writing');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();torch.set_num_threads(1)
 path=ROOT/f'experiments/results/token_language/{a.run}.resume.pt';result=path.with_suffix('').with_suffix('.json')
 output=ROOT/f'.git/{a.run}.learning_health.json';previous={};seen=None;trajectory=[]
 while True:
  if result.exists():
   atomic(output,dict(status='completed',run=a.run,trajectory=trajectory));return
  if path.exists() and (stamp:=(path.stat().st_mtime_ns,path.stat().st_size))!=seen:
   raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest();saved=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False);del raw
   weights=saved['model'];cursor=saved['cursor'];bad=[n for n,v in weights.items() if (v.is_floating_point() or v.is_complex()) and not bool(torch.isfinite(v).all())]
   for d,layer in enumerate(saved['state']['layers']):
    bad += [f'layer{d}.{n}' for n,v in layer.items() if (v.is_floating_point() or v.is_complex()) and not bool(torch.isfinite(v).all())]
   history=saved['history'][-128:];depth=torch.tensor([r['depth'] for r in history],dtype=torch.float64).mean(0)
   groups={}
   for prefix in ['core.mix','core.query','core.banks','core.memory_queries','core.memory_keys','core.assemblers','causes.embed','causes.unit_out']:
    selected={n:v for n,v in weights.items() if n.startswith(prefix)}
    norm=sum(float(v.double().square().sum()) for v in selected.values())**.5
    drift=None if not previous else sum(float((v-previous[n]).double().square().sum()) for n,v in selected.items())**.5
    groups[prefix]=dict(weight_norm=norm,change_since_previous_checkpoint=drift)
   row=dict(cursor=cursor,checkpoint_sha256=digest,last_history_targets=32*len(history),mean_train_nll=sum(r['mean_nll'] for r in history)/len(history),
    depth_responsibility=depth.tolist(),effective_depths=float((-depth*depth.clamp_min(1e-30).log()).sum().exp()),groups=groups,
    bound_publications=len(saved['state']['bindings'].entries),producer_records=len(saved['state'].get('producer_records',{})),producer_replay_reads=saved['state'].get('producer_replay_reads',0),
    nonfinite_fields=bad,optimizer_step_min=min(float(s['step']) for s in saved['optimizer']['state'].values()),optimizer_step_max=max(float(s['step']) for s in saved['optimizer']['state'].values()))
   trajectory.append(row);previous={n:v.detach().clone() for n,v in weights.items() if any(n.startswith(p) for p in groups)}
   atomic(output,dict(status='needs_attention' if bad else 'monitoring',run=a.run,latest=row,trajectory=trajectory,
    scope='Read-only checkpoint summaries,no fitting or DEV/TEST. TRAINloss uses different chronological content;depth responsibility is not semantic abstraction. Parameter drift measures actual saved updates;it does not prove useful learning.'))
   print(json.dumps(dict(cursor=cursor,nonfinite_fields=bad,producer_replay_reads=row['producer_replay_reads'])),flush=True)
   del saved,weights;gc.collect();seen=stamp
  time.sleep(30)
if __name__=='__main__':main()
