"""Frozen August model choice, first October chronological online confirmation."""
import argparse,copy,json,resource,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments'));sys.path.insert(0,str(Path(__file__).parent))
import b10_online_comparison as standard
import b10_cached_comparison as cached
from capability_admission import validate

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--arm',required=True,choices=['native','lstm','transformer_cached','transformer_reencoded']);a=p.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);began=time.monotonic();protocol=ROOT/'experiments/market/B10_OCTOBER_CONFIRMATION.json';r=json.loads(protocol.read_text());validate(r);spec=r['arms'][a.arm];source=ROOT/spec['acquired_record'];prior=json.loads(source.read_text());assert prior['status']=='completed' and prior['targets']==65536;weights=ROOT/prior['final_weights'];saved=torch.load(weights,weights_only=False);assert saved['targets']==65536;online=cached if a.arm=='transformer_cached' else standard;family='transformer' if a.arm.startswith('transformer') else a.arm;data,_=online.prepare();gaps=np.concatenate([np.diff(t) for t,m in data['train']]);gaps=gaps[gaps>0];model=online.MatchedRace(gaps,family);model.load_state_dict(saved['model']);model.eval();opt=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=1e-4,foreach=False);opt.load_state_dict(copy.deepcopy(saved['optimizer']));state=online.detach(saved['state']);receipt=ROOT/r['data_receipt'];data_record=json.loads(receipt.read_text());assert data_record['status']=='completed' and data_record['day']=='2026-10-01';path=ROOT/'data/binance/npz/2026-10-01.npz';assert online.digest(path)==data_record['npz_sha256'];z=np.load(ROOT/'data/binance/npz/2026-08-24.npz');origin=int(z['t'][0]);z=np.load(path);t=torch.tensor((z['t'][:65664]-origin)*1e-6)[None];m=torch.tensor(z['mark'][:65664].astype(np.int64))[None];assert (t[:,0]>state['last_time'].flatten()).all();ledger=online.Ledger();warm=online.Ledger()
 with torch.no_grad(),warm:_,state=online.forecast(model,t[:,:127],m[:,:127],state,2048);state=online.detach(state)
 total=0.;count=0;curve=[];prefix=ROOT/f'experiments/results/market/{a.tag}';segment=time.monotonic()
 with ledger:
  for start in range(127,127+65536,32):
   loss,state=online.apply(model,opt,t[:,start:start+32],m[:,start:start+32],t[:,start+1:start+33],m[:,start+1:start+33],state,2048);total-=float(loss.double().sum());count+=loss.numel()
   if count%4096==0:
    curve.append(dict(targets=count,nll=total/count));progress=dict(model=model.state_dict(),optimizer=opt.state_dict(),state=state,targets=count,loss_sum=total,curve=curve,preregistration_sha256=online.digest(protocol));tmp=prefix.with_suffix('.tmp');torch.save(progress,tmp);tmp.replace(prefix.with_suffix('.progress.pt'));print(json.dumps(curve[-1]),flush=True)
 torch.save(progress,prefix.with_suffix('.pt'));names=['experiments/market/b10_shift_online.py','experiments/market/b10_online_comparison.py','experiments/market/b10_cached_comparison.py','experiments/market/b10_online_stream.py','experiments/market/b10_cached_transformer.py'];record=dict(status='completed',battle='B10',arm=a.arm,targets=count,nll=total/count,curve=curve,ledger=ledger.record(),context_only_work=warm.record(),online_wall_s=time.monotonic()-segment,source_sha256={n:online.digest(ROOT/n) for n in names},input_sha256={str(p.relative_to(ROOT)):online.digest(p) for p in [protocol,source,weights,receipt,path]},final_weights=str(prefix.with_suffix('.pt').relative_to(ROOT)),max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,process_wall_s=time.monotonic()-began,scope='Predeclared firstOctober confirmation; all initialacquisition/modelselection and prior online adaptation use August observations only. No October tuning. Every arm adapts after32revealed outcomes, retains its complete August learner/memory and observes128new context events before65536forecasttargets. Fixed6-mark/1us likelihood. Source/archive checksumverified. No SeptemberTEST opened, tradingreturns, replicatedseeds or universalfrontier claim.')
 with prefix.with_suffix('.json').open('x') as f:json.dump(record,f,indent=2)
if __name__=='__main__':main()
