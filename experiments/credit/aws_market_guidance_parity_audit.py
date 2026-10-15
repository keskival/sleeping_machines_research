"""R1/B10 exact numerical diagnosis of scalar versus batched acquired updates."""
import argparse,copy,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/market'));sys.path.insert(0,str(ROOT/'experiments/credit'))
import aws_market_guided_online as bridge

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);tag=p.parse_args().tag;torch.set_num_threads(1);torch.set_default_dtype(torch.float64);began=time.monotonic();data,_=bridge.prepare();gaps=np.concatenate([np.diff(t) for t,m in data['train']]);gaps=gaps[gaps>0];model=bridge.MatchedRace(gaps,'native');base=ROOT/'experiments/results/market/aws_b10_stable_native_s0_20261014_v1.json';r=json.loads(base.read_text());weights=ROOT/r['final_weights'];model.load_state_dict(torch.load(weights,weights_only=False)['model']);model.eval();scalar=copy.deepcopy(model.m);vector=copy.deepcopy(model);so=torch.optim.AdamW(scalar.parameters(),lr=.0001,weight_decay=1e-4,foreach=False);vo=torch.optim.AdamW(vector.parameters(),lr=.0001,weight_decay=1e-4,foreach=False)
 z=np.load(ROOT/'data/binance/npz/2026-08-24.npz');origin=int(z['t'][0]);t,m,path=bridge.load('2026-08-24',65,origin);ss=bridge.initial(scalar,1);ss['last_time']=t[:1].clone();vs=None;rows=[]
 for k in (0,32):
  ss,sg,scores=bridge.exact.window(scalar,ss,t,m,k,32,'bptt');vo.zero_grad();parameters,vs=bridge.forecast(vector,t[k:k+32][None],m[k:k+32][None],vs);terms=bridge.revealed_terms(vector,parameters,t[k:k+32][None],t[k+1:k+33][None],m[k+1:k+33][None]);(-terms.mean()).backward();by=dict(zip(dict(scalar.named_parameters()),sg));grad=[]
  for name,p in vector.m.named_parameters():
   g=torch.zeros_like(p) if p.grad is None else p.grad;grad.append(dict(name=name,max_error=float((g-by[name]).abs().max()),scalar_norm=float(by[name].norm()),vector_norm=float(g.norm())))
  bridge.guide.step(scalar,so,sg);torch.nn.utils.clip_grad_norm_(vector.parameters(),1.,foreach=False);vo.step();vs=bridge.detach(vs);weights_by=dict(scalar.named_parameters());drift=sorted([(name,float((p-weights_by[name]).detach().abs().max())) for name,p in vector.m.named_parameters()],key=lambda x:-x[1]);rows.append(dict(step=k//32+1,score_error=max(abs(a+float(b)) for a,b in zip(scores,terms.flatten())),gradients=sorted(grad,key=lambda x:-x['max_error'])[:10],parameter_drift=drift[:10]));print(json.dumps(rows[-1]),flush=True)
 names=['experiments/credit/aws_market_guidance_parity_audit.py','experiments/credit/aws_market_guided_online.py','experiments/market/b10_online_stream.py','experiments/market/episode_state.py','experiments/credit/market_streaming_clock_updates.py'];record=dict(status='completed',battle='B10',rows=rows,source_sha256={n:bridge.digest(ROOT/n) for n in names},wall_s=time.monotonic()-began,scope='Two same-issuedTRAIN updates; diagnose acquiredactor parity, no qualityselection or model promotion.')
 with (ROOT/f'experiments/results/market/{tag}.json').open('x') as f:json.dump(record,f,indent=2)
if __name__=='__main__':main()
