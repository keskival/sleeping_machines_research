"""Finite-range probability/gradient parity plus exact failed-LSTM replay."""
import argparse,copy,json,resource,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/market'));sys.path.insert(0,str(ROOT/'experiments/tpp'))
from b10_matched_models import MatchedRace
from b10_matched_data import prepare
from b10_matched_update import update
from b10_matched_pipeline import sha
from b10_stable_clock_head import install_log_scale_head

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic();data,_=prepare();gaps=np.concatenate([np.diff(t) for t,m in data['train']]);gaps=gaps[gaps>0]
 torch.manual_seed(4);original=MatchedRace(gaps,'lstm');stable=copy.deepcopy(original);install_log_scale_head(stable)
 rows=data['train'][:2];t=torch.tensor(np.stack([r[0][:32] for r in rows]));m=torch.tensor(np.stack([r[1][:32] for r in rows]));original.eval();stable.eval()
 x=original.terms(t,m);y=stable.terms(t,m);error=float((x-y).abs().max());assert torch.allclose(x,y,rtol=1e-12,atol=1e-12)
 g=torch.autograd.grad(x.mean(),tuple(original.parameters()),allow_unused=True);h=torch.autograd.grad(y.mean(),tuple(stable.parameters()),allow_unused=True)
 max_grad=0.
 for u,v in zip(g,h):
  if u is None:assert v is None
  else:assert torch.allclose(u,v,rtol=1e-9,atol=1e-11);max_grad=max(max_grad,float((u-v).abs().max()))
 # Replay exact saved LSTM state and same TRAIN batch order through failure.
 source=ROOT/'experiments/results/market/aws_b10_matched_lstm_s0_20261014_v2.progress.pt';saved=torch.load(source,weights_only=False);model=MatchedRace(gaps,'lstm');install_log_scale_head(model);model.load_state_dict(saved['model']);opt=torch.optim.AdamW(model.parameters(),lr=saved['lr'],weight_decay=1e-4,foreach=False);opt.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng']);rng=np.random.default_rng();rng.bit_generator.state=saved['numpy_rng'];order=saved['state']['order'];cursor=saved['state']['cursor'];losses=[]
 for attempt in range(8):
  if cursor>=len(order):order=rng.permutation(len(data['train'])).tolist();cursor=0
  idx=order[cursor:cursor+16];rows=[data['train'][i] for i in idx];tt=torch.tensor(np.stack([r[0] for r in rows]));mm=torch.tensor(np.stack([r[1] for r in rows]));losses.append(update(model,opt,tt,mm));cursor+=len(idx)
 names=['experiments/market/b10_stable_head_contract.py','experiments/market/b10_stable_clock_head.py','experiments/market/b10_matched_models.py','experiments/tpp/race_tpp_v5.py','experiments/market/b10_matched_update.py']
 record=dict(status='completed',battle='B10',finite_range_score_max_error=error,finite_range_gradient_max_error=max_grad,exact_failed_checkpoint_replay_finite_updates=8,losses=losses,source_sha256={n:sha(ROOT/n) for n in names},input_sha256={str(source.relative_to(ROOT)):sha(source)},wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Algebraic log-sigma reformulation, float64 finite-range score/gradient parity and exact failingLSTM data/moments replay; no quality fit or new numerical budget admission. All families must use common stablehead in repaired comparison.')
 out=ROOT/f'experiments/results/market/{a.tag}.json'
 with out.open('x') as f:json.dump(record,f,indent=2)
 print(json.dumps(record),flush=True)
if __name__=='__main__':main()
