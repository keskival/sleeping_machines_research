"""B10 native episode carry, gradient and real recorded-cell parity contracts."""
import argparse,hashlib,json,time
from copy import deepcopy
import numpy as np
import torch
import b10_tpp as p
from episode_state import consume,revealed_log_probability,detach_state
from recall_tpp_v4 import KeyedRaceTPP
SOURCES=['experiments/market/check_episode_state.py','experiments/market/episode_state.py','experiments/market/b10_tpp.py','experiments/tpp/race_tpp_v5.py','experiments/tpp/recall_tpp_v4.py']

def batched(model,t,m):
    h,s=model.encode(t,m,torch.ones_like(m,dtype=torch.bool));params=model.clocks(h[:,:-1],s[:,:-1]);g=t[:,1:]-t[:,:-1];_,s0=model.clock_terms(g,*params[:4]);_,s1=model.clock_terms(g+p.CELL,*params[:4]);hazard,_=model.clock_terms(g+p.CELL/2,*params[:4]);selected=params[4].gather(-1,m[:,1:,None,None].expand(-1,-1,model.M,1)).squeeze(-1)
    return p.log_interval(s0.sum(-1),s1.sum(-1))+torch.logsumexp(hazard+selected,-1)-torch.logsumexp(hazard,-1)

def stream(model,t,m,cut=None):
    state=None;forecast=None;values=[]
    for j in range(t.shape[1]):
        if forecast is not None:values.append(revealed_log_probability(model,forecast,t[:,j],m[:,j]))
        forecast,state=consume(model,t[:,j],m[:,j],state,parameter_version=0)
        if cut==j:state=detach_state(state)
    return torch.stack(values,1),state

def compare(model,t,m,gradients=True):
    model.eval();copy=deepcopy(model);x=batched(model,t,m);y,state=stream(copy,t,m);error=float((x-y).abs().max());assert error<1e-8,error
    ge=0.
    if gradients:
        g=torch.autograd.grad(x.mean(),tuple(model.parameters()),allow_unused=True);h=torch.autograd.grad(y.mean(),tuple(copy.parameters()),allow_unused=True)
        for param,a,b in zip(model.parameters(),g,h):
            a=torch.zeros_like(param) if a is None else a;b=torch.zeros_like(param) if b is None else b;ge=max(ge,float((a-b).abs().max()))
        assert ge<1e-7,ge
    detached,_=stream(copy,t,m,cut=t.shape[1]//2);assert float((detached-y).abs().max())<1e-8
    return dict(score_error=error,gradient_error=ge,persistent_state_elements=sum(z.numel() for z in state['layers'])+state['slots'].numel()+(0 if state['keys'] is None else state['keys'].numel()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(181);start=time.monotonic();rows=[]
    t=torch.rand(2,17);t[:,::3]=0;t=t.cumsum(1);m=torch.randint(0,6,(2,17))
    for depth in (2,4,8):
        for keyed in (False,True):
            cls=KeyedRaceTPP if keyed else p.RaceTPP;kwargs=dict(dk=4,local=False,prev_msg=True,qk_norm=False) if keyed else {}
            model=cls(6,8,3,depth,1,2,4,0.,1.,[-1.,0.],p.CELL,**kwargs).double();result=compare(model,t,m);result.update(depth=depth,keyed=keyed);rows.append(result)
    # Reconstruct saved market scale from identical TRAIN-only statistics.
    record=json.loads((p.ROOT/'experiments/results/market/curie_b10_race_dev_s0_20261009T1800Z.json').read_text());training=[w for day in p.TRAIN for w in p.windows(day,record['args']['train_every'])];gaps=np.concatenate([np.diff(w[0]) for w in training]);positive=gaps[gaps>0]
    real=p.Race(positive,record['args']['d'],record['args']['n_ln']);checkpoint=p.ROOT/'experiments/results/market/curie_b10_race_dev_s0_20261009T1800Z.pt';real.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));real.eval();tt,mm=p.windows(p.TRAIN[0],20)[0];rt=torch.tensor(tt[:128])[None];rm=torch.tensor(mm[:128])[None];actual=compare(real.m,rt,rm,False)
    # Forecast already issued stays fixed after a later parameter update.
    pending,state=consume(real.m,rt[:,0],rm[:,0],parameter_version=0);before=revealed_log_probability(real.m,pending,rt[:,1],rm[:,1]).detach().clone()
    with torch.no_grad():real.m.clock.bias.add_(.01)
    after=revealed_log_probability(real.m,pending,rt[:,1],rm[:,1]).detach();assert torch.equal(before,after)
    result=dict(status='completed',tag=a.tag,battle='B10/R1',synthetic=rows,market=actual,forecast_parameter_version_preserved=True,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),wall_s=time.monotonic()-start,source_sha256={s:hashlib.sha256((p.ROOT/s).read_bytes()).hexdigest() for s in SOURCES},scope='Explicit persistent temporal/mark/key state, exact first-order factual credit when graph retained; detachment preserves numeric state but truncates credit. Frozen-parameter parity and cached forecast version check, not changing-weight full-history gradient correctness. Real data uses TRAIN only, no VAL or TEST; serial CPU event execution is not asynchronous hardware.')
    output=p.ROOT/'experiments/results/market'/f'{a.tag}.json';assert not output.exists();output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
