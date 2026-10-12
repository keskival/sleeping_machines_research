"""R1 acquired packed-bank/Adam transport and activated-receiver witness."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import window_replay_token_language as engine
from sleeping_machines.expanded_token_core import ExpandedTokenCore
from sleeping_machines.optimizer_expansion import optimizer_for_expanded_model


def check(tied):
    a=SimpleNamespace(seed=6,payload=4,depth=2,heads=2,pool=2,input_init='balanced',tie_pools=tied,
        readout='adaptive',readout_init='frequency',minimum_tail_width=0,credit_window=8,state_mode='carry',
        route_credit='sampled',future_site='uniform',free_bias=0.,temperature=1.,compiled=False)
    torch.manual_seed(6)
    old=engine.Model(a,torch.arange(50257),torch.ones(50257,dtype=torch.long))
    opt=torch.optim.AdamW(old.parameters(),lr=.003)
    generator=torch.Generator().manual_seed(71);state=None
    buf=torch.randint(1,32,(2,9),generator=torch.Generator().manual_seed(88));buf[0,4]=50256
    for _ in range(4):
        opt.zero_grad(set_to_none=True)
        features,state,_=old(buf[:,:8],state,generator,a)
        old.readout.nll(features,buf[:,1:]).mean().backward()
        torch.nn.utils.clip_grad_norm_(old.parameters(),1.);opt.step();state=engine.detach(state)
    new=copy.deepcopy(old);new.core=ExpandedTokenCore(old.core,2)
    expanded_opt=optimizer_for_expanded_model(old,opt,new)
    retained=dict(old.named_parameters());target=dict(new.named_parameters())
    assert all(torch.equal(p,target[n]) and p.data_ptr()!=target[n].data_ptr() for n,p in retained.items())
    added={n:p for n,p in target.items() if n not in retained}
    assert added and all(p not in expanded_opt.state for p in added.values())
    for n,p in retained.items():
        if p not in opt.state:continue
        for k,v in opt.state[p].items():
            other=expanded_opt.state[target[n]][k]
            if isinstance(v,torch.Tensor):assert torch.equal(v,other) and v.data_ptr()!=other.data_ptr()
            else:assert v==other
    grown=new.core.expand_numeric_state(state)
    for name in ('mem','arr','seen'):
        for x,y in zip(state[name],grown[name]):
            assert torch.equal(x,y[:,:,:2]) and not bool(y[:,:,2:].any())
            assert x.data_ptr()!=y.data_ptr()
    for left,right in zip(old.core._stacked(0),new.core._stacked(0)):
        for name,x in left.items():
            if not isinstance(x,torch.Tensor):assert x==right[name];continue
            if name=='query':assert torch.equal(x,right[name]);continue
            y=right[name].reshape(2,4,*x.shape[1:])[:,:2].reshape_as(x)
            assert torch.equal(x,y),name
    # Isolate retained leaves to certify old Adam continuation. Activating new
    # competing routes is a different intervention and need not preserve loss.
    for _ in range(3):
        for model,optimizer in ((old,opt),(new,expanded_opt)):
            optimizer.zero_grad(set_to_none=True)
            params=dict(model.named_parameters())
            loss=sum(params[n].square().mean() for n in retained)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        assert all(torch.equal(p,dict(new.named_parameters())[n]) for n,p in old.named_parameters())
    assert all(p not in expanded_opt.state for p in added.values())
    a.pool=4;new.force_site=(0,0,0,torch.full((2,),2,dtype=torch.long))
    expanded_opt.zero_grad(set_to_none=True)
    features,grown,pis=new(buf[:,:8],grown,torch.Generator().manual_seed(71),a)
    loss=new.readout.nll(features,buf[:,1:]).mean()+.0001*sum(pi.square().mean() for _,pi in pis)
    loss.backward();assert bool(torch.isfinite(loss));torch.nn.utils.clip_grad_norm_(new.parameters(),1.)
    activated=[p for p in added.values() if p.grad is not None]
    assert activated and bool((grown['last_writes'][:,:,2:]>0).any())
    assert any(bool(p.grad.abs().max()>0) for p in activated)
    expanded_opt.step()
    assert all(int(expanded_opt.state[p]['step'])==1 for p in activated)
    assert int(expanded_opt.state[new.core.banks['input']]['step'])==8
    return dict(tied_processing=tied,source_updates=4,retained_continuation_updates=3,
        old_bank_adam_step=8,new_bank_adam_step=1,retained_parameter_error=0.,
        added_parameters=sum(p.numel() for p in added.values()),actual_new_receiver_loss=float(loss.detach()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1)
    rows=[check(tied) for tied in (False,True)]
    sources={}
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file:
            path=Path(file).resolve()
            if path.is_file() and path.is_relative_to(ROOT) and path.suffix=='.py':
                relative=path.relative_to(ROOT)
                if relative.parts[0] in ('experiments','sleeping_machines'):sources[str(relative)]=hashlib.sha256(path.read_bytes()).hexdigest()
    result=dict(status='completed',battle='R1',rows=rows,source_sha256=sources,
        scope='Component numerical witness on integrated proper-token model, tied/untied processing. Preserves old leaves/moments and physical state; old-only continuation exact; new receivers execute, receive gradients and start Adam at step1 while old banks reach8. Expansion changes competition/noise layout and is not old inference parity, learned recruitment, capacity utility or transfer-quality evidence.')
    out=ROOT/f'experiments/results/diagnostics/{a.tag}.json'
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
