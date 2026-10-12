"""R1 full-prefix versus cached numerical-window teacher/update parity."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'experiments'))
import common_prior_token_language as reference
import window_replay_token_language as candidate


def equal_state(a, b):
    for key in ('mem','arr','seen','ctx_vals','ctx_arr','has_ctx','position'):
        if key not in a: continue
        aa,bb=a[key],b[key]
        if isinstance(aa,list):
            assert all(torch.equal(x,y) for x,y in zip(aa,bb)),key
        else: assert torch.equal(aa,bb),key


def check(depth, mode):
    args=SimpleNamespace(seed=6,payload=8,depth=depth,heads=2,pool=4,input_init='balanced',
        tie_pools=False,readout='adaptive',readout_init='frequency',minimum_tail_width=0,
        credit_window=8,state_mode=mode,route_credit='sampled',future_site='uniform',
        free_bias=0.,temperature=1.,compiled=False)
    order=torch.arange(50257); counts=torch.ones(50257,dtype=torch.long)
    torch.manual_seed(6); old=reference.Model(args,order,counts)
    torch.manual_seed(6); new=candidate.Model(args,order,counts)
    torch.manual_seed(7)
    with torch.no_grad():
        for name,p in old.readout.named_parameters():
            if p.ndim>1:p.add_(.01*torch.randn_like(p))
    new.load_state_dict(old.state_dict())
    optimizers=[torch.optim.AdamW(m.parameters(),lr=.003) for m in (old,new)]
    generators=[torch.Generator().manual_seed(81) for _ in range(2)]
    states=[None,None]; skipped=0
    for step, token in enumerate((2,10,21)):
        buf=torch.randint(0,31,(2,25),generator=torch.Generator().manual_seed(91+step))
        buf[0,9]=50256;buf[1,17]=50256
        layer=step%depth;head=step%2;n=24
        before=[g.get_state().clone() for g in generators]
        initial=list(states); features=[]; probabilities=[]; losses=[]
        for model,gen,state in zip((old,new),generators,states):
            model.train();model.suppress_site=(token,layer,head)
            x,state,pis=model(buf[:,:n],state,gen,args)
            features.append(x);probabilities.append(pis);losses.append(model.readout.nll(x,buf[:,1:]))
            states[len(features)-1]=state
        assert torch.equal(features[0],features[1]);equal_state(*states)
        assert torch.equal(generators[0].get_state(),generators[1].get_state())
        winner=states[0]['race_winners'][token,layer,:,head];alt=(winner+1)%4
        end=min(n,(token//8+1)*8);positions=torch.arange(token,min(end,token+4));length=int(positions[-1])+1
        begin=new.replay_window['begin'];skipped+=begin
        differences=[];shadow_generators=[]
        for i,model in enumerate((old,new)):
            offset=0 if i==0 else begin
            model.force_site=(token-offset,layer,head,alt);model.suppress_site=(token-offset,layer,head)
            model.shadow_mode=True;gen=torch.Generator()
            gen.set_state(before[i] if i==0 else model.replay_window['rng'])
            with torch.no_grad():
                shadow,shadow_state,_=model(buf[:,offset:length],initial[i] if i==0 else model.replay_window['state'],gen,args)
                delta=reference.suffix_difference(model.readout,shadow,buf[:,offset+1:],losses[i].reshape(2,n)[:,offset:],positions-offset)
            if i==0:full_shadow,full_state=shadow,shadow_state
            else:
                assert torch.equal(full_shadow[:,begin:],shadow);equal_state(full_state,shadow_state)
            differences.append(delta);shadow_generators.append(gen.get_state())
            model.shadow_mode=False;model.force_site=model.suppress_site=None
        assert torch.equal(*differences);assert torch.equal(*shadow_generators)
        for i,(model,opt) in enumerate(zip((old,new),optimizers)):
            pi=probabilities[i][token*depth+layer][1][:,head,:].double()
            weight=n*depth*2*(end-token)/n
            term=reference.paired_route_credit(pi,alt,torch.full((2,),1/3,dtype=torch.float64),differences[i]*weight)
            opt.zero_grad();(losses[i].mean()+term).backward()
        for (name,p),(_,q) in zip(old.named_parameters(),new.named_parameters()):
            assert (p.grad is None)==(q.grad is None),name
            if p.grad is not None:assert torch.equal(p.grad,q.grad),name
        for model,opt in zip((old,new),optimizers):
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        for (name,p),(_,q) in zip(old.named_parameters(),new.named_parameters()):assert torch.equal(p,q),name
        states=[reference.detach(s) for s in states]
    return dict(depth=depth,state_mode=mode,updates=3,skipped_prefix_events_per_lane=skipped,
                feature_teacher_gradient_parameter_error=0.)


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1)
    rows=[check(depth,mode) for depth in (2,4) for mode in ('carry','reset')]
    sources={}
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file:
            path=Path(file).resolve()
            if path.is_file() and path.is_relative_to(ROOT) and path.suffix=='.py':
                relative=path.relative_to(ROOT)
                if relative.parts[0] in ('experiments','sleeping_machines'):sources[str(relative)]=hashlib.sha256(path.read_bytes()).hexdigest()
    result=dict(status='completed',battle='R1',rows=rows,source_sha256=sources,
        scope='CPU float32 integrated numerical witness, depth2/4 and carry/reset, EOS before/after credit cuts and three clipped AdamW updates. Full-prefix and cached-window replay agree exactly on suffix features, physical state, suffix utility, RNG, all parameter gradients and parameters. Diagnostic replay counters differ because prefix work is skipped. No full-fit quality or efficiency claim.')
    out=ROOT/f'experiments/results/diagnostics/{a.tag}.json'
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
