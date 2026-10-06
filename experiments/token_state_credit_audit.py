"""Conditional factual NLL credit to trained addressed state and event messages."""
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'));sys.path.insert(0,str(ROOT))
import horizon_token_language_engine as lab

def leaf_state(state):
    result={}
    for key,v in state.items():
        if isinstance(v,list):result[key]=[x.detach().clone().requires_grad_(key=='mem') for x in v]
        else:result[key]=v.detach().clone().requires_grad_(key=='ctx_vals')
    return result

def metrics(values):
    g=torch.cat([(torch.zeros_like(v) if v.grad is None else v.grad).flatten() for v in values]).double()
    x=torch.cat([v.detach().flatten() for v in values]).double()
    return dict(coordinates=g.numel(),gradient_l2=float(g.norm()),gradient_rms=float(g.square().mean().sqrt()),
        state_l2=float(x.norm()),gradient_state_inner_product=float(g@x),all_finite=bool(torch.isfinite(g).all()))

torch.set_num_threads(1);rows=[]
folder=ROOT/'experiments/results/token_language'
for seed in (6,7):
    for window in (16,64):
        tag=f'curie_horizon_2k_batch64_credit{window}'+('' if seed==6 else '_s7')+'_20261005_v1'
        result=json.loads((folder/(tag+'.json')).read_text());selection=json.loads((folder/(tag+'.selection.json')).read_text())['selected']
        a=SimpleNamespace(**result['args']);a.credit_window=64
        train=lab.load_tokens(a.train_file);dev=lab.load_tokens(a.dev_file)
        counts=lab.LaneTokens(train,0,a.train_tokens,a.lanes).counts();torch.manual_seed(seed)
        model=lab.Model(a,torch.argsort(counts,descending=True,stable=True),counts)
        checkpoint=Path(selection['checkpoint']);model.load_state_dict(torch.load(checkpoint,weights_only=True));model.eval()
        data=lab.interval_tensor(dev,a.dev_offset,a.dev_tokens,a.eval_lanes)
        gen=torch.Generator().manual_seed(seed+100000)
        with torch.no_grad():_,base,_=model(data[:,:32],None,gen,a,deterministic=a.deterministic_eval)
        rng=gen.get_state()
        for horizon in (1,16,64):
            state=leaf_state(base);g=torch.Generator();g.set_state(rng);model.zero_grad(set_to_none=True)
            x,_,_=model(data[:,32:32+horizon],state,g,a,deterministic=a.deterministic_eval)
            nll=model.readout.nll(x,data[:,33:33+horizon]).mean();nll.backward()
            m=metrics(state['mem']);c=metrics([state['ctx_vals']]);assert m['all_finite'] and c['all_finite']
            rows.append(dict(tag=tag,seed=seed,trained_credit_window=window,diagnostic_horizon=horizon,
                targets=8*horizon,prefix_tokens_per_lane=32,nll=float(nll.detach()),memory=m,message=c,
                memory_to_message_gradient_l2=m['gradient_l2']/max(c['gradient_l2'],1e-30),
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
            print(json.dumps(rows[-1]),flush=True)
out=ROOT/'experiments/results/diagnostics/curie_token_state_credit_audit_20261005_v1.json'
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(dict(status='completed',rows=rows,
    scope='Selected trained models, same frozen causal development prefix and continuation RNG. Diagnostic permits continuous state credit for up to64positions, overriding training detach window only for derivative measurement. Gradients of mean NLL to initial stored values/message, conditional on factual hard winners; include differentiable clocks but omit discrete alternate-route utility. No fitting, optimizer update, public validation, or claim of full route-learning credit.'),indent=2)+'\n')
