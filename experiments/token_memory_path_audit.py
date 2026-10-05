"""Actual selected-model memory retention and key/value path instrumentation."""
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'));sys.path.insert(0,str(ROOT))
import horizon_token_language_engine as lab
import sleeping_machines.sparse_counterfactual_episodes as episodes
from sleeping_machines.sparse_training import _unit,_rotate
original=episodes.sparse_counterfactual_layer
signature=inspect.signature(original)
ledger={}

def instrument(*args,**kwargs):
    b=signature.bind(*args,**kwargs);b.apply_defaults();a=b.arguments
    out=original(*args,**kwargs)
    m=a['m'];n,H,U,P=m.shape;winner=out[-1]
    idx=torch.arange(H)[None,:]*U+winner
    selected=torch.gather(m,2,winner[:,:,None,None].expand(n,H,1,P)).squeeze(2)
    seen=torch.gather(a['seen_d'],2,winner[:,:,None]).squeeze(2)
    stamp=torch.gather(a['arr_d'],2,winner[:,:,None]).squeeze(2)
    age=torch.where(seen,(a['arrival'][:,None]-stamp).clamp_min(0),0.)
    mixed=F.linear(a['x'],a['mix_w'],a['mix_b']);incoming=mixed.view(n,H,P)
    controls=torch.einsum('nhcp,nhp->nhc',a['control_w'].view(H*U,2,P)[idx],F.layer_norm(incoming,(P,)))+a['control_b'].view(H*U,2)[idx]
    forget=F.softplus(controls[...,0])/math.log(2)
    decay=torch.exp(-age.to(m.dtype)[...,None]*a['rate'].view(H*U,P//2)[idx]*forget[...,None])
    retained=_rotate(selected*decay.repeat_interleave(2,-1),age[...,None]*a['frequency'].view(H*U,P//2)[idx])
    written=2*torch.sigmoid(controls[...,1,None])*torch.einsum('nhpq,nhq->nhp',a['input_w'].view(H*U,P,P)[idx],incoming)
    unit_names=('control_w','control_b','rate','frequency','input_w','output_w','gate_w','gate_b','gain')
    value0,_,_=_unit(winner,torch.zeros_like(m),a['arr_d'],a['seen_d'],a['arrival'],incoming,U,P,*[a[k] for k in unit_names])
    q=torch.einsum('hpd,ld->lhp',a['query'],F.layer_norm(mixed,(H*P,)))
    memorykeys=torch.einsum('hupq,lhuq->lhup',a['key_read'].view(H,U,P,P),m)
    keydelta=(q[:,:,None,:]*memorykeys).sum(-1)/math.sqrt(P)
    for name,value in dict(revisit_age=age[seen],retention=decay[seen],
        retained_norm=retained.norm(dim=-1)[seen],written_norm=written.norm(dim=-1)[seen],
        factual_value_delta=(out[6]-value0).norm(dim=-1)[seen],
        score_memory_delta=keydelta[a['seen_d']]).items():
        ledger.setdefault(name,[]).append(value.detach().flatten())
    ledger.setdefault('selected_seen',[]).append(seen.detach().flatten().float())
    return out

def summary(values):
    x=torch.cat(values).double()
    if not x.numel():return dict(count=0)
    return dict(count=x.numel(),mean=float(x.mean()),rms=float(x.square().mean().sqrt()),
                quantiles=dict(zip(('p0','p10','p50','p90','p100'),torch.quantile(x,torch.tensor([0.,.1,.5,.9,1.],dtype=x.dtype)).tolist())))

torch.set_num_threads(1);rows=[]
folder=ROOT/'experiments/results/token_language'
for seed in (6,7):
    for window in (16,64):
        tag=f'curie_horizon_2k_batch64_credit{window}'+('' if seed==6 else '_s7')+'_20261005_v1'
        result=json.loads((folder/(tag+'.json')).read_text());selection=json.loads((folder/(tag+'.selection.json')).read_text())['selected']
        a=SimpleNamespace(**result['args']);assert not a.compiled
        train=lab.load_tokens(a.train_file);dev=lab.load_tokens(a.dev_file)
        counts=lab.LaneTokens(train,0,a.train_tokens,a.lanes).counts();torch.manual_seed(seed)
        model=lab.Model(a,torch.argsort(counts,descending=True,stable=True),counts)
        checkpoint=Path(selection['checkpoint']);model.load_state_dict(torch.load(checkpoint,weights_only=True))
        data=lab.interval_tensor(dev,a.dev_offset,a.dev_tokens,a.eval_lanes)
        ledger.clear();episodes.sparse_counterfactual_layer=instrument
        try:nll=lab.evaluate(model,data,a)
        finally:episodes.sparse_counterfactual_layer=original
        assert math.isclose(nll,selection['dev_nll'],abs_tol=2e-6)
        row=dict(tag=tag,seed=seed,credit_window=window,dev_nll=nll,forward_parity=True,
            statistics={k:summary(v) for k,v in ledger.items()},
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        rows.append(row);print(json.dumps(row),flush=True)
out=ROOT/'experiments/results/diagnostics/curie_token_memory_path_audit_20261005_v1.json'
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(dict(status='completed',rows=rows,
    scope='Frozen development factual trajectories, no training. Ages/retention/value differences condition on selected previously seen slots. Same-winner zero-memory proposal isolates immediate value sensitivity, not route or suffix utility. Score differences are pre-clamp contributions of all previously seen stored keys. Instrumentation reproduces selected loss.'),indent=2)+'\n')
