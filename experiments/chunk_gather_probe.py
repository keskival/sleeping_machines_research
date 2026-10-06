"""Integrated chunk-gather parity/update and CPU timing; guarded queue only."""
import argparse
from contextlib import nullcontext
import copy
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import horizon_token_language_engine as engine
from contextlib import contextmanager
from sleeping_machines.chunk_gather_counterfactual_episodes import token_features as gathered_features

@contextmanager
def gather_kernel():
    original=engine.token_features
    engine.token_features=gathered_features
    try:yield
    finally:engine.token_features=original
from sleeping_machines.paired_route_credit import paired_route_credit
from torch.utils._python_dispatch import TorchDispatchMode

class LookupAudit(TorchDispatchMode):
    def __init__(self):
        super().__init__();self.calls=0;self.output_elements=0
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        out=func(*args,**(kwargs or {}))
        if func==torch.ops.aten.embedding_dense_backward.default:
            self.calls+=1;self.output_elements+=out.numel()
        return out


def graph_nodes(loss):
    todo=[loss.grad_fn];seen=set()
    while todo:
        node=todo.pop()
        if node is None or node in seen:continue
        seen.add(node);todo.extend(x for x,_ in node.next_functions)
    return len(seen)


def objective(model,buf,args):
    gen=torch.Generator().manual_seed(999);before=gen.get_state()
    x,state,pis=model(buf[:,:-1],None,gen,args)
    factual=model.readout.nll(x,buf[:,1:]).mean()
    event,depth,head=2,0,0
    pi=pis[event*args.depth+depth][1][:,head].double()
    winner=state['race_winners'][event,depth,:,head]
    alternative=(winner+1)%args.pool
    model.force_site=(event,depth,head,alternative);model.shadow_mode=True
    try:
        with torch.no_grad():
            shadow,_,_=model(buf[:,:-1],None,torch.Generator().set_state(before),args)
            shadow_loss=model.readout.nll(shadow,buf[:,1:]).reshape(buf.shape[0],-1).mean(1)
    finally:model.force_site=None;model.shadow_mode=False
    difference=shadow_loss-model.readout.nll(x,buf[:,1:]).reshape(buf.shape[0],-1).mean(1).detach()
    # Fixed forced alternative contracts the continuation, not an unbiased
    # full-route estimator. No fit or quality conclusion is inferred.
    q=torch.ones(buf.shape[0],dtype=torch.float64)
    loss=factual+paired_route_credit(pi,alternative,q,difference.double())
    return loss,x,state,difference


def main():
    p=argparse.ArgumentParser();p.add_argument('--control',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1);start=time.monotonic()
    path=Path(a.control);result=json.loads(path.read_text());assert result['status']=='completed'
    for name,digest in result['identity']['source_sha256'].items():assert engine.sha(ROOT/name)==digest,name
    args=SimpleNamespace(**result['args']);train=engine.load_tokens(args.train_file)
    counts=engine.LaneTokens(train,0,args.train_tokens,args.lanes).counts()
    selection=json.loads(path.with_suffix('.selection.json').read_text())['selected'];ck=Path(selection['checkpoint'])
    model=engine.Model(args,torch.argsort(counts,descending=True,stable=True),counts)
    model.load_state_dict(torch.load(ck,weights_only=True));other=copy.deepcopy(model)
    buf=engine.interval_tensor(train,0,72,8)
    if buf.shape!=(8,9):raise ValueError(buf.shape)
    buf=buf.clone();buf[0,3]=50256;buf[1,1:4]=buf[1,0]
    opt=[torch.optim.AdamW(m.parameters(),lr=args.lr) for m in (model,other)]
    snapshots=[];counts_nodes=[];lookup_counts=[]
    for m,o,patch in [(model,opt[0],nullcontext()),(other,opt[1],gather_kernel())]:
        o.zero_grad(set_to_none=True)
        audit=LookupAudit()
        with patch,audit:
            loss,x,state,difference=objective(m,buf,args);counts_nodes.append(graph_nodes(loss));loss.backward()
        lookup_counts.append(dict(calls=audit.calls,output_elements=audit.output_elements))
        snapshots.append(dict(loss=loss.detach(),x=x.detach(),mem=[t.detach().clone() for t in state['mem']],arr=[t.detach().clone() for t in state['arr']],difference=difference.detach(),grad={n:None if v.grad is None else v.grad.clone() for n,v in m.named_parameters()}))
    first,second=snapshots;torch.testing.assert_close(first['loss'],second['loss'],rtol=0,atol=0)
    for key in ('x','difference'):torch.testing.assert_close(first[key],second[key],rtol=0,atol=0)
    for key in ('mem','arr'):
        for x,y in zip(first[key],second[key]):torch.testing.assert_close(x,y,rtol=0,atol=0)
    gradient_error=0.
    for name,x in first['grad'].items():
        y=second['grad'][name]
        if x is None or y is None:assert x is None and y is None,name;continue
        torch.testing.assert_close(x,y,rtol=2e-5,atol=2e-6,msg=name)
        gradient_error=max(gradient_error,float((x-y).abs().max()))
    for optimizer in opt:optimizer.step()
    update_error=0.
    for (name,x),(_,y) in zip(model.named_parameters(),other.named_parameters()):
        torch.testing.assert_close(x,y,rtol=2e-5,atol=2e-6,msg=name)
        update_error=max(update_error,float((x-y).abs().max().detach()))
    timings={'reference':[],'gathered':[]}
    for repeat in range(6):
        order=[('reference',model,False),('gathered',other,True)]
        if repeat%2:order.reverse()
        for name,m,patched in order:
            m.zero_grad(set_to_none=True);before=time.perf_counter()
            with gather_kernel() if patched else nullcontext():
                loss,_,_,_=objective(m,buf,args);loss.backward()
            if repeat:timings[name].append(time.perf_counter()-before)
    record=dict(status='completed',control=str(path),control_sha256=engine.sha(path),checkpoint_sha256=engine.sha(ck),integrated=dict(input_tokens=64,forward_state_continuation_max_error=0.,all_gradient_max_error=gradient_error,adamw_update_max_error=update_error,graph_nodes=dict(reference=counts_nodes[0],gathered=counts_nodes[1])),embedding_backward=dict(reference=lookup_counts[0],gathered=lookup_counts[1]),interleaved_wall_s=timings,wall_s=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={s:engine.sha(ROOT/s) for s in ['sleeping_machines/chunk_gather_counterfactual_episodes.py','experiments/chunk_gather_probe.py']},scope='One selected trained P24 checkpoint, 64 FIT-derived fixture tokens including duplicates/EOS, exact forward/state/forced-write continuation plus gradients/AdamW parity. Five interleaved CPU objective/backward timings per kernel, no optimizer/evaluation inside timing; no full-fit throughput, FLOP, energy or quality gain. Fixed forced alternative is a contract, not a claimed unbiased training estimator. Existing source kernels unchanged.')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)

if __name__=='__main__':main()
