"""First-order/integrated parity and interleaved CPU timings; guarded queue only."""
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
from sleeping_machines.explicit_rotation_backward import explicit_rotate,rotation_kernel
from sleeping_machines.parallel_stream_language import precise_rotate
from sleeping_machines.paired_route_credit import paired_route_credit


def primitive_contracts():
    errors=[]
    generator=torch.Generator().manual_seed(713)
    for dtype in (torch.float32,torch.float64):
        for shape,angle_shape in [((8,24),(8,12)),((8,24),(8,1)),((8,2,24),(8,2,12)),((8,24),(12,))]:
            for scale in (0.,1.,1e6):
                value=torch.randn(shape,generator=generator,dtype=dtype)
                angles=torch.randn(angle_shape,generator=generator,dtype=torch.float64)*scale
                upstream=torch.randn(shape,generator=generator,dtype=dtype)
                gradients=[];outputs=[]
                for function in (precise_rotate,explicit_rotate):
                    v=value.clone().requires_grad_();a=angles.clone().requires_grad_()
                    output=function(v,a);outputs.append(output.detach())
                    gradients.append(torch.autograd.grad(output,(v,a),upstream))
                torch.testing.assert_close(*outputs,rtol=0,atol=0)
                for first,second in zip(*gradients):
                    torch.testing.assert_close(first,second,rtol=2e-6,atol=2e-6)
                    errors.append(float((first-second).abs().max()))
    value=torch.randn(2,4,dtype=torch.float64,requires_grad=True)
    angle=torch.randn(2,2,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(explicit_rotate,(value,angle),eps=1e-6,atol=1e-5,rtol=1e-4)
    return dict(cases=24,forward_max_error=0.,first_gradient_max_error=max(errors),finite_difference_passed=True,higher_derivatives_supported=False)


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
    torch.set_num_threads(1);start=time.monotonic();primitive=primitive_contracts()
    path=Path(a.control);result=json.loads(path.read_text());assert result['status']=='completed'
    for name,digest in result['identity']['source_sha256'].items():assert engine.sha(ROOT/name)==digest,name
    args=SimpleNamespace(**result['args']);train=engine.load_tokens(args.train_file)
    counts=engine.LaneTokens(train,0,args.train_tokens,args.lanes).counts()
    selection=json.loads(path.with_suffix('.selection.json').read_text())['selected'];ck=Path(selection['checkpoint'])
    model=engine.Model(args,torch.argsort(counts,descending=True,stable=True),counts)
    model.load_state_dict(torch.load(ck,weights_only=True));other=copy.deepcopy(model)
    buf=engine.interval_tensor(train,0,72,8)
    if buf.shape!=(8,9):raise ValueError(buf.shape)
    opt=[torch.optim.AdamW(m.parameters(),lr=args.lr) for m in (model,other)]
    snapshots=[];counts_nodes=[]
    for m,o,patch in [(model,opt[0],nullcontext()),(other,opt[1],rotation_kernel())]:
        o.zero_grad(set_to_none=True)
        with patch:
            loss,x,state,difference=objective(m,buf,args);counts_nodes.append(graph_nodes(loss));loss.backward()
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
    timings={'reference':[],'explicit':[]}
    for repeat in range(6):
        order=[('reference',model,False),('explicit',other,True)]
        if repeat%2:order.reverse()
        for name,m,patched in order:
            m.zero_grad(set_to_none=True);before=time.perf_counter()
            with rotation_kernel() if patched else nullcontext():
                loss,_,_,_=objective(m,buf,args);loss.backward()
            if repeat:timings[name].append(time.perf_counter()-before)
    record=dict(status='completed',primitive=primitive,control=str(path),control_sha256=engine.sha(path),checkpoint_sha256=engine.sha(ck),integrated=dict(input_tokens=64,forward_state_continuation_max_error=0.,all_gradient_max_error=gradient_error,adamw_update_max_error=update_error,graph_nodes=dict(reference=counts_nodes[0],explicit=counts_nodes[1])),interleaved_wall_s=timings,wall_s=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={s:engine.sha(ROOT/s) for s in ['sleeping_machines/explicit_rotation_backward.py','experiments/rotation_backward_probe.py']},scope='One selected trained P24 checkpoint, 64 FIT tokens, exact forward/state/forced-write continuation plus first-order gradients/AdamW parity. Five interleaved CPU objective/backward timings per kernel, no optimizer/evaluation inside timing; no full-fit throughput, FLOP, energy or quality gain. Fixed forced alternative is a contract, not a claimed unbiased training estimator. Existing source kernels unchanged.')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)

if __name__=='__main__':main()
