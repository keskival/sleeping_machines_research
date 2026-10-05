"""Guarded numerical private-bank construction and integrated factual-path contracts."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--tag',required=True)
    a=p.parse_args();packet=json.loads(Path(a.manifest).read_text())
    pins=packet['contract_source_sha256']
    def check():
        for name,digest in pins.items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest: raise ValueError(name)
    check()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if out.exists(): raise FileExistsError(out)
    import torch
    import horizon_token_language_engine as lab
    from aws_private_bank_model import PrivateBankModel
    torch.set_num_threads(1)
    args=SimpleNamespace(seed=6,payload=24,depth=2,heads=2,pool=4,
        input_init='balanced',tie_pools=True,readout='adaptive',readout_init='frequency',
        minimum_tail_width=0,state_mode='carry',route_credit='none',future_site='uniform',
        free_bias=0.,temperature=1.,compiled=False,credit_window=16)
    counts=torch.ones(50257,dtype=torch.long);order=torch.arange(50257)
    torch.manual_seed(6);base=lab.Model(args,order,counts);rng=torch.get_rng_state()
    torch.manual_seed(6);small=PrivateBankModel(args,order,counts)
    assert torch.equal(rng,torch.get_rng_state())
    for k,v in base.state_dict().items(): assert torch.equal(v,small.state_dict()[k]),k
    args.pool=16;torch.manual_seed(6);large=PrivateBankModel(args,order,counts)
    assert torch.equal(rng,torch.get_rng_state())
    private=large.bank_policy['private_fields']
    for k,v in small.state_dict().items():
        if k=='_extra_state':continue
        field=k.removeprefix('core.banks.')
        if field in private:
            want=v.repeat_interleave(4,dim=2)
            if field=='clock_bias':want=want+large.bank_policy['clock_bias_shift']
        else:want=v
        assert torch.equal(want,large.state_dict()[k]),k
    # Storage replicas are independent, although their initial numbers agree.
    key=large.core.banks['key'];before=key[0,0,1].detach().clone()
    with torch.no_grad():key[0,0,0].add_(1)
    assert torch.equal(before,key[0,0,1])
    with torch.no_grad():key[0,0,0].copy_(key[0,0,1])
    try:small.load_state_dict(large.state_dict())
    except (ValueError,RuntimeError):pass
    else:raise AssertionError('Cross-bank checkpoint was accepted')
    # Rebuild after the deliberately rejected load, which can mutate matching leaves.
    args.pool=4;torch.manual_seed(6);small=PrivateBankModel(args,order,counts)
    rows=[]
    for model,pool in [(small,4),(large,16)]:
        args.pool=pool
        ids=torch.tensor([[100,101,102,103,104,105,106,107]])
        targets=ids+1
        x,state,pis=model(ids,None,torch.Generator().manual_seed(9),args)
        loss=model.readout.nll(x,targets).mean();loss.backward()
        assert torch.isfinite(loss)
        assert int(state['last_writes'].sum())==8*2*2
        assert all(v.shape[2]==pool for v in state['mem'])
        assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
        rows.append(dict(pool=pool,nll=float(loss.detach()),selected_writes=int(state['last_writes'].sum()),
                         parameters=sum(v.numel() for v in model.parameters())))
    check();out.parent.mkdir(parents=True,exist_ok=True)
    record=dict(status='completed',construction_rng_exact=True,canonical_parameter_parity=True,
        shared_rule_parity=True,private_replica_storage=True,cross_recipe_checkpoint_rejected=True,
        factual_gradient_finite=True,rows=rows,source_sha256=pins,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Synthetic token factual-path contracts only; future-write teacher and training quality require bounded integrated fits.')
    out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)


if __name__=='__main__':main()
