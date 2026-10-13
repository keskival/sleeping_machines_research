"""R1 actual integrated token gradients -> independent versioned update banks."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import check_window_replay as witness
import window_replay_token_language as driver
from sleeping_machines.learning.local_credit_banks import CreditPacket,LocalCreditBanks


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1)
    from types import SimpleNamespace
    args=SimpleNamespace(seed=6,payload=4,depth=2,heads=2,pool=4,input_init='balanced',tie_pools=False,
        readout='adaptive',readout_init='frequency',minimum_tail_width=0,credit_window=8,state_mode='carry',
        route_credit='sampled',future_site='uniform',free_bias=0.,temperature=1.,compiled=False)
    order=torch.arange(50257);counts=torch.ones(50257,dtype=torch.long)
    torch.manual_seed(6);old=driver.Model(args,order,counts)
    torch.manual_seed(6);new=driver.Model(args,order,counts)
    exact=torch.optim.AdamW(old.parameters(),lr=.003);local=LocalCreditBanks(new.named_parameters(),lr=.003)
    states=[None,None];gens=[torch.Generator().manual_seed(81) for _ in range(2)]
    for step in range(3):
        ids=torch.randint(0,31,(2,17),generator=torch.Generator().manual_seed(91+step));ids[0,9]=50256
        for i,m in enumerate((old,new)):
            m.zero_grad(set_to_none=True);features,states[i],_=m(ids[:,:-1],states[i],gens[i],args)
            m.readout.nll(features,ids[:,1:]).mean().backward()
        exact.step()
        items=list(new.named_parameters())
        if step%2:items.reverse()
        for name,param in items:
            if param.grad is not None:local.apply(CreditPacket(name,local.versions[name],param.grad,f'update{step}'))
        for (name,p),(_,q) in zip(old.named_parameters(),new.named_parameters()):assert torch.equal(p,q),name
        witness.equal_state(*states);states=[driver.detach(s) for s in states]
    name,param=next(new.named_parameters());version=local.versions[name]
    try:local.apply(CreditPacket(name,version-1,torch.zeros_like(param),'stale'))
    except ValueError:stale_rejected=True
    else:raise AssertionError('Stale packet accepted')
    # A local clip depends only on its bank gradient. This deliberately differs
    # from the earlier un-clipped exact Adam equivalence mode.
    left=torch.nn.Parameter(torch.zeros(2));right=torch.nn.Parameter(torch.zeros(2))
    clipped=LocalCreditBanks([('left',left),('right',right)],local_clip=1.)
    clipped.apply(CreditPacket('left',0,torch.tensor([3.,4.]),'clip'))
    assert clipped.versions=={'left':1,'right':0}
    assert torch.equal(right,torch.zeros_like(right))
    source_paths={str(Path(__file__).relative_to(ROOT)), 'sleeping_machines/learning/local_credit_banks.py'}
    saved=json.loads((ROOT/'experiments/results/token_language/aws_integrated_window_replay_512k_20261012_s6_v1.json').read_text())
    sources=dict(saved['source_sha256'])
    source_paths.add('experiments/credit/check_window_replay.py')
    for source in source_paths:sources[source]=hashlib.sha256((ROOT/source).read_bytes()).hexdigest()
    result=dict(status='completed',battle='R1',updates=3,parameter_error=0,stale_packet_rejected=stale_rejected,
        cross_bank_delivery_order_independent=True,storage=local.storage(),source_sha256=sources,
        scope='Actual integrated token gradients with full factual BP,depth2/payload4. Independent leaf-bank Adam matches un-clipped global Adam exactly for three updates with reordered packet delivery. No global clipping/gradient synchronization inside bank application; credit generation still uses window BP. Shared leaves require one canonical bank; multiple same-version contributions must be aggregated before this final-gradient packet. Stale packets rejected, not transported. Local clipping demonstrated separately and changes the learning rule. Packed receiver rows still share a bank; no per-receiver placement, future-credit replacement, quality or hardware win claim.')
    with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}),flush=True)


if __name__=='__main__':main()
