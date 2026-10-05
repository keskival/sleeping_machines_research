import torch
from test_token_episodes import model
from sleeping_machines.token_episodes import token_features
from sleeping_machines.counterfactual_token_episodes import token_features as counterfactual_features


def test_no_intervention_matches_factual_features_and_all_gradients():
    m=model();ids=torch.tensor([[0,1,2],[3,4,5]]);runs=[]
    for fn in (token_features,counterfactual_features):
        m.zero_grad(set_to_none=True)
        x,_,_=fn(m,ids,generator=torch.Generator().manual_seed(5))
        m.head(x).square().sum().backward()
        runs.append((x.detach(),{n:p.grad.clone() for n,p in m.named_parameters() if p.grad is not None}))
    assert torch.equal(runs[0][0],runs[1][0])
    for n,g in runs[0][1].items():assert torch.equal(g,runs[1][1][n]),n


def test_equal_immediate_deliveries_can_have_different_future_write_outcomes():
    m=model();ids=torch.tensor([[0,1,2],[3,4,5]])
    with torch.no_grad():
        for layer in m.units:
            for head in layer:
                for pool in head:
                    for unit in pool:
                        unit.key.zero_();unit.key_read.weight.zero_();unit.clock_bias.zero_()
        pool=m.units[0][0][0]
        for unit in pool[1:]:
            for name in ('input','output','gate','control','key_read'):
                setattr(unit,name,getattr(pool[0],name))
    a,sa,_=counterfactual_features(m,ids,deterministic=True,force_site=(0,0,0,torch.zeros(2,dtype=torch.long)))
    b,sb,_=counterfactual_features(m,ids,deterministic=True,force_site=(0,0,0,torch.ones(2,dtype=torch.long)))
    assert torch.equal(a[:,0],b[:,0])
    assert not torch.allclose(a[:,1:],b[:,1:],atol=1e-12,rtol=1e-12)
    assert torch.equal(sa['race_winners'][0,0,:,0],torch.zeros(2,dtype=torch.long))
    assert torch.equal(sb['race_winners'][0,0,:,0],torch.ones(2,dtype=torch.long))


def test_enumerated_paired_suffix_teacher_equals_conditional_expected_loss_gradient():
    m=model();ids=torch.tensor([[0,1,2],[3,4,5]])
    x,st,pis=counterfactual_features(m,ids,generator=torch.Generator().manual_seed(5),suppress_site=(0,0,0))
    pi=pis[0][1][:,0,:];winner=st['race_winners'][0,0,:,0];losses=[]
    with torch.no_grad():
        for choice in range(m.pool):
            xx,_,_=counterfactual_features(m,ids,generator=torch.Generator().manual_seed(5),
                   force_site=(0,0,0,torch.full((2,),choice,dtype=torch.long)))
            losses.append(m.head(xx).square().mean((1,2)))
    costs=torch.stack(losses,-1);base=costs.gather(1,winner[:,None]).squeeze(1)
    params=tuple(u.clock_bias for u in m.units[0][0][0])
    exact=torch.autograd.grad((pi*costs).sum(-1).mean(),params,retain_graph=True)
    # Expectation over every eligible sampled alternative, with its actual suffix cost.
    conditional=pi.detach().clone();conditional.scatter_(1,winner[:,None],0.)
    conditional/=conditional.sum(-1,keepdim=True)
    paired=0.
    for offset in range(1,m.pool):
        alt=(winner+offset)%m.pool;index=alt[:,None]
        q=conditional.gather(1,index).squeeze(1)
        pj=pi.gather(1,index).squeeze(1)
        difference=costs.gather(1,index).squeeze(1)-base
        paired+=((pj-pj.detach())*difference/q*q).mean()
    actual=torch.autograd.grad(paired,params)
    for a,b in zip(exact,actual):assert torch.allclose(a,b,atol=1e-12,rtol=1e-12)
