import torch
from sleeping_machines.token_episodes_compiled import token_features as dense_features
from sleeping_machines.sparse_token_episodes import token_features as sparse_features, detach
from test_token_episodes import model


def test_sparse_factual_logits_and_every_gradient_match_dense_with_persistent_state_and_new_weights():
    m=model();ids=torch.tensor([[0,1,2],[3,6,4]])
    _,st,_=sparse_features(m,ids,generator=torch.Generator().manual_seed(3),route_credit='none',eos=6)
    st=detach(st)
    with torch.no_grad():
        for p in m.parameters():p.add_(.013)
    runs=[]
    for fn in (dense_features,sparse_features):
        m.zero_grad(set_to_none=True)
        x,ss,_=fn(m,ids,st,generator=torch.Generator().manual_seed(9),route_credit='none',eos=6)
        m.head(x).square().sum().backward()
        runs.append((x.detach(),{n:p.grad.clone() for n,p in m.named_parameters() if p.grad is not None},ss))
    a,b=runs
    assert torch.allclose(a[0],b[0],atol=1e-12,rtol=1e-12)
    assert torch.equal(a[2]['last_writes'],b[2]['last_writes'])
    for n,g in a[1].items():assert torch.allclose(g,b[1][n],atol=1e-10,rtol=1e-10),n


def test_sampled_credit_changes_gradients_without_changing_factual_state_or_features():
    m=model();ids=torch.tensor([[0,1,2],[3,4,5]])
    a,sa,_=sparse_features(m,ids,generator=torch.Generator().manual_seed(3),route_credit='none')
    b,sb,_=sparse_features(m,ids,generator=torch.Generator().manual_seed(3),
                          alternative_generator=torch.Generator().manual_seed(43))
    assert torch.equal(a,b)
    assert all(torch.equal(x,y) for x,y in zip(sa['mem'],sb['mem']))
    ga=torch.autograd.grad(a.square().sum(),m.units[0][0][0][0].key,retain_graph=True)[0]
    gb=torch.autograd.grad(b.square().sum(),m.units[0][0][0][0].key)[0]
    assert not torch.allclose(ga,gb,atol=1e-12,rtol=1e-12)


def test_sampled_partition_preserves_both_generators_and_state():
    m=model();ids=torch.tensor([[0,1,2,3],[3,4,5,6]])
    g=torch.Generator().manual_seed(8);ag=torch.Generator().manual_seed(12)
    a,sa,_=sparse_features(m,ids,generator=g,alternative_generator=ag,eos=6)
    g2=torch.Generator().manual_seed(8);ag2=torch.Generator().manual_seed(12)
    b,sb,_=sparse_features(m,ids[:,:2],generator=g2,alternative_generator=ag2,eos=6)
    c,sb,_=sparse_features(m,ids[:,2:],sb,generator=g2,alternative_generator=ag2,eos=6)
    assert torch.equal(a,torch.cat((b,c),1))
    assert torch.equal(g.get_state(),g2.get_state()) and torch.equal(ag.get_state(),ag2.get_state())
    assert all(torch.equal(x,y) for x,y in zip(sa['mem'],sb['mem']))
