import torch
from sleeping_machines.addressable_parent_hierarchy import AddressableParentHierarchy
from sleeping_machines.cached_parent_hierarchy import CachedParentHierarchy


def pair():
    torch.manual_seed(6)
    a=AddressableParentHierarchy(input_width=3,width=4,levels=3,refinements=2,roles=2,classes=3).double()
    b=CachedParentHierarchy(input_width=3,width=4,levels=3,refinements=2,roles=2,classes=3).double()
    b.load_state_dict(a.state_dict())
    return a,b


def test_every_cell_and_valid_alternative_suffix_matches_full_replay():
    a,b=pair();features=torch.rand(1,3,8,8,dtype=torch.double)
    output=b(features,seed=5)
    for index,audit in enumerate(output['audits']):
        for candidate in audit['valid'][0,0].nonzero().flatten().tolist():
            full=a(features,seed=5,force=(index,0,0,candidate))
            suffix,executed=b.suffix_logits(output,index,0,0,candidate)
            assert torch.equal(full['logits'],suffix)
            assert executed==6-index


def test_objective_and_all_parameter_gradients_equal_full_replay():
    a,b=pair();features=torch.rand(1,3,8,8,dtype=torch.double);labels=torch.tensor([1])
    site=2*4*4*2+2*2*2*2
    loss_a,_=a.route_objective(features,labels,site,seed=7)
    loss_b,info=b.route_objective(features,labels,site,seed=7)
    assert torch.equal(loss_a,loss_b)
    loss_a.backward();loss_b.backward()
    for (name,pa),(_,pb) in zip(a.named_parameters(),b.named_parameters()):
        if pa.grad is None: assert pb.grad is None,name
        else: assert torch.equal(pa.grad,pb.grad),name
    assert info['suffix_cell_evaluations']<info['full_replay_cell_evaluations']


def test_suffix_matches_with_retained_state_feedback_and_blocked_sources():
    _,b=pair();features=torch.rand(1,3,8,8,dtype=torch.double)
    memory=b(features,seed=2)['memory']
    output=b(None,memory,seed=3,blocked_reads=((2,0),))
    candidate=int(output['audits'][4]['winners'][0,0,0])
    logits,_=b.suffix_logits(output,4,0,0,candidate)
    full=b(None,memory,seed=3,force=(4,0,0,candidate),blocked_reads=((2,0),))
    assert torch.equal(logits,full['logits'])
