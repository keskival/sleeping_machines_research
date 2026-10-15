import itertools
import torch
from sleeping_machines.relational_frame_composition import ChildRoleComposer,RelationalFrameComposition
from sleeping_machines.frame_native_composition import FrameNativeComposition


def test_ordered_roles_and_explicit_cross_child_interactions():
    composer=ChildRoleComposer(4).double()
    a=torch.tensor([1.,2.,-1.,.5],dtype=torch.float64)
    b=torch.tensor([-.5,1.,2.,-1.],dtype=torch.float64)
    zero=torch.zeros(4,dtype=torch.float64)
    assert not torch.allclose(composer({0:a,1:b}),composer({0:b,1:a}))
    joint=composer({0:a,1:b})-composer({0:zero,1:b})-composer({0:a,1:zero})+composer({0:zero,1:zero})
    assert joint.norm()>1e-4
    composer({0:a}).sum().backward()
    assert composer.pair_weight.grad is None # dormant pair capacity not evaluated


def test_three_level_composition_has_live_eight_way_product_path():
    composer=ChildRoleComposer(2).double()
    with torch.no_grad():
        composer.left.copy_(torch.eye(2));composer.right.copy_(torch.eye(2))
        composer.role_weight.zero_();composer.pair_weight.zero_()
        composer.pair_weight[0,0,0]=1.;composer.pair_weight[1,0,1]=1.
    for signs in itertools.product((-1.,1.),repeat=8):
        values=[torch.tensor([s,1.],dtype=torch.float64,requires_grad=True) for s in signs]
        active=values
        for _ in range(3):active=[composer({0:active[i],1:active[i+1]}) for i in range(0,len(active),2)]
        expected=torch.tensor(signs,dtype=torch.float64).prod()
        assert torch.allclose(active[0][0],expected,atol=1e-5)
        gradients=torch.autograd.grad(active[0][0],values)
        assert all(g[0].abs()>.49 for g in gradients)


def test_native_state_migration_rng_and_all_level_gradient_support():
    args=dict(input_width=2,payload=4,heads=2,levels=8,depth=4,classes=2)
    torch.manual_seed(71);base=FrameNativeComposition(**args).double();after=torch.get_rng_state()
    torch.manual_seed(71);model=RelationalFrameComposition(**args).double()
    assert torch.equal(after,torch.get_rng_state())
    model.load_native_state(base.state_dict())
    torch.manual_seed(73);state=model.new_state()
    for i,xy in enumerate([(0,0),(0,100),(100,0),(100,100)]):model.observe(state,xy,i*.01,[(-1.)**i,1.])
    output,_=model.compose(state);output.square().sum().backward()
    assert all(obj.core.content.weight.grad.abs().sum()>0 for obj in model.objects)
    assert model.objects[-1].core.composer.pair_weight.grad.abs().sum()>0
    work=model.work(state);assert work['role_slots']>0 and work['role_projection_madds']>0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_native_relational_forward_matches_between_training_and_evaluation():
    torch.manual_seed(83)
    m=RelationalFrameComposition(input_width=2,payload=4,heads=2,levels=8,depth=4,classes=2).double()
    outputs=[]
    for training in [True,False]:
        m.train(training);torch.manual_seed(89);state=m.new_state()
        for i,xy in enumerate([(0,0),(0,100),(100,0),(100,100)]):m.observe(state,xy,i*.01,[(-1.)**i,1.])
        outputs.append(m.compose(state))
    assert all(torch.equal(a,b) for a,b in zip(*outputs))
