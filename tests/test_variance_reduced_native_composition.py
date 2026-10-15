import torch
from sleeping_machines.native_memory_composition import NativeMemoryComposition
from sleeping_machines.variance_reduced_native_composition import VarianceReducedNativeComposition


def test_complete_native_forward_and_parameters_stay_bitwise_identical():
    kwargs=dict(input_width=2,payload=4,heads=2,levels=8,depth=4,classes=3)
    torch.manual_seed(191);a=NativeMemoryComposition(**kwargs).double();ra=torch.get_rng_state()
    torch.manual_seed(191);b=VarianceReducedNativeComposition(**kwargs).double();rb=torch.get_rng_state()
    assert torch.equal(ra,rb)
    for key,value in a.state_dict().items():assert torch.equal(value,b.state_dict()[key]),key
    def execute(model):
        torch.manual_seed(193);state=model.new_state()
        model.observe(state,(0,0),0.,[1.,1.]);model.observe(state,(127,127),.01,[-1.,1.])
        logits,time=model.compose(state)
        return logits,time,state
    ya,ta,sa=execute(a);yb,tb,sb=execute(b)
    assert torch.equal(ya,yb) and torch.equal(ta,tb)
    assert sa.queries==sb.queries and sa.publications==sb.publications
    for aa,bb in zip(sa.banks,sb.banks):
        for address in aa.published:assert torch.equal(aa.published[address].values,bb.published[address].values)
    yb.square().sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in b.parameters() if p.grad is not None)
