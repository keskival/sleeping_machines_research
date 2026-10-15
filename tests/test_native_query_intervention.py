import torch
from sleeping_machines.native_memory_composition import NativeMemoryComposition
from experiments.hierarchy.native_query_attribution import ZeroQueryContent


def test_zero_query_content_preserves_metadata_event_counts_and_noise_consumption():
    torch.manual_seed(173)
    kw=dict(input_width=2,payload=4,heads=2,levels=3,depth=2,classes=3)
    full=NativeMemoryComposition(**kw).double().eval();zero=ZeroQueryContent(**kw).double().eval()
    zero.load_state_dict(full.state_dict())
    def prepare(model):
        torch.manual_seed(179);state=model.new_state()
        model.observe(state,(0,0),0.,[1.,1.]);model.observe(state,(7,7),.01,[-1.,1.]);model.drain(state)
        return state
    a,b=prepare(full),prepare(zero)
    for x,y in zip(a.banks,b.banks):
        for address in x.published:assert torch.equal(x.published[address].values,y.published[address].values)
    def trace(model,state):
        rows=[];core=model.objects[-1].core;original=core.consume_event
        def consume(source,timestamp,content,private):
            rows.append(content.detach().clone())
            return original(source,timestamp,content,private)
        core.consume_event=consume
        torch.manual_seed(181);model.compose(state)
        return rows,torch.get_rng_state()
    aa,ra=trace(full,a);bb,rb=trace(zero,b)
    assert a.queries==b.queries and a.publications==b.publications
    assert torch.equal(ra,rb)
    for x,y in zip(aa,bb):
        assert torch.equal(x[-6:],y[-6:])
        assert not y[:-6].any()
    assert any(x[:-6].abs().sum()>0 for x in aa)
