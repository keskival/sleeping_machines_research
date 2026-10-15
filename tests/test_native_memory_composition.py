import torch
from sleeping_machines.native_memory_composition import NativeMemoryComposition


def make():
    torch.manual_seed(71)
    return NativeMemoryComposition(input_width=2,payload=4,heads=2,levels=8,depth=4,classes=3).double()


def execute(model):
    state=model.new_state()
    model.observe(state,(0,0),0.,[1.,1.])
    model.observe(state,(127,127),.01,[-1.,1.])
    model.drain(state)
    return state


def test_learning_and_evaluation_use_identical_stochastic_computation():
    model=make()
    torch.manual_seed(83);model.train();a=execute(model);ya,ta=model.compose(a)
    torch.manual_seed(83);model.eval();b=execute(model);yb,tb=model.compose(b)
    assert torch.equal(ya,yb) and torch.equal(ta,tb)
    assert a.publications==b.publications and a.queries==b.queries
    for aa,bb in zip(a.banks,b.banks):
        for address in aa.published:
            assert torch.equal(aa.published[address].values,bb.published[address].values)


def test_wide_spatial_support_and_many_memories_reach_same_parent():
    model=make();state=execute(model)
    assert len(state.banks[0].published)==2
    source_count=sum(len(b.published) for b in state.banks[:-1])
    root_before=state.banks[-1].published[(0,0)].version
    output,_=model.compose(state)
    assert state.queries==source_count and source_count>8
    assert state.banks[-1].published[(0,0)].version==root_before+source_count
    assert sum(state.input_issued.values())==2
    output.square().sum().backward()
    assert model.input.weight.grad.abs().sum()>0
    assert model.objects[-1].core.content.weight.grad.abs().sum()>0
    assert not state.queue


def test_query_round_is_scoped_to_the_requested_root():
    model=make();state=execute(model)
    model.observe(state,(300,300),state.now,[.3,1.]);model.drain(state)
    assert (1,1) in state.banks[-1].published
    expected=sum(sum(tuple(v//2**(model.levels-1-level) for v in addr)==(0,0) for addr in bank.published)
                 for level,bank in enumerate(state.banks[:-1]))
    model.compose(state,(0,0));assert state.queries==expected
