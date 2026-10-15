import torch
from sleeping_machines.frontier_native_composition import FrontierNativeComposition


class TraceFrontier(FrontierNativeComposition):
    def query(self,state,source_level,source_address,target_level,target_address):
        self.trace.append((source_level,source_address,target_level,target_address))
        return super().query(state,source_level,source_address,target_level,target_address)


def test_only_adjacent_memories_are_queried_and_all_levels_compose():
    torch.manual_seed(19)
    m=TraceFrontier(input_width=2,payload=4,heads=2,levels=8,depth=4,classes=2).double()
    m.trace=[];state=m.new_state()
    for i,(y,x) in enumerate([(0,0),(127,127),(0,127),(127,0)]):m.observe(state,(y,x),i*.01,[(-1.)**i,1.])
    out,time=m.compose(state)
    assert {q[2] for q in m.trace}==set(range(1,8))
    assert all(q[0]+1==q[2] and tuple(v//2 for v in q[1])==q[3] for q in m.trace)
    assert len([q for q in m.trace if q[2]==7])==len(state.banks[6].published)
    assert state.queries==len(m.trace) and state.timer_emissions==1
    assert sum(state.input_issued.values())==4 and not state.queue
    out.square().sum().backward()
    assert torch.isfinite(time)
    assert all(obj.core.content.weight.grad.abs().sum()>0 for obj in m.objects)


def test_frontier_queries_respect_root_support():
    m=TraceFrontier(input_width=2,payload=4,heads=2,levels=8,depth=4,classes=2).double()
    m.trace=[];state=m.new_state()
    m.observe(state,(0,0),0.,[1.,1.]);m.observe(state,(300,300),.01,[1.,1.])
    m.compose(state)
    assert all(tuple(v//2**(7-q[0]) for v in q[1])==(0,0) for q in m.trace)
