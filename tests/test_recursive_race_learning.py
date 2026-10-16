import pytest
import torch
from sleeping_machines.race_readout import RaceReadout, event_terms
from sleeping_machines.recursive_race_learning import RaceExposure, NativeRecursiveRaceHeads
from sleeping_machines.relational_frame_composition import RelationalFrameComposition


def setup():
    torch.manual_seed(1031)
    head = RaceReadout(3,1,2,4,4,hidden=8).double()
    mem = torch.randn(1,1,2,4,dtype=torch.float64,requires_grad=True)
    laws = head(mem, mem.mean((1,2)))
    return head, mem, laws, torch.tensor([[-.3,0.]],dtype=torch.float64), torch.tensor([0.],dtype=torch.float64)


def test_silence_partition_preserves_joint_likelihood_and_gradients():
    head,mem,p,refs,start=setup(); end=start+2.3; mark=torch.tensor([1])
    direct=-event_terms(p,refs,start,end,mark,1e-3)[0]
    ledger=RaceExposure(p,refs,start,1e-3)
    split=ledger.silence(start+.2)+ledger.silence(start+1.1)+ledger.event(end,mark)[0]
    assert torch.allclose(direct,split,atol=1e-13,rtol=1e-13)
    params=[mem,*head.parameters()]
    a=torch.autograd.grad(direct.sum(),params,retain_graph=True,allow_unused=True)
    b=torch.autograd.grad(split.sum(),params,allow_unused=True)
    assert all((x is None and y is None) or torch.allclose(x,y,atol=1e-12,rtol=1e-12) for x,y in zip(a,b))


def test_duplicate_queries_have_zero_exposure_and_closed_forecast_rejects_reuse():
    _,_,p,refs,start=setup(); ledger=RaceExposure(p,refs,start,1e-3)
    ledger.silence(start+1.)
    assert torch.equal(ledger.silence(start+1.),torch.zeros(1,dtype=torch.float64))
    with pytest.raises(ValueError):ledger.silence(start)
    ledger.event(start+2.,torch.tensor([0]))
    with pytest.raises(ValueError):ledger.event(start+3.,torch.tensor([0]))


def test_censoring_partitions_telescope_without_a_fictitious_event():
    _,_,p,refs,start=setup()
    a=RaceExposure(p,refs,start,1e-3); b=RaceExposure(p,refs,start,1e-3)
    assert torch.allclose(a.silence(start+3),b.silence(start+1)+b.silence(start+3),atol=1e-14)
    assert not a.closed


def test_native_published_memory_integration_preserves_actor_and_receives_credit():
    torch.manual_seed(1049)
    actor=RelationalFrameComposition(payload=4,heads=2,levels=4,depth=4,classes=2).double()
    state=actor.new_state()
    for i,xy in enumerate([(0,0),(0,6),(6,0),(6,6)]): actor.observe(state,xy,i*.01,[(-1.)**i,1.])
    actor.compose(state); count=sum(state.input_issued.values())
    snapshots=[{a:(r.version,r.values.clone(),r.times.clone(),r.ready.clone()) for a,r in bank.published.items()} for bank in state.banks]
    activity=(state.deliveries,state.publications,state.timer_emissions,state.now)
    heads=NativeRecursiveRaceHeads(actor.width,actor.levels,slots=1,identities=3,hidden=8).double()
    losses=[]
    for level in range(actor.levels):
        address=sorted(state.banks[level].published)[0]
        ledger,work=heads.forecast(actor,state,level,[address])
        assert work['scored_race_slots']==1 and work['readout_madds']>0
        losses.append(ledger.silence(ledger.cursor+1.)+ledger.event(ledger.cursor+2.,torch.tensor([level%3]))[0])
    sum(x.sum() for x in losses).backward()
    assert all(obj.core.content.weight.grad is not None and obj.core.content.weight.grad.norm()>0 for obj in actor.objects)
    assert all(head.out.weight.grad.norm()>0 for head in heads.heads)
    assert sum(state.input_issued.values())==count
    assert activity==(state.deliveries,state.publications,state.timer_emissions,state.now)
    for bank,snapshot in zip(state.banks,snapshots):
        for a,(version,value,times,ready) in snapshot.items():
            r=bank.published[a]
            assert r.version==version and torch.equal(r.values,value) and torch.equal(r.times,times) and torch.equal(r.ready,ready)


def test_label_and_pool_validation():
    _,_,p,refs,start=setup(); ledger=RaceExposure(p,refs,start,1e-3)
    with pytest.raises(ValueError):ledger.event(start+1.,torch.tensor([3]))
    with pytest.raises(ValueError):ledger.event(start+1.,torch.tensor([1.]))
    assert not ledger.closed and torch.equal(ledger.cursor,start)
