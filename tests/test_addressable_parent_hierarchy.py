import torch
from sleeping_machines.addressable_parent_hierarchy import AddressableParentHierarchy, memory_candidates
from sleeping_machines.parent_race_hierarchy import ParentRaceCell


def test_ancestor_read_distinguishes_detail_erased_from_immediate_parent():
    # Two worlds with exactly the same compressed lower-level representations.
    raw_a=torch.zeros(1,4,16,16,dtype=torch.double)
    raw_b=raw_a.clone(); raw_b[:,:,0,0]=2
    lower=[torch.zeros(1,4,8,8,dtype=torch.double),torch.zeros(1,4,4,4,dtype=torch.double)]
    a,valid,_=memory_candidates([raw_a,*lower],2,(2,2))
    b,_,_=memory_candidates([raw_b,*lower],2,(2,2))
    cell=ParentRaceCell(5,4,roles=2).double()
    state=torch.zeros(1,4,4,dtype=torch.double)
    _,aa=cell.bind(a,valid,state,force=(0,0,4),sample=False)
    _,bb=cell.bind(b,valid,state,force=(0,0,4),sample=False)
    assert not torch.equal(aa['chosen'],bb['chosen'])
    adjacent_a,_,_=memory_candidates([raw_a,*lower],2,(2,2),False)
    adjacent_b,_,_=memory_candidates([raw_b,*lower],2,(2,2),False)
    assert torch.equal(adjacent_a,adjacent_b)


def test_address_stencil_never_reads_outside_source_and_keeps_source_type():
    memories=[torch.ones(1,4,7,5),torch.ones(1,4,4,3)*2]
    bank,valid,addresses=memory_candidates(memories,1,(2,2))
    assert bank.shape==(1,4,19,7)
    assert len(addresses)==19
    assert not valid[0,0,0]
    assert valid[:,:,-1].all() and not bank[:,:,-1].any()
    assert torch.equal(bank[:,:,4,4],torch.zeros(1,4))
    assert torch.equal(bank[:,:,13,4],torch.ones(1,4)*.5)


def test_full_32_cell_replay_and_memory_reuse_are_deterministic_and_immutable():
    torch.manual_seed(11)
    model=AddressableParentHierarchy(input_width=3,width=4,levels=8,refinements=4,roles=2)
    features=torch.rand(1,3,16,16)
    first=model(features,seed=7)
    assert len(first['audits'])==32
    winner=int(first['audits'][-1]['winners'][0,0,0])
    forced=model(features,seed=7,force=(31,0,0,winner))
    assert torch.equal(first['logits'],forced['logits'])
    snapshot=[v.clone() for v in first['memory']['levels']]
    second=model(None,first['memory'],seed=8)
    repeat=model(None,first['memory'],seed=8)
    assert torch.equal(second['logits'],repeat['logits'])
    assert all(torch.equal(a,b) for a,b in zip(snapshot,first['memory']['levels']))
    assert first['memory']['input'].grad_fn is not None


def test_final_task_credit_teaches_highest_level_ancestor_selector():
    torch.manual_seed(9)
    model=AddressableParentHierarchy(input_width=3,width=4,levels=3,refinements=2,roles=2).double()
    features=torch.rand(1,3,8,8,dtype=torch.double)
    # Two cells at 4x4, two at 2x2, then the first 1x1 top cell.
    loss,info=model.route_objective(features,torch.tensor([2]),2*4*4*2+2*2*2*2,seed=4)
    loss.backward()
    grad=model.cells[2][0].key.weight.grad
    assert torch.isfinite(grad).all() and grad.abs().sum()>0
    assert info['replays']==13  # 4 valid addresses per source x 3 sources + null
    assert model.input.weight.grad.abs().sum()>0


def test_scoring_cost_is_charged_for_every_memory_source():
    model=AddressableParentHierarchy(width=4,roles=2)
    descriptor=model.work_descriptor(128,128)
    assert descriptor['levels'][7]['candidates_per_site']==73
    assert descriptor['levels'][7]['key_scores']==1*2*73*4
    assert descriptor['levels'][7]['selected_deliveries']==8


def test_event_wrapper_can_refine_retained_memory_without_new_observation():
    from sleeping_machines.addressable_parent_hierarchy import EventAddressableParentHierarchy
    model=EventAddressableParentHierarchy(bins=2,width=4,levels=3,refinements=2,roles=2)
    initial=model(torch.rand(1,2,2,8,8),seed=5)
    refined=model(None,initial['memory'],seed=6)
    assert torch.isfinite(refined['logits']).all()
    assert torch.equal(initial['memory']['input'],refined['memory']['input'])


def test_source_intervention_preserves_tape_shape_and_other_levels():
    torch.manual_seed(15)
    model=AddressableParentHierarchy(input_width=3,width=4,levels=3,refinements=2,roles=2)
    features=torch.rand(1,3,8,8)
    factual=model(features,seed=3)
    blocked=model(features,seed=3,blocked_reads=((2,0),))
    for a,b in zip(factual['audits'][:4],blocked['audits'][:4]):
        assert torch.equal(a['winners'],b['winners'])
    assert factual['audits'][4]['probabilities'].shape==blocked['audits'][4]['probabilities'].shape
    assert not blocked['audits'][4]['probabilities'][:,:,:,:9].any()
