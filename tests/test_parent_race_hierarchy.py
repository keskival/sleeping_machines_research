import torch
from sleeping_machines.parent_race_hierarchy import ParentRaceCell, ParentRaceHierarchy


def test_candidate_order_independent_when_content_and_pose_travel_together():
    torch.manual_seed(7)
    cell = ParentRaceCell(3,4).double()
    candidates, valid, shape = cell.candidates(torch.randn(1,3,4,4,dtype=torch.double))
    old = torch.randn(1,4,4,dtype=torch.double)
    a, aa = cell.bind(candidates,valid,old,sample=False)
    permutation = torch.tensor([3,0,8,1,7,2,6,4,5,9])
    b, bb = cell.bind(candidates[:,:,permutation],valid[:,:,permutation],old,sample=False)
    assert torch.allclose(a,b)
    assert torch.equal(aa['winners'],permutation[bb['winners']])


def test_silence_preserves_state_and_emits_zero():
    cell = ParentRaceCell(3,4)
    candidates, valid, _ = cell.candidates(torch.rand(1,3,4,4))
    valid[:,:,:-1] = False
    old = torch.randn(1,4,4)
    new, audit = cell.bind(candidates,valid,old)
    assert torch.equal(new,old)
    assert not audit['emission'].any()
    assert not audit['active'].any()


def test_complementary_roles_can_bind_different_children():
    cell = ParentRaceCell(2,2,roles=2).double()
    with torch.no_grad():
        cell.key.weight.zero_(); cell.key.weight[:,:2] = torch.eye(2,dtype=torch.double)
        cell.query.weight.zero_(); cell.query.bias.zero_()
        cell.role_queries.copy_(torch.eye(2,dtype=torch.double)*4)
    candidates = torch.tensor([[[[2.,0.,0.,0.],[0.,2.,0.,0.],[0.,0.,0.,0.]]]],dtype=torch.double)
    valid = torch.ones(1,1,3,dtype=torch.bool)
    _, audit = cell.bind(candidates,valid,torch.zeros(1,1,2,dtype=torch.double),sample=False)
    assert audit['winners'].tolist() == [[[0,1]]]


def test_persistence_feedback_and_32_cell_replay_do_not_mutate_inputs():
    torch.manual_seed(1)
    model = ParentRaceHierarchy(input_width=3,width=4).double()
    features = torch.rand(1,3,8,8,dtype=torch.double)
    first = model(features,seed=2)
    assert len(first['audits']) == 32
    saved = [s.clone() for s in first['state']]
    second = model(features,first['state'],seed=3)
    repeat = model(features,first['state'],seed=3)
    assert torch.equal(second['logits'],repeat['logits'])
    assert all(torch.equal(a,b) for a,b in zip(saved,first['state']))
    assert any(not torch.equal(a,b) for a,b in zip(saved,second['state']))
    assert not torch.equal(second['logits'],model(features,seed=3)['logits'])


def test_force_factual_replays_exactly_and_final_loss_teaches_role_keys():
    torch.manual_seed(5)
    model = ParentRaceHierarchy(input_width=3,width=4,levels=3,refinements=3,roles=2).double()
    features = torch.rand(1,3,8,8,dtype=torch.double)
    factual = model(features,seed=4)
    winner = int(factual['audits'][0]['winners'][0,0,0])
    replay = model(features,seed=4,force=(0,0,0,winner))
    assert torch.equal(factual['logits'],replay['logits'])
    loss,info = model.route_objective(features,torch.tensor([2]),0,seed=4)
    loss.backward()
    gradient = model.cells[0][0].key.weight.grad
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
    assert info['replays'] == 5  # corner: four valid children plus null


def test_old_parent_queries_change_child_binding_scores():
    torch.manual_seed(8)
    cell = ParentRaceCell(3,4).double()
    children = torch.rand(1,3,4,4,dtype=torch.double)
    old = torch.zeros(1,4,2,2,dtype=torch.double)
    _, a = cell(children,old,feedback=torch.zeros_like(old),sample=False)
    _, b = cell(children,old,feedback=torch.ones_like(old),sample=False)
    assert not torch.allclose(a['probabilities'],b['probabilities'])


def test_child_transport_keeps_earliest_final_task_key_credit_visible_in_float32():
    torch.manual_seed(17)
    model = ParentRaceHierarchy(input_width=3,width=8,levels=8,refinements=4,roles=3,classes=3)
    features = torch.rand(1,3,16,16)
    loss,info = model.route_objective(features,torch.tensor([1]),0,seed=19)
    loss.backward()
    grad = model.cells[0][0].key.weight.grad
    assert torch.isfinite(grad).all() and grad.abs().sum()>0


def test_cell_balanced_audit_reports_correct_nonuniform_probability():
    model = ParentRaceHierarchy(input_width=3,width=4,levels=3,refinements=2,roles=2)
    features = torch.rand(1,3,8,8)
    rng = torch.Generator().manual_seed(5)
    counts = [32,32,8,8,2,2]
    visited = set()
    for _ in range(100):
        index,q = model.sample_audit_site(features,rng)
        offset = index
        for cell,count in enumerate(counts):
            if offset<count:
                break
            offset-=count
        visited.add(cell)
        assert q == 1/(6*counts[cell])
    assert len(visited) == 6


def test_event_adapter_integrates_time_without_extending_observation_window():
    from sleeping_machines.parent_race_hierarchy import EventParentRaceHierarchy
    model = EventParentRaceHierarchy(bins=2,width=4,levels=3,refinements=2,roles=2,classes=3)
    events = torch.rand(1,2,2,8,8)
    loss,info = model.route_objective(events,torch.tensor([1]),0,seed=6)
    loss.backward()
    assert info['output']['logits'].shape == (1,3)
    assert torch.isfinite(model.log_rates.grad).all()
    assert model.log_rates.grad.abs().sum()>0
