import torch
from sleeping_machines.coupled_native_query_credit import ordered_pair_probabilities,CoupledNativeQueryCredit
from sleeping_machines.producer_replay_native_race import ProducerReplayNativeRace
from sleeping_machines.queryable_local_deep_race import QueryableLocalDeepRace


def test_two_memories_constant_legal_assembly_exposes_old_spurious_gradient():
    s=torch.tensor([[.3,-.2],[.4,.1]],dtype=torch.float64,requires_grad=True)
    pi=ordered_pair_probabilities(s);assert torch.allclose(pi.sum(),s.new_tensor(1.))
    values=s.new_tensor([0.,2.]);legal=values[:,None]+values[None,:]
    exact=torch.autograd.grad((pi*legal).sum(),s,retain_graph=True)[0]
    old=torch.autograd.grad(s[0].softmax(0)[1]*(values[1]-values[0]),s)[0]
    assert exact.abs().max()<1e-14 and old.abs().max()>.1


def test_joint_alternative_mean_matches_true_finite_sum_for_every_factual_pair():
    s=torch.tensor([[.2,-.4,.1],[.3,.1,-.2]],dtype=torch.float64,requires_grad=True)
    pi=ordered_pair_probabilities(s).flatten();losses=torch.arange(9,dtype=torch.float64).sin();expected=torch.autograd.grad((pi*losses).sum(),s,retain_graph=True)[0]
    valid=[i for i in range(9) if i//3!=i%3]
    for winner in valid:
        total=0.
        for j in valid:
            if j==winner:continue
            q=(pi[j]/(1-pi[winner])).detach()
            total=total+q*(pi[j]-pi[j].detach())*(losses[j]-losses[winner])/q
        actual=torch.autograd.grad(total,s,retain_graph=True)[0]
        assert torch.allclose(actual,expected,atol=2e-14,rtol=2e-14)


def test_factual_native_predictions_and_bindings_are_unchanged():
    torch.manual_seed(16801);source=QueryableLocalDeepRace(width=8,depth=3,heads=2,pool=3,publication_capacity=3).double()
    base=ProducerReplayNativeRace(source);model=CoupledNativeQueryCredit(source);bs=base.new_state();state=model.new_state()
    bg=torch.Generator().manual_seed(11);g=torch.Generator().manual_seed(11);ba=torch.Generator().manual_seed(12);a=torch.Generator().manual_seed(12)
    packets=[]
    for message in torch.randn(4,8,dtype=torch.float64):
        x,bs,_,_=base.consume(message,bs,bg,ba);y,state,p,_=model.consume(message,state,g,a);packets.extend(p)
        assert torch.equal(x,y)
        assert torch.equal(bg.get_state(),g.get_state())
    y.mul(torch.arange(1,9,dtype=torch.float64)).sum().backward();model.deliver_credit(packets,state)
    assert all(m.weight.grad is not None and m.weight.grad.norm()>0 for m in model.memory_queries)
    assert state['coupled_query_audits']>0
    for entry in state['bindings'].entries.values():
        if len(entry.parts)==2:assert entry.parts[0].source!=entry.parts[1].source
