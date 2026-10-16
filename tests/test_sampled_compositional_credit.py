import io
import torch
from sleeping_machines.queryable_local_deep_race import QueryableLocalDeepRace
from sleeping_machines.coupled_native_query_credit import CoupledNativeQueryCredit
from sleeping_machines.sampled_compositional_credit import SampledCompositionalCredit
from tests.test_compositional_producer_credit import oracle
from tests.test_native_query_migration import identical


def setup():
 torch.manual_seed(16911);s=QueryableLocalDeepRace(width=8,depth=4,heads=2,pool=3,publication_capacity=4).double();return CoupledNativeQueryCredit(s),SampledCompositionalCredit(s)


def paths(state,key):
 if key not in state['compositional_lineage']:return [((),1.)]
 children=state['compositional_lineage'][key].sources
 return [((r,)+suffix,p/len(children)) for r,child in enumerate(children) for suffix,p in paths(state,child)]


def test_expected_sampled_path_gradient_matches_complete_conditional_dag():
 _,m=setup();state=m.new_state();g=torch.Generator().manual_seed(91);a=torch.Generator().manual_seed(92)
 for x in torch.randn(8,8,dtype=torch.float64):m.consume(x,state,g,a)
 key=state['publications'][3][-1];now=state['bindings'].entries[key].available+1;weight=torch.arange(1,9,dtype=torch.float64)
 m.zero_grad(set_to_none=True);full=oracle(m,state,key,now,3);(full*weight).sum().backward()
 expected={n:p.grad.clone() for n,p in m.named_parameters() if p.grad is not None and (n=='mix' or n.startswith('assemblers.'))};got={n:torch.zeros_like(v) for n,v in expected.items()}
 possible=paths(state,key);assert abs(sum(p for _,p in possible)-1)<1e-14
 for path,p in possible:
  choices=list(path)
  m.select_credit_role=lambda state,roles:(choices.pop(0),1/len(roles))
  m.zero_grad(set_to_none=True);before=state['recursive_replay_edges'];value=m.read_publication(state,key,now,[],learn=True)
  assert state['recursive_replay_edges']-before<=m.depth-1 and not choices
  assert torch.allclose(value,full,atol=3e-11,rtol=3e-11)
  (value*weight).sum().backward()
  for n,param in m.named_parameters():
   if n in got and param.grad is not None:got[n]+=p*param.grad
 for n in expected:torch.testing.assert_close(got[n],expected[n],atol=2e-8,rtol=2e-8)
 assert got['assemblers.0.pair_weight'].norm()>0


def test_factual_rng_parity_and_dedicated_credit_rng_recovery():
 base,m=setup();bs=base.new_state();state=m.new_state();bg=torch.Generator().manual_seed(11);g=torch.Generator().manual_seed(11);ba=torch.Generator().manual_seed(12);a=torch.Generator().manual_seed(12)
 for x in torch.randn(5,8,dtype=torch.float64):
  y,bs,_,_=base.consume(x,bs,bg,ba);z,state,_,_=m.consume(x,state,g,a)
  assert torch.equal(y,z) and torch.equal(bg.get_state(),g.get_state()) and torch.equal(ba.get_state(),a.get_state())
 checkpoint=io.BytesIO();torch.save((m.state_dict(),state,g.get_state(),a.get_state()),checkpoint);x=torch.randn(8,dtype=torch.float64)
 expected=m.consume(x,state,g,a);checkpoint.seek(0);w,state,gs,als=torch.load(checkpoint,weights_only=False);m.load_state_dict(w);g.set_state(gs);a.set_state(als);actual=m.consume(x,state,g,a)
 assert torch.equal(expected[0],actual[0]) and identical(expected[1],actual[1]) and expected[3]==actual[3]
