import copy,io,math
import torch
from torch.nn import functional as F
from sleeping_machines.queryable_local_deep_race import QueryableLocalDeepRace
from sleeping_machines.coupled_native_query_credit import CoupledNativeQueryCredit
from sleeping_machines.compositional_producer_credit import CompositionalProducerCredit
from sleeping_machines.producer_replay_native_race import local_write
from tests.test_native_query_migration import identical


def setup(depth=3):
 torch.manual_seed(16911);source=QueryableLocalDeepRace(width=8,depth=depth,heads=2,pool=3,publication_capacity=4).double()
 return CoupledNativeQueryCredit(source),CompositionalProducerCredit(source)


def acquire(model,n=8):
 state=model.new_state();g=torch.Generator().manual_seed(91);a=torch.Generator().manual_seed(92)
 for message in torch.randn(n,8,dtype=torch.float64):model.consume(message,state,g,a)
 return state,g,a


def test_forward_rng_and_numerical_lineage_are_preserved():
 base,model=setup();bs=base.new_state();s=model.new_state();bg=torch.Generator().manual_seed(91);g=torch.Generator().manual_seed(91);ba=torch.Generator().manual_seed(92);a=torch.Generator().manual_seed(92)
 for message in torch.randn(6,8,dtype=torch.float64):
  x,bs,_,_=base.consume(message,bs,bg,ba);y,s,_,_=model.consume(message,s,g,a)
  assert torch.equal(x,y) and torch.equal(bg.get_state(),g.get_state()) and torch.equal(ba.get_state(),a.get_state())
 assert s['recursive_replay_edges']>0 and s['compositional_lineage']
 for key,r in s['compositional_lineage'].items():
  assert key in s['bindings'].entries
  assert all(t.grad_fn is None and not t.requires_grad for t in [*r.issued,*r.reply_times,r.ready])
  assert all(c in s['bindings'].entries and c[0]<key[0] for c in r.sources)


def oracle(model,state,key,now,hops):
 """Independent finite conditional DAG, with actual values rather than zero-forward carriers."""
 record=state['bindings'].entries[key].native;p=state['producer_records'][key];views=dict(p.views);d=key[0]
 if hops and key in state['compositional_lineage']:
  lineage=state['compositional_lineage'][key]
  for role,child in enumerate(lineage.sources):
   value=oracle(model,state,child,lineage.issued[role],hops-1)
   views[role]=model.aligned_child(d,value,lineage,role)
 stream=p.base_input+p.inbox
 if views:stream=stream+model.assemblers[d-1](views)
 mixed=F.linear(stream,model.mix[d]).reshape(model.heads,model.payload)
 memory=local_write(p.prior_value,p.write_parameters,mixed,p.elapsed)
 hi=torch.arange(model.heads);winner=torch.tensor(record.winners);params={n:b[d,hi,winner] for n,b in model.banks.items()}
 values=model.unit_value(params,mixed,memory);age=now-record.head_times
 mag=torch.exp(-age[:,None]*(F.softplus(model.transport_rate[d])+1e-6));angle=age[:,None]*model.transport_frequency[d]
 pair=values.reshape(model.heads,-1,2);re,im=pair[...,0]*mag,pair[...,1]*mag
 aligned=torch.stack((re*angle.cos()-im*angle.sin(),re*angle.sin()+im*angle.cos()),-1).flatten()
 return F.layer_norm(stream+aligned/math.sqrt(model.depth),(model.width,))


def test_delayed_upper_loss_reaches_child_composer_and_matches_finite_dag():
 base,model=setup();state,_,_=acquire(model);key=state['publications'][2][-1];now=state['bindings'].entries[key].available+1;weight=torch.arange(1,9,dtype=torch.float64)
 base.load_state_dict(model.state_dict());base.zero_grad(set_to_none=True)
 before=base.read_publication(state,key,now,[],learn=True);(before*weight).sum().backward()
 assert base.assemblers[0].pair_weight.grad is None
 model.zero_grad(set_to_none=True);packets=[];start=state['recursive_replay_edges'];actual=model.read_publication(state,key,now,packets,learn=True);(actual*weight).sum().backward()
 assert torch.equal(actual,before)
 assert model.assemblers[0].pair_weight.grad is not None and model.assemblers[0].pair_weight.grad.norm()>0
 got={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
 assert state['recursive_replay_edges']-start<=model.replay_edge_budget
 model.zero_grad(set_to_none=True);expected=oracle(model,state,key,now,2)
 assert torch.allclose(actual,expected,atol=3e-11,rtol=3e-11)
 (expected*weight).sum().backward()
 for n,p in model.named_parameters():
  if n=='mix' or n.startswith('assemblers.'):
   assert p.grad is not None and torch.allclose(p.grad,got[n],atol=2e-8,rtol=2e-8),n


def test_work_cap_and_full_numerical_state_recovery():
 _,model=setup(depth=4);model.replay_edge_budget=1;state,g,a=acquire(model,5);key=state['publications'][3][-1]
 old=state['recursive_replay_edges'];model.read_publication(state,key,state['bindings'].entries[key].available+1,[],learn=True)
 assert state['recursive_replay_edges']-old<=1 and state['recursive_replay_truncated']>0
 payload=io.BytesIO();torch.save((model.state_dict(),state,g.get_state(),a.get_state()),payload);message=torch.randn(8,dtype=torch.float64)
 expected=model.consume(message,state,g,a);payload.seek(0);weights,state,gs,als=torch.load(payload,weights_only=False);model.load_state_dict(weights);g.set_state(gs);a.set_state(als);actual=model.consume(message,state,g,a)
 assert torch.equal(expected[0],actual[0]) and identical(expected[1],actual[1]) and expected[3]==actual[3]
