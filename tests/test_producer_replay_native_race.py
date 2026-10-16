import math
import torch
from torch.nn import functional as F
from sleeping_machines.queryable_local_deep_race import QueryableLocalDeepRace
from sleeping_machines.packed_queryable_local_deep_race import PackedQueryableLocalDeepRace
from sleeping_machines.producer_replay_native_race import ProducerReplayNativeRace,local_write


def setup():
 torch.manual_seed(16601);reference=QueryableLocalDeepRace(width=8,depth=3,heads=2,pool=3,publication_capacity=3).double()
 return PackedQueryableLocalDeepRace(reference),ProducerReplayNativeRace(reference)


def test_forward_is_unchanged_by_local_producer_credit():
 base,model=setup();bs=base.new_state();state=model.new_state();bg=torch.Generator().manual_seed(71);g=torch.Generator().manual_seed(71);ba=torch.Generator().manual_seed(72);a=torch.Generator().manual_seed(72)
 for message in torch.randn(5,8,dtype=torch.float64):
  x,bs,_,_=base.consume(message,bs,bg,ba);y,state,_,_=model.consume(message,state,g,a)
  assert torch.equal(x,y)
 assert state['producer_replay_reads']>0
 assert all(r.prior_value.grad_fn is None and r.base_input.grad_fn is None for r in state['producer_records'].values())


def test_later_read_teaches_original_joint_producer_and_matches_local_write_oracle():
 _,model=setup();state=model.new_state();g=torch.Generator().manual_seed(71);a=torch.Generator().manual_seed(72)
 for message in torch.randn(3,8,dtype=torch.float64):model.consume(message,state,g,a)
 key=state['publications'][1][-1];entry=state['bindings'].entries[key];producer=state['producer_records'][key]
 now=entry.available+1.;weight=torch.arange(1,9,dtype=torch.float64)
 model.zero_grad(set_to_none=True);packets=[]
 actual=model.read_publication(state,key,now,packets,learn=True)
 (actual*weight).sum().backward()
 gradients={name:p.grad.detach().clone() for name,p in model.named_parameters() if p.grad is not None}
 assert gradients['assemblers.0.pair_weight'].norm()>0
 # An independent bounded local graph: no stored state graph or future label.
 model.zero_grad(set_to_none=True);d=key[0]
 stream=producer.base_input+producer.inbox+model.assemblers[d-1](producer.views)
 mixed=F.linear(stream,model.mix[d]).reshape(model.heads,model.payload)
 memory=local_write(producer.prior_value,producer.write_parameters,mixed,producer.elapsed)
 hi=torch.arange(model.heads);winners=torch.tensor(entry.native.winners)
 params={name:bank[d,hi,winners] for name,bank in model.banks.items()}
 values=model.unit_value(params,mixed,memory);age=now-entry.native.head_times
 mag=torch.exp(-age[:,None]*(F.softplus(model.transport_rate[d])+1e-6));angle=age[:,None]*model.transport_frequency[d]
 pairs=values.reshape(model.heads,-1,2);real,imag=pairs[...,0]*mag,pairs[...,1]*mag
 aligned=torch.stack((angle.cos()*real-angle.sin()*imag,angle.sin()*real+angle.cos()*imag),-1).flatten()
 expected=F.layer_norm(stream+aligned/math.sqrt(model.depth),(model.width,))
 assert torch.allclose(actual,expected,atol=2e-11,rtol=2e-11)
 (expected*weight).sum().backward()
 for name,p in model.named_parameters():
  if name=='mix' or name.startswith('assemblers.0.'):
   assert p.grad is not None and torch.allclose(p.grad,gradients[name],atol=3e-10,rtol=3e-10),name
