"""Memory-coupling numerical and actual-fit contracts, one guarded job."""
import gc
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import horizon_token_language_engine as old
import coupled_token_language_engine as new
from sleeping_machines.sparse_counterfactual_episodes import initial_state

def equal(x, y):
    if isinstance(x, torch.Tensor): assert torch.equal(x, y)
    elif isinstance(x, dict):
        assert x.keys() == y.keys()
        for k in x: equal(x[k], y[k])
    elif isinstance(x, (list, tuple)):
        assert len(x) == len(y)
        for a, b in zip(x, y): equal(a, b)
    else: assert x == y

torch.set_num_threads(1)
args = SimpleNamespace(seed=6,payload=4,depth=2,heads=2,pool=3,input_init='balanced',
    tie_pools=False,readout='adaptive',readout_init='frequency',minimum_tail_width=0,
    state_mode='carry',route_credit='sampled',future_site='uniform',free_bias=0.,
    temperature=1.,compiled=False,credit_window=32,unit_gain_scale=1.)
order = torch.arange(50257); counts = torch.ones(50257,dtype=torch.long)
torch.manual_seed(6); a = old.Model(args,order,counts)
torch.manual_seed(6); b = new.Model(args,order,counts)
equal(a.state_dict(),b.state_dict())
ids=torch.arange(64).reshape(2,32)
g1=torch.Generator().manual_seed(7);g2=torch.Generator().manual_seed(7)
x,s,_=a(ids,None,g1,args);y,t,_=b(ids,None,g2,args)
equal(x,y);equal(s,t);equal(g1.get_state(),g2.get_state())
weights=torch.linspace(-1.,1.,x.numel()).reshape_as(x)
(x*weights).sum().backward();(y*weights).sum().backward()
for (_,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):equal(p.grad,q.grad)
contracts=['scale1 all parameters/features/state/RNG/gradient parity']
args.unit_gain_scale=2.;torch.manual_seed(6);m=new.Model(args,order,counts)
args.credit_window=32;g=torch.Generator().manual_seed(7);x,s,_=m(ids,None,g,args);rng=g.get_state()
args.credit_window=8;g=torch.Generator().manual_seed(7);y,t,_=m(ids,None,g,args)
equal(x,y);equal(s,t);equal(rng,g.get_state())
contracts.append('scale2 exact chunk partition/state/RNG parity')
with torch.no_grad():
    probs=m.readout.log_prob(x[:,:2]).exp().sum(-1)
    assert torch.allclose(probs,torch.ones_like(probs),atol=3e-6)
contracts.append('scale2 exact normalized initial output')
args.credit_window=32
state=initial_state(m.core,2)
state['mem']=[torch.randn_like(v).requires_grad_() for v in state['mem']]
state['seen']=[torch.ones_like(v) for v in state['seen']]
x,_,_=m(ids,state,torch.Generator().manual_seed(7),args)
grads=torch.autograd.grad((x*weights).sum(),state['mem'])
assert all(bool(torch.isfinite(v).all()) and bool((v.abs()>0).any()) for v in grads)
contracts.append('scale2 finite nonzero deep addressed-state gradient')
# EOS observation must clear lane0 before processing it, leaving lane1 intact.
seed_state=initial_state(m.core,2)
seed_state['mem']=[torch.randn_like(v) for v in seed_state['mem']]
seed_state['seen']=[torch.ones_like(v) for v in seed_state['seen']]
zero_state=initial_state(m.core,2)
for key in ('mem','arr','seen'):
    zero_state[key]=[torch.stack([z[0],v[1]]) for z,v in zip(zero_state[key],seed_state[key])]
xx,ss,_=m(torch.tensor([[50256],[7]]),seed_state,torch.Generator().manual_seed(8),args)
yy,tt,_=m(torch.tensor([[50256],[7]]),zero_state,torch.Generator().manual_seed(8),args)
equal(xx,yy);equal(ss,tt)
contracts.append('scale2 lane-local EOS reset')
del a,b,m,x,y,s,t,grads;gc.collect()
base=['--readout-init','frequency','--input-init','balanced','--steps','6','--eval-every','3',
      '--train-tokens','512','--dev-tokens','128','--payload','8','--heads','1','--depth','2',
      '--future-site','uniform','--future-score-positions','2','--pool','2','--lanes','2','--eval-lanes','2','--chunk','16','--credit-window','4','--future-every','1']
folder=ROOT/'experiments/results/token_language'
paths=[(old,'curie_coupling_scale1_old_contract_20261005_v1',[]),
       (new,'curie_coupling_scale1_new_contract_20261005_v1',['--unit-gain-scale','1']),
       (new,'curie_coupling_scale2_full_contract_20261005_v1',['--unit-gain-scale','2']),
       (new,'curie_coupling_scale2_partial_contract_20261005_v1',['--unit-gain-scale','2','--stop-after-step','3']),
       (new,'curie_coupling_scale2_resume_contract_20261005_v1',['--unit-gain-scale','2','--resume',str(folder/'curie_coupling_scale2_partial_contract_20261005_v1.pt')])]
for engine,tag,extra in paths:
    sys.argv=['engine','--tag',tag,*base,*extra];engine.main();gc.collect()
loaded=[torch.load(folder/(tag+'.pt'),weights_only=False) for _,tag,_ in paths]
keys=('model','optimizer','state','generator','alternative_generator','local_generator','position_generator','site_generator','torch_rng','cursor','step','total_presentations','writes')
for k in keys:equal(loaded[0][k],loaded[1][k]);equal(loaded[2][k],loaded[4][k])
for ar,br in zip(loaded[0]['curve'],loaded[1]['curve']):
    for k in ('step','train_nll','dev_nll'):equal(ar[k],br[k])
contracts += ['scale1 actual tiny-fit curve/model/optimizer/state/RNG parity',
              'scale2 actual interruption/resume model/optimizer/state/all RNG parity']
try:
    sys.argv=['engine','--tag','curie_coupling_wrong_resume_contract_20261005_v1',*base,
              '--unit-gain-scale','1','--resume',str(folder/'curie_coupling_scale2_partial_contract_20261005_v1.pt')]
    new.main()
except ValueError as error:
    assert 'identical data, source and settings' in str(error)
else:raise AssertionError('Changed gain resumed silently')
contracts.append('changed gain scale rejected at resume')
output=ROOT/'experiments/results/diagnostics/curie_token_coupling_contracts_20261005_v1.json'
if output.exists():raise FileExistsError(output)
output.write_text(json.dumps(dict(status='completed',contracts=contracts,
    scope='Synthetic numerical checks plus five sequential tiny integrated fits; actual uniform-site alternative-write teacher, four internal credit windows per update. Original sources and prior results preserved.'),indent=2)+'\n')
print(json.dumps(dict(status='completed',contracts=contracts)),flush=True)
