from types import SimpleNamespace
import torch
from horizon_token_language_engine import Model
from sleeping_machines.sparse_counterfactual_episodes import initial_state


def setup():
    a = SimpleNamespace(seed=6,payload=4,depth=2,heads=2,pool=3,input_init='balanced',
        tie_pools=False,readout='adaptive',readout_init='frequency',minimum_tail_width=0,
        state_mode='carry',route_credit='sampled',future_site='uniform',free_bias=0.,
        temperature=1.,compiled=False,credit_window=32)
    torch.manual_seed(6)
    m=Model(a,torch.arange(50257),torch.ones(50257,dtype=torch.long))
    return m,a


def test_credit_partition_preserves_every_feature_numeric_state_and_race_rng():
    m,a=setup();ids=torch.arange(64).reshape(2,32)
    g=torch.Generator().manual_seed(7)
    x,s,_=m(ids,None,g,a);rng=g.get_state()
    a.credit_window=8;g=torch.Generator().manual_seed(7)
    y,t,_=m(ids,None,g,a)
    assert torch.equal(x,y)
    assert torch.equal(rng,g.get_state())
    for key in ('mem','arr','seen'):
        assert all(torch.equal(v,w) for v,w in zip(s[key],t[key]))
    for key in ('position','ctx_vals','ctx_arr','last_writes','race_winners'):
        assert torch.equal(s[key],t[key]),key


def test_short_window_cuts_initial_memory_credit_but_full_window_retains_it():
    m,a=setup();ids=torch.arange(64).reshape(2,32)
    grads=[]
    for window in (8,32):
        a.credit_window=window
        state=initial_state(m.core,2)
        state['mem']=[torch.randn_like(v).requires_grad_() for v in state['mem']]
        state['seen']=[torch.ones_like(v) for v in state['seen']]
        x,_,_=m(ids,state,torch.Generator().manual_seed(7),a)
        grads.append(torch.autograd.grad(x[:,-1].square().sum(),state['mem'][0],allow_unused=True)[0])
    assert grads[0] is None or not bool((grads[0] != 0).any())
    assert grads[1] is not None and bool((grads[1].abs()>0).any())
