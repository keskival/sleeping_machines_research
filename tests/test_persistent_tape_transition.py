import copy,io
import numpy as np
import pytest,torch
from experiments.credit.persistent_tape_transition import transition_batch
from experiments.r1_local_native_deep import LocalNativeDeepLM
from tests.test_native_query_migration import identical

def test_boundary_scores_every_new_target_once_without_state_mutation():
    tape=np.array([2,3,4,5]);state={'expected_next_input':7,'memory':[11,12]};old=copy.deepcopy(state)
    assert transition_batch(tape,0,2,state,tape_start=0).tolist()==[7,2,3]
    assert state==old
    state['expected_next_input']=3
    assert transition_batch(tape,2,2,state,tape_start=0).tolist()==[3,4,5]
    state['expected_next_input']=1
    with pytest.raises(ValueError):transition_batch(tape,2,2,state,tape_start=0)

def test_native_training_crosses_tape_boundary_and_resumes_exactly():
    torch.manual_seed(17017)
    model=LocalNativeDeepLM(8,torch.tensor([0,0,1,1,2,2,3,3]),torch.arange(8).reshape(4,2),width=8,depth=3,heads=2,pool=3,concept_units=16,topk=2,explore=1,dk=4,packed=True)
    opt=torch.optim.AdamW(model.parameters(),lr=.001);state=model.new_state()
    g=torch.Generator().manual_seed(17);alt=torch.Generator().manual_seed(18)
    def step(w):
        opt.zero_grad(set_to_none=True);nll,obj,_,packets,_=model.stream_nll(w[:-1],w[1:],state,g,alt,stats=False)
        obj.backward();model.deliver_credit(packets,state);opt.step();return nll.detach().clone()
    step(torch.tensor([1,5,7]));events=state['events'];bank=state['cause_bank'].count
    saved=io.BytesIO();torch.save((model.state_dict(),opt.state_dict(),state,g.get_state(),alt.get_state(),torch.get_rng_state()),saved)
    tape=np.array([2,3,4,6]);expected=step(transition_batch(tape,0,2,state,tape_start=0))
    assert state['events']==events+2 and state['cause_bank'].count==bank+2
    outcome=copy.deepcopy((model.state_dict(),opt.state_dict(),state,g.get_state(),alt.get_state(),torch.get_rng_state()))
    saved.seek(0);weights,moments,state,gs,als,rng=torch.load(saved,weights_only=False)
    model.load_state_dict(weights);opt.load_state_dict(moments);g.set_state(gs);alt.set_state(als);torch.set_rng_state(rng)
    actual=step(transition_batch(tape,0,2,state,tape_start=0))
    assert torch.equal(actual,expected)
    assert identical(outcome,(model.state_dict(),opt.state_dict(),state,g.get_state(),alt.get_state(),torch.get_rng_state()))
    step(transition_batch(tape,2,2,state,tape_start=0));assert state['events']==events+4
