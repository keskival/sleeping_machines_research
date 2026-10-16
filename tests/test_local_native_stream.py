import io
import numpy as np
import torch
from experiments.r1_local_native_deep import LocalNativeDeepLM,word_classes


def setup():
    torch.manual_seed(721);c,w=word_classes(np.arange(1,18),4)
    model=LocalNativeDeepLM(17,c,w,width=8,depth=3,heads=2,pool=3,concept_units=16,topk=2,
        explore=1,dk=4,packed=True,bank_capacity=16,bank_candidates=8)
    return model,torch.tensor([1,3,1,4,1,3,2,1,3])


def generators():return torch.Generator().manual_seed(81),torch.Generator().manual_seed(82)


def test_persistent_native_stream_is_credit_chunk_partition_invariant():
    model,tokens=setup();model.eval();g,a=generators()
    with torch.no_grad():
        full,_,fs,_,_=model.stream_nll(tokens[:-1],tokens[1:],model.new_state(),g,a,learn=False)
        g,a=generators();s=model.new_state();pieces=[]
        for start,end in [(0,3),(3,5),(5,8)]:
            nll,_,s,_,_=model.stream_nll(tokens[start:end],tokens[start+1:end+1],s,g,a,learn=False);pieces.append(nll)
    assert torch.equal(full,torch.cat(pieces))
    assert s['cause_bank'].count==8 and torch.equal(s['cause_inbox'],fs['cause_inbox'])
    assert s['cause_inbox'].grad_fn is None and s['cause_bank'].features.grad_fn is None


def test_future_observations_cannot_change_earlier_scored_probabilities():
    model,tokens=setup();model.eval();changed=tokens.clone();changed[5:]=torch.tensor([8,9,10,11])
    def run(t):
        g,a=generators()
        with torch.no_grad():return model.stream_nll(t[:-1],t[1:],model.new_state(),g,a,learn=False)[0]
    assert torch.equal(run(tokens)[:4],run(changed)[:4])


def test_serialized_bank_inbox_receivers_and_rng_continue_exactly():
    model,tokens=setup();model.eval();g,a=generators();state=model.new_state()
    with torch.no_grad():
        _,_,state,_,_=model.stream_nll(tokens[:4],tokens[1:5],state,g,a,learn=False)
        buf=io.BytesIO();torch.save(dict(state=state,g=g.get_state(),a=a.get_state()),buf)
        expected=model.stream_nll(tokens[4:-1],tokens[5:],state,g,a,learn=False)[0]
        buf.seek(0);saved=torch.load(buf,weights_only=False);g.set_state(saved['g']);a.set_state(saved['a'])
        actual=model.stream_nll(tokens[4:-1],tokens[5:],saved['state'],g,a,learn=False)[0]
    assert torch.equal(expected,actual)


def test_native_stream_learns_locally_across_an_optimizer_boundary():
    model,tokens=setup();model.train();opt=torch.optim.AdamW(model.parameters(),lr=.001)
    state=model.new_state();g,a=generators()
    for start,end in [(0,4),(4,8)]:
        opt.zero_grad(set_to_none=True)
        nll,obj,state,packets,report=model.stream_nll(tokens[start:end],tokens[start+1:end+1],state,g,a,stats=True)
        obj.backward();model.deliver_credit(packets,state)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step()
    assert state['cause_bank'].count==8 and state['events']==8
    assert sum(report['depth_responsibility'])>.99999
    assert state['cause_inbox'].grad_fn is None
