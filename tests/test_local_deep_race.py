import copy
import io
import torch
from sleeping_machines.local_deep_race import LocalDeepRace


def seeds():return torch.Generator().manual_seed(413),torch.Generator().manual_seed(414)


def test_eight_native_layers_receive_local_credit_without_temporal_graph():
    torch.manual_seed(412);model=LocalDeepRace(width=8,depth=8,heads=2,pool=3)
    decoder=torch.nn.Linear(8,3);state=model.new_state();g,a=seeds();messages=torch.randn(5,8)
    for message in messages:
        model.zero_grad(set_to_none=True);decoder.zero_grad(set_to_none=True)
        out,state,packets,_=model.consume(message,state,g,a)
        loss=torch.nn.functional.cross_entropy(decoder(out),torch.ones(8,dtype=torch.long));loss.backward()
        model.deliver_credit(packets,state)
        for layer in model.units:
            assert sum(float(p.grad.square().sum()) for p in layer.parameters() if p.grad is not None)>0
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        assert all(v.value.grad_fn is None and v.control_weight.grad_fn is None for v in state['receivers'].values())
        del out,packets,loss
    assert state['selected_updates']==5*8*2
    assert state['alternative_values']==state['selected_updates']
    assert model.storage(state)['temporal_graph_entries']==0


def test_chunk_partition_and_serialized_resume_preserve_native_events():
    torch.manual_seed(418);model=LocalDeepRace(width=8,depth=3,heads=2,pool=4)
    messages=torch.randn(7,8);g,a=seeds();state=model.new_state();first=[]
    with torch.no_grad():
        for message in messages[:3]:
            out,state,_,_=model.consume(message,state,g,a,learn=False);first.append(out)
        buffer=io.BytesIO();torch.save(dict(state=state,g=g.get_state(),a=a.get_state()),buffer)
        tail=[]
        for message in messages[3:]:
            out,state,_,_=model.consume(message,state,g,a,learn=False);tail.append(out)
        buffer.seek(0);saved=torch.load(buffer,weights_only=False);g.set_state(saved['g']);a.set_state(saved['a']);state=saved['state'];again=[]
        for message in messages[3:]:
            out,state,_,_=model.consume(message,state,g,a,learn=False);again.append(out)
    assert torch.equal(torch.stack(tail),torch.stack(again))


def test_later_inputs_cannot_change_earlier_outputs():
    torch.manual_seed(420);model=LocalDeepRace(width=8,depth=3,heads=2,pool=4);messages=torch.randn(6,8)
    def run(x):
        state=model.new_state();g,a=seeds();out=[]
        with torch.no_grad():
            for m in x:
                y,state,_,_=model.consume(m,state,g,a,learn=False);out.append(y)
        return torch.stack(out)
    changed=messages.clone();changed[4:]+=100
    assert torch.equal(run(messages)[:4],run(changed)[:4])
