import copy,io
from collections import deque
import torch
from experiments.native_query_migration import migrate
from experiments.r1_local_native_deep import LocalNativeDeepLM


def identical(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(identical(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple,deque)):return len(a)==len(b) and all(identical(x,y) for x,y in zip(a,b))
    if hasattr(a,'__dict__'):return type(a)==type(b) and identical(vars(a),vars(b))
    return a==b


def test_migration_preserves_acquisition_and_full_learner_resume():
    torch.manual_seed(16517)
    classes=torch.tensor([0,0,1,1,2,2,3,3]);words=torch.arange(8).reshape(4,2)
    model=LocalNativeDeepLM(8,classes,words,width=8,depth=3,heads=2,pool=3,concept_units=16,topk=2,explore=1,dk=4,packed=True)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    state=model.new_state();g=torch.Generator().manual_seed(15);alt=torch.Generator().manual_seed(16)
    tokens=torch.tensor([1,2,3,4,5,6,7,1,2])
    def update(start,end):
        opt.zero_grad(set_to_none=True)
        nll,obj,_,packets,_=model.stream_nll(tokens[start:end],tokens[start+1:end+1],state,g,alt,stats=True)
        obj.backward();model.deliver_credit(packets,state);opt.step()
        return nll.detach().clone()
    update(0,2);old_parameters=copy.deepcopy(model.state_dict())
    old_moments={n:copy.deepcopy(opt.state[p]) for n,p in model.named_parameters() if p in opt.state}
    model,opt,state,receipt=migrate(model,opt,state,publication_capacity=2)
    assert receipt['parent_events']==2 and state['cause_bank'].count==2
    for n,value in old_parameters.items():assert torch.equal(value,model.state_dict()[n])
    for n,value in old_moments.items():assert identical(value,opt.state[dict(model.named_parameters())[n]])
    update(2,4)
    snapshot=io.BytesIO();torch.save((model.state_dict(),opt.state_dict(),state,g.get_state(),alt.get_state(),torch.get_rng_state()),snapshot)
    expected_loss=update(4,6);expected=(copy.deepcopy(model.state_dict()),copy.deepcopy(opt.state_dict()),copy.deepcopy(state),g.get_state(),alt.get_state(),torch.get_rng_state())
    snapshot.seek(0);weights,moments,state,gs,als,rng=torch.load(snapshot,weights_only=False)
    model.load_state_dict(weights);opt.load_state_dict(moments);g.set_state(gs);alt.set_state(als);torch.set_rng_state(rng)
    actual_loss=update(4,6)
    assert torch.equal(expected_loss,actual_loss)
    assert identical(expected,(model.state_dict(),opt.state_dict(),state,g.get_state(),alt.get_state(),torch.get_rng_state()))
