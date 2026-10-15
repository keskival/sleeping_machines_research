import io
import numpy as np
import torch
from sleeping_machines.relational_frame_composition import RelationalFrameComposition
from experiments.hierarchy.paired_composition_learning import paired_backward


class AnonymousScene:
    def __init__(self,events,label):self.events,self.label=events,label
    def model_input(self):return self.events.copy(),self.label


def make():
    torch.manual_seed(101)
    return RelationalFrameComposition(input_width=2,payload=4,heads=2,levels=8,depth=4,classes=2)


def examples():
    a=np.array([[0,0,0,1],[10000,100,100,-1]],dtype=np.int64)
    b=a.copy();b[:,2]=100-b[:,2]
    return [(AnonymousScene(a,0),0),(AnonymousScene(b,1),1)]


def step(model,opt):
    model.train();opt.zero_grad(set_to_none=True)
    loss,_=paired_backward(model,examples())
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
    return loss


def test_exact_resume_reproduces_paired_native_update_and_rng():
    a=make();oa=torch.optim.AdamW(a.parameters(),lr=.001)
    torch.manual_seed(103);first=step(a,oa)
    saved=dict(model=a.state_dict(),optimizer=oa.state_dict(),rng=torch.get_rng_state())
    buffer=io.BytesIO();torch.save(saved,buffer);buffer.seek(0)
    second=step(a,oa);expected_rng=torch.get_rng_state()
    b=make();ob=torch.optim.AdamW(b.parameters(),lr=.001)
    loaded=torch.load(buffer,weights_only=False)
    b.load_state_dict(loaded['model']);ob.load_state_dict(loaded['optimizer']);torch.set_rng_state(loaded['rng'])
    assert step(b,ob)==second
    assert torch.equal(torch.get_rng_state(),expected_rng)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())


def test_opposite_label_pair_cancels_bias_gradient_at_equal_zero_logits():
    # An algebraic witness for the pipeline change, not a model-quality claim.
    x=torch.tensor([[1.,2.],[-1.,3.]],requires_grad=True)
    weight=torch.zeros(2,requires_grad=True);bias=torch.tensor(0.,requires_grad=True)
    z=x@weight+bias;y=torch.tensor([0.,1.])
    loss=torch.nn.functional.binary_cross_entropy_with_logits(z,y);loss.backward()
    assert bias.grad==0
    assert torch.equal(weight.grad,(x.detach()[0]-x.detach()[1])/4)
