import unittest
import torch
from torch import nn
from sleeping_machines.packed_token_core import PackedTokenCore
from sleeping_machines.learning.receiver_bank_adamw import ReceiverBankAdamW
from sleeping_machines.learning.stage_gain_learner import migrate_stage_gain_learner
from sleeping_machines.sparse_counterfactual_episodes import token_features,detach
from tests.test_stage_gain_token_core import model


class Actor(nn.Module):
    def __init__(self):
        super().__init__();self.core=PackedTokenCore(model((.35,.35)))
        self.readout=nn.Linear(8,7).double()
        self.generator=torch.Generator().manual_seed(11)


def update(actor,opt,state):
    x,state,_=token_features(actor.core,torch.tensor([[0,1,2],[3,4,5]]),state,
        generator=actor.generator,route_credit='none')
    loss=actor.readout(x).square().mean();loss.backward()
    if isinstance(opt,torch.optim.AdamW):torch.nn.utils.clip_grad_norm_(actor.parameters(),1.)
    opt.step();opt.zero_grad(set_to_none=True)
    return float(loss.detach()),detach(state)


def equal(test,a,b):
    if isinstance(a,torch.Tensor):test.assertTrue(torch.equal(a,b))
    elif isinstance(a,dict):
        test.assertEqual(a.keys(),b.keys())
        for key in a:equal(test,a[key],b[key])
    elif isinstance(a,(list,tuple)):
        test.assertEqual(len(a),len(b))
        for x,y in zip(a,b):equal(test,x,y)
    else:test.assertEqual(a,b)


class LearnerMigrationTest(unittest.TestCase):
    def test_next_native_update_moments_versions_state_and_rng_are_preserved(self):
        for law in ('global','banked','local'):
            actor=Actor()
            opt=(torch.optim.AdamW(actor.parameters(),lr=.003,amsgrad=True) if law=='global' else
                ReceiverBankAdamW(actor,lr=.003,amsgrad=True,local_clip=.5 if law=='local' else None))
            _,state=update(actor,opt,None)
            before=torch.get_rng_state().clone()
            new,other,carried=migrate_stage_gain_learner(actor,opt,state)
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            equal(self,opt.state_dict(),other.state_dict());equal(self,state,carried)
            for a,b in zip(actor.parameters(),new.parameters()):self.assertNotEqual(a.data_ptr(),b.data_ptr())
            self.assertNotEqual(state['mem'][0].data_ptr(),carried['mem'][0].data_ptr())
            loss,state=update(actor,opt,state);next_loss,next_state=update(new,other,carried)
            self.assertEqual(loss,next_loss);equal(self,state,next_state)
            equal(self,opt.state_dict(),other.state_dict())
            for a,b in zip(actor.parameters(),new.parameters()):self.assertTrue(torch.equal(a,b))
            self.assertTrue(torch.equal(actor.generator.get_state(),new.generator.get_state()))

    def test_pending_gradients_and_live_state_are_rejected(self):
        actor=Actor();opt=torch.optim.AdamW(actor.parameters())
        next(actor.parameters()).grad=torch.ones_like(next(actor.parameters()))
        with self.assertRaises(ValueError):migrate_stage_gain_learner(actor,opt)
        opt.zero_grad(set_to_none=True)
        with self.assertRaises(ValueError):migrate_stage_gain_learner(actor,opt,{'live':torch.ones(1,requires_grad=True)})
