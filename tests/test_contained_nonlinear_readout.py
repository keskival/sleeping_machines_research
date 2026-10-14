import unittest
import torch
from sleeping_machines.contained_nonlinear_readout import ContainedNonlinearHead, check_native_readout_containment


class ContainmentTest(unittest.TestCase):
    def test_normalized_adaptive_readout_contract_on_contextual_weights(self):
        from sleeping_machines.frequency_token_readout import FrequencyTokenReadout
        readout=FrequencyTokenReadout(8,torch.arange(1,21),cutoffs=(5,12)).double()
        with torch.no_grad():
            readout.output.head.weight.normal_()
            for tail in readout.output.tail:tail[1].weight.normal_()
        features=torch.arange(64.,dtype=torch.float64).reshape(2,4,8)/20
        targets=torch.tensor([[0,5,12,19],[1,6,13,18]])
        result=check_native_readout_containment(readout,features,targets)
        self.assertEqual(result['status'],'passed')
        self.assertGreater(result['initial_extra_weight_gradient_norm'],0.)

    def test_initial_predictions_cotangents_and_existing_parameters_are_retained(self):
        torch.manual_seed(12)
        base=torch.nn.Linear(4,7,dtype=torch.float64)
        weight,bias=base.weight,base.bias
        state=torch.get_rng_state().clone()
        expanded=ContainedNonlinearHead(base,seed=13)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertIs(expanded.base.weight,weight);self.assertIs(expanded.base.bias,bias)
        x=torch.arange(24.,dtype=torch.float64).reshape(6,4)/10;x.requires_grad_()
        targets=torch.arange(6)
        parent_loss=torch.nn.functional.cross_entropy(base(x),targets)
        child_loss=torch.nn.functional.cross_entropy(expanded(x),targets)
        self.assertTrue(torch.equal(parent_loss,child_loss))
        parent=torch.autograd.grad(parent_loss,(x,weight,bias),retain_graph=True)
        child=torch.autograd.grad(child_loss,(x,weight,bias,expanded.extra.weight,expanded.lift.weight))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(parent,child[:3])))
        self.assertGreater(float(child[3].norm()),0.)
        self.assertEqual(float(child[4].norm()),0.)

    def test_existing_adam_state_remains_attached_to_same_parameter(self):
        base=torch.nn.Linear(4,7,dtype=torch.float64)
        opt=torch.optim.AdamW(base.parameters(),lr=.001)
        base(torch.ones(2,4,dtype=torch.float64)).sum().backward();opt.step();opt.zero_grad()
        weight=base.weight;moment=opt.state[weight]['exp_avg'].clone()
        expanded=ContainedNonlinearHead(base,seed=13)
        opt.add_param_group({'params':expanded.added_parameters()})
        self.assertTrue(torch.equal(opt.state[weight]['exp_avg'],moment))
        self.assertTrue(all(p not in opt.state for p in expanded.added_parameters()))
