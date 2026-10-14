import unittest
import torch
from sleeping_machines.acquisition_budget_extension import extend_acquisition_checkpoint


class ExtensionTest(unittest.TestCase):
    def fixture(self):
        ck=dict(settings={'steps':64,'exposure':'fresh','lr':.003},step=64,
                identity={'data':'pinned'},model={'weight':torch.tensor([1.,2.])},
                optimizer={'moments':torch.tensor([.1,.2])},state={'memory':torch.tensor([3.])},
                cursor=63,training_block=0,rng=torch.tensor([7,8]),curve=[{'step':64,'nll':5.}])
        packet=dict(status='completed',args=dict(ck['settings'],tag='parent',resume=None,stop_after_step=None),identity={'data':'pinned'})
        return ck,packet

    def test_only_endpoint_changes_and_parent_is_independent(self):
        ck,packet=self.fixture();child=extend_acquisition_checkpoint(ck,packet,128)
        self.assertEqual(ck['settings']['steps'],64);self.assertEqual(child['settings']['steps'],128)
        self.assertTrue(torch.equal(child['optimizer']['moments'],ck['optimizer']['moments']))
        self.assertTrue(torch.equal(child['rng'],ck['rng']))
        child['model']['weight'][0]=99
        self.assertEqual(float(ck['model']['weight'][0]),1.)

    def test_incomplete_or_changed_identity_cannot_be_migrated(self):
        ck,packet=self.fixture();packet['status']='checkpointed'
        with self.assertRaises(ValueError):extend_acquisition_checkpoint(ck,packet,128)
        packet['status']='completed';packet['identity']={'data':'different'}
        with self.assertRaises(ValueError):extend_acquisition_checkpoint(ck,packet,128)

    def test_no_budget_reduction_or_repeat_data_arm(self):
        ck,packet=self.fixture()
        with self.assertRaises(ValueError):extend_acquisition_checkpoint(ck,packet,64)
        ck['settings']['exposure']='repeated'
        with self.assertRaises(ValueError):extend_acquisition_checkpoint(ck,packet,128)

    def test_changed_learning_settings_in_packet_are_rejected(self):
        ck,packet=self.fixture();packet['args']['lr']=.03
        with self.assertRaises(ValueError):extend_acquisition_checkpoint(ck,packet,128)

    def test_longer_data_plan_retains_prefix_and_admits_next_fresh_block(self):
        import sys
        from pathlib import Path
        import numpy as np
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
        from allocated_depth_token_language import BlockLaneTokens
        data=np.arange(256,dtype=np.uint16)
        parent=BlockLaneTokens(data,32,2,'fresh',16,4)
        child=BlockLaneTokens(data,32,2,'fresh',32,4)
        self.assertTrue(torch.equal(parent.counts(),child.counts()))
        for block in range(parent.blocks):
            parent.block=child.block=block
            self.assertTrue(torch.equal(parent[:,0:16],child[:,0:16]))
        child.advance()
        self.assertEqual(int(child[:,0:2][0,0]),parent.admitted)
        self.assertEqual(child.admitted,256)
