import copy
import unittest
from experiments.aws_private_bank_admission import validate_contract


class AdmissionTest(unittest.TestCase):
    def setUp(self):
        self.pins={'contract':'abc'}
        self.record=dict(status='completed',source_sha256=self.pins,
            construction_rng_exact=True,canonical_parameter_parity=True,shared_rule_parity=True,
            private_replica_storage=True,cross_recipe_checkpoint_rejected=True,
            factual_gradient_finite=True,rows=[dict(pool=u,selected_writes=32) for u in (4,16)])
    def test_complete_passes(self):self.assertTrue(validate_contract(self.record,self.pins))
    def test_each_numerical_contract_required(self):
        for key in ('construction_rng_exact','canonical_parameter_parity','shared_rule_parity',
                    'private_replica_storage','cross_recipe_checkpoint_rejected','factual_gradient_finite'):
            r=copy.deepcopy(self.record);r[key]=False
            with self.assertRaises(ValueError):validate_contract(r,self.pins)
    def test_pending_rejected(self):
        self.record['status']='running'
        with self.assertRaises(ValueError):validate_contract(self.record,self.pins)
    def test_changed_source_rejected(self):
        with self.assertRaises(ValueError):validate_contract(self.record,{'contract':'def'})
    def test_missing_arm_rejected(self):
        self.record['rows'].pop()
        with self.assertRaises(ValueError):validate_contract(self.record,self.pins)
    def test_extra_activity_rejected(self):
        self.record['rows'][1]['selected_writes']=128
        with self.assertRaises(ValueError):validate_contract(self.record,self.pins)
