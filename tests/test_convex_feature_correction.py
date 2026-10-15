import copy
import unittest
import torch
from sleeping_machines.token_readout import TokenReadout
from sleeping_machines.convex_feature_correction import fit_feature_correction


class ConvexCorrectionTest(unittest.TestCase):
    def test_dense_and_adaptive_fit_preserve_readout_and_certify_zero_comparator(self):
        for cutoffs in ((),(2,4)):
            torch.manual_seed(8)
            readout=TokenReadout(8,7,cutoffs).double()
            saved=copy.deepcopy(readout.state_dict());rng=torch.get_rng_state().clone()
            parent=torch.zeros(12,8,dtype=torch.float64)
            branch=torch.eye(6,dtype=torch.float64).repeat(2,1)
            labels=torch.arange(6).repeat(2)
            result=fit_feature_correction(readout,parent,branch,labels,iterations=15)
            self.assertLess(result['fitted_nll'],result['baseline_nll']-.01)
            self.assertLessEqual(result['regularized_objective'],result['baseline_nll']+1e-10)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            for name,value in saved.items():self.assertTrue(torch.equal(value,readout.state_dict()[name]))
            self.assertTrue(all(p.grad is None for p in readout.parameters()))

    def test_zero_features_and_invalid_inputs(self):
        readout=TokenReadout(4,3).double()
        p=torch.zeros(3,4,dtype=torch.float64);b=torch.zeros(3,2,dtype=torch.float64);y=torch.arange(3)
        r=fit_feature_correction(readout,p,b,y,iterations=2)
        self.assertEqual(r['baseline_nll'],r['fitted_nll'])
        self.assertEqual(int(torch.count_nonzero(r['weight'])),0)
        with self.assertRaises(ValueError):fit_feature_correction(readout,p,b,y,l2=0.)

if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
