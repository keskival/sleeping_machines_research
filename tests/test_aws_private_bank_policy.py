import math
import unittest
from experiments.aws_private_bank_policy import replica_policy, grouped_initial_rates


class BankPolicyTest(unittest.TestCase):
    def test_fixed_activity_capacity(self):
        p = replica_policy(4,16)
        self.assertEqual(p['replicas'],4)
        self.assertEqual(p['clock_bias_shift'],-math.log(4))
        self.assertEqual(len(p['shared_fields']),6)
        self.assertEqual(len(p['private_fields']),5)

    def test_group_hazard_is_preserved(self):
        scores = [-3.,-.2,1.,4.]
        rates = grouped_initial_rates(scores,replica_policy(4,16))
        for s,r in zip(scores,rates): self.assertAlmostEqual(r,math.exp(s),places=12)
        self.assertAlmostEqual(sum(rates),sum(map(math.exp,scores)),places=12)

    def test_no_growth_identity(self):
        p=replica_policy(4,4)
        self.assertEqual(p['clock_bias_shift'],0)
        self.assertEqual(grouped_initial_rates([0]*4,p),[1]*4)

    def test_invalid_sizes(self):
        for c,t in [(0,16),(4,3),(4,10),(True,4),(4,16.)]:
            with self.assertRaises(ValueError): replica_policy(c,t)

    def test_clipped_law_rejected(self):
        for score in [-11.,12.,float('nan')]:
            with self.assertRaises(ValueError):
                grouped_initial_rates([score]*4,replica_policy(4,16))


if __name__=='__main__': unittest.main()
