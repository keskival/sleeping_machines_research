import unittest
import numpy as np
import torch
from experiments.common_prior_token_language import BlockLaneTokens, learner_fingerprint


class CommonPriorTests(unittest.TestCase):
    def test_only_later_block_changes_and_prior_never_reads_it(self):
        data=np.arange(32,dtype=np.uint16)
        repeated=BlockLaneTokens(data,16,2,'repeated',2,64)
        fresh=BlockLaneTokens(data,16,2,'fresh',2,64)
        index=(slice(None),slice(0,2))
        self.assertTrue(torch.equal(repeated[index],fresh[index]))
        prior=fresh.counts().clone()
        repeated.advance();fresh.advance()
        self.assertEqual(repeated[index].tolist(),[[0,1],[8,9]])
        self.assertEqual(fresh[index].tolist(),[[16,17],[24,25]])
        self.assertTrue(torch.equal(prior,fresh.counts()))
        self.assertEqual(prior[16:].sum().item(),0)
        with self.assertRaises(ValueError):fresh.advance()
        self.assertEqual(fresh.block,1)

    def test_fingerprint_binds_numeric_learner_state(self):
        first={'model':torch.tensor([1.,2.]),'rng':torch.tensor([3,4])}
        self.assertEqual(learner_fingerprint(first),learner_fingerprint(dict(first)))
        second={'model':torch.tensor([1.,2.1]),'rng':first['rng']}
        self.assertNotEqual(learner_fingerprint(first),learner_fingerprint(second))


if __name__=='__main__':unittest.main()
