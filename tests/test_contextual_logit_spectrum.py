import unittest
import torch
from sleeping_machines.contextual_logit_spectrum import centered_logit_spectrum


class LogitSpectrumTest(unittest.TestCase):
    def test_factor_spectrum_matches_dense_double_centered_logits(self):
        gen=torch.Generator().manual_seed(9)
        for contexts,classes,width in ((12,17,4),(3,5,8)):
            x=torch.randn(contexts,width,generator=gen,dtype=torch.float64)
            w=torch.randn(classes,width,generator=gen,dtype=torch.float64)
            bias=torch.randn(classes,generator=gen,dtype=torch.float64)
            logits=x@w.T+bias
            # Softmax subtracts a per-context scalar; neither changes the spectrum.
            logp=logits.log_softmax(-1)
            centered=logp-logp.mean(0,keepdim=True)-logp.mean(1,keepdim=True)+logp.mean()
            direct=torch.linalg.svdvals(centered)
            result=centered_logit_spectrum(x,w)
            factored=torch.tensor(result['singular_values'],dtype=torch.float64)
            self.assertTrue(torch.allclose(factored,direct[:len(factored)],atol=1e-12,rtol=1e-12))
            self.assertTrue(torch.all(direct[len(factored):]<1e-12))

    def test_constant_contexts_have_zero_energy_and_rank(self):
        result=centered_logit_spectrum(torch.ones(8,3),torch.arange(12.).reshape(4,3))
        self.assertEqual(result['squared_logit_energy'],0.)
        self.assertEqual(result['energy_effective_rank'],0.)

    def test_invalid_coordinates_are_rejected(self):
        with self.assertRaises(ValueError):centered_logit_spectrum(torch.ones(8,3),torch.ones(4,2))
