"""Finite witnesses for the asynchronous-depth synthesis; no model quality claim."""
import math
import unittest

import torch


class CausalAsynchronousDepthTheory(unittest.TestCase):
    def test_irregular_affine_packets_preserve_state_and_parameter_credit(self):
        rate = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
        writes = torch.tensor([.3, -.8, 1.2, .4], dtype=torch.float64, requires_grad=True)
        times = [.02, .17, .19, .91]
        final = 1.31
        state = torch.tensor(.6, dtype=torch.float64)
        previous = 0.
        for time, value in zip(times, writes):
            state = torch.exp(-rate * (time - previous)) * state + value
            previous = time
        sequential = torch.exp(-rate * (final - previous)) * state
        # Independent packet closures need not be periodic or coincide with queries.
        closures = [.23, 1.07]
        packets = []
        for indices, close in zip(([0, 1, 2], [3]), closures):
            packets.append(sum(torch.exp(-rate * (close - times[i])) * writes[i] for i in indices))
        merged = torch.exp(-rate * (closures[1] - closures[0])) * packets[0] + packets[1]
        packed = torch.exp(-rate * final) * .6 + torch.exp(-rate * (final - closures[1])) * merged
        torch.testing.assert_close(sequential, packed, atol=1e-14, rtol=0)
        first = torch.autograd.grad(sequential, (rate, writes), retain_graph=True)
        second = torch.autograd.grad(packed, (rate, writes))
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, atol=1e-14, rtol=0)

    def test_same_packet_endpoint_does_not_preserve_nonlinear_downstream_input(self):
        def outputs(writes):
            state = 0.
            downstream = 0.
            for value in writes:
                state += value
                downstream += state ** 2
            return state, downstream
        self.assertEqual(outputs([1., -1.]), (0., 1.))
        self.assertEqual(outputs([0., 0.]), (0., 0.))

    def test_silence_can_supply_all_information_with_constant_payload(self):
        # Abstract revealed-current-bit policy, not an unrevealed-target decision.
        joint = {(0, 'silence'): .5, (1, 'symbol'): .5}
        information = sum(p * math.log(p / (.5 * .5)) for p in joint.values())
        payload_entropy_given_emit = 0.
        self.assertAlmostEqual(information, math.log(2))
        self.assertEqual(payload_entropy_given_emit, 0.)

    def test_full_support_residual_audit_mean_and_variance(self):
        g = torch.tensor([[2., -1.], [-.5, 3.], [1., .2]], dtype=torch.float64)
        baseline = torch.tensor([[1.8, -.9], [-.4, 2.7], [.8, .3]], dtype=torch.float64)
        probabilities = torch.tensor([.2, .3, .5], dtype=torch.float64)
        residual = g - baseline
        estimates = baseline.sum(0) + residual / probabilities[:, None]
        mean = (probabilities[:, None] * estimates).sum(0)
        torch.testing.assert_close(mean, g.sum(0), atol=1e-14, rtol=0)
        variance = (probabilities * ((estimates - mean) ** 2).sum(1)).sum()
        formula = ((residual ** 2).sum(1) / probabilities).sum() - (residual.sum(0) ** 2).sum()
        torch.testing.assert_close(variance, formula, atol=1e-14, rtol=0)
        oracle = g.sum(0) + (g - g) / probabilities[:, None]
        torch.testing.assert_close(oracle, g.sum(0).expand_as(g))

    def test_unordered_shared_nonlinear_writes_need_an_order(self):
        # Two updates to one receiver are causally unordered but do not commute.
        first = lambda x: x + 1.
        second = lambda x: 2. * x
        self.assertNotEqual(first(second(0.)), second(first(0.)))
        # Independent additive corrections at a common reference do commute.
        self.assertAlmostEqual((.5 + .3) - .2, (.5 - .2) + .3)


if __name__ == '__main__':
    unittest.main()
