import unittest
import numpy as np
from sleeping_machines.route_activation_memory import RouteActivationMemory


class RouteActivationMemoryTests(unittest.TestCase):
    def test_matches_explicit_weighted_distribution(self):
        projection = np.array([[1., 2., -1.], [0., 1., .5]])
        memory = RouteActivationMemory(3, projection)
        points = np.array([[1., -2., 3.], [4., 1., -1.], [-1., 2., 5.]])
        weights = np.array([1.])
        for i, (x, a) in enumerate(zip(points, [.7, .8, .3])):
            memory.observe('r', x, float(i), 'v1', a)
            if i: weights = np.r_[a*weights, 1-a]
            expected_mean = weights @ points[:i+1]
            centered = points[:i+1]-expected_mean
            covariance = centered.T @ (weights[:, None]*centered)
            snap = memory.snapshot('r', float(i), 'v1')
            np.testing.assert_allclose(snap.mean, expected_mean, atol=1e-14)
            np.testing.assert_allclose(snap.variance, np.diag(covariance), atol=1e-14)
            np.testing.assert_allclose(snap.projected_covariance,
                                       projection @ covariance @ projection.T, atol=1e-13)
            self.assertAlmostEqual(snap.effective_count, 1/(weights @ weights))

    def test_issued_snapshot_is_causal_and_independent(self):
        memory = RouteActivationMemory(2, np.eye(2))
        memory.observe('r', [1., 2.], 1., 'v1')
        issued = memory.snapshot('r', 1., 'v1')
        memory.observe('r', [5., 6.], 1., 'v2', .5)
        np.testing.assert_array_equal(issued.mean, [1., 2.])
        snap = memory.snapshot('r', 2., 'v3')
        self.assertTrue(snap.operator_stale)
        self.assertEqual(snap.version_changes, 1)
        self.assertEqual(snap.age, 1.)
        snap.mean[:] = 0
        np.testing.assert_array_equal(memory.snapshot('r', 2., 'v2').mean, [3., 4.])
        with self.assertRaises(ValueError): memory.snapshot('r', 0., 'v2')

    def test_invalid_or_excess_observations_do_not_mutate(self):
        memory = RouteActivationMemory(1, np.ones((1, 1)), max_routes=1)
        memory.observe('r', [2.], 2., 'v1')
        for args in [('r', [3.], 1., 'v1'), ('r', [float('nan')], 3., 'v1')]:
            with self.assertRaises(ValueError): memory.observe(*args)
        with self.assertRaises(MemoryError): memory.observe('s', [3.], 3., 'v1')
        self.assertEqual(memory.updates, 1)
        self.assertEqual(memory.snapshot('r', 3., 'v1').observations, 1)


if __name__ == '__main__': unittest.main()
