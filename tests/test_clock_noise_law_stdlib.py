"""Algebra and pre-import admission; native numerical checks have their own queue."""
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments')]
from sleeping_machines.clock_noise_law import clock_from_unit_noise, factorized_clock_credit, log_normalizer, moments, race
import clock_noise_admission as admission


class ClockLawContracts(unittest.TestCase):
    def test_original_clock_endpoint_and_winners(self):
        scores, noise = [-1., .7, 2.], [.01, 3., 1.]
        original = [e / math.exp(s) for s, e in zip(scores, noise)]
        expected = min(range(3), key=original.__getitem__)
        for temperature in (0., .25, .5, 1.):
            winner, first = race(scores, noise, temperature)
            self.assertEqual(winner, expected)
            if temperature == 1.:
                self.assertAlmostEqual(first, original[winner], places=14)
        self.assertNotEqual(expected, scores.index(max(scores)))

    def test_mean_fixed_and_variance_reduced(self):
        scores = [0., 1., -1.]
        rows = [moments(scores, t) for t in (0., .25, .5, 1.)]
        self.assertEqual(len({r['mean_unbounded_clock'] for r in rows}), 1)
        self.assertEqual(rows[0]['variance_unbounded_clock'], 0.)
        self.assertAlmostEqual(rows[-1]['coefficient_of_variation'], 1.)
        self.assertTrue(all(a['variance_unbounded_clock'] < b['variance_unbounded_clock'] for a, b in zip(rows, rows[1:])))

    def test_deterministic_clock_still_depends_on_all_scores(self):
        scores = [0., 1.]
        a = race(scores, [.1, 9.], 0.)
        b = race(scores, [9., .1], 0.)
        self.assertNotEqual(a[0], b[0])
        self.assertEqual(a[1], b[1])
        c = race([.5, 1.], [.1, 9.], 0.)
        self.assertNotEqual(c[1], a[1])

    def test_common_score_shift_scales_clocks_without_changing_choices(self):
        scores, noise = [-2., .1, 1.], [.3, .7, 1.2]
        for temperature in (0., .5, 1.):
            winner, first = race(scores, noise, temperature)
            shifted, new = race([s + .7 for s in scores], noise, temperature)
            self.assertEqual(winner, shifted)
            self.assertAlmostEqual(new / first, math.exp(-.7))

    def test_factorized_gradient_finite_difference_at_fixed_unit_noise(self):
        scores = [-.2, .7, .4]
        for temperature in (0., .25, .5, 1.):
            first = clock_from_unit_noise(scores, .6, temperature)
            credit = factorized_clock_credit(scores, first)
            self.assertAlmostEqual(sum(credit), -first)
            for i in range(3):
                plus, minus = scores.copy(), scores.copy()
                plus[i] += 1e-6
                minus[i] -= 1e-6
                fd = (clock_from_unit_noise(plus, .6, temperature) - clock_from_unit_noise(minus, .6, temperature)) / 2e-6
                self.assertAlmostEqual(fd, credit[i], places=9)

    def test_gamma_normalization_matches_quantile_integration(self):
        # Deterministic scalar quadrature, no model/runtime/training work.
        scores = [-.4, .8]
        count = 5000
        expected = math.exp(-log_normalizer(scores))
        for temperature in (0., .25, .5, 1.):
            mean = sum(clock_from_unit_noise(scores, -math.log((i + .5) / count), temperature) for i in range(count)) / count
            self.assertLess(abs(mean / expected - 1), .0002)

    def test_bounded_native_delay_mean_is_not_claimed_invariant(self):
        first = [clock_from_unit_noise([0.], w, 1.) for w in (.2, 1.8)]
        transform = lambda t: .001 + .010 * t / (1 + t)
        self.assertLess(sum(map(transform, first)) / 2, transform(clock_from_unit_noise([0.], 1., 0.)))

    def test_invalid_inputs_reject(self):
        for temperature in (-1, 2, float('nan'), True, '0.5'):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                race([0.], [1.], temperature)
        for scores, noise in (([], []), ([0.], []), ([float('nan')], [1.]), ([0.], [0.]), ([0.], [-1.])):
            with self.subTest(scores=scores, noise=noise), self.assertRaises(ValueError):
                race(scores, noise, .5)


class AdmissionContracts(unittest.TestCase):
    def fixture(self, directory):
        path = directory / 'manifest.json'
        cfg = dict(tag='unique_contract', output='results/unique_contract.json')
        record = dict(status='prepared_unrun', source_sha256={}, stages={'contracts': cfg, 'pilot': dict(cfg, tag='pilot', output='results/pilot.json')})
        path.write_text(json.dumps(record))
        return path, record

    def test_autonomous_target_alignment_and_realized_fp32_rounding(self):
        self.assertEqual(admission.forecast_from_increments(10., 1., [2., -3.]), [13., 10.])
        # Every +1 rounds away at this magnitude; base+cumsum would not agree.
        self.assertEqual(admission.forecast_from_increments(1e8, 0., [1.] * 10), [1e8] * 10)

    def test_mutated_manifest_rejects_before_numerical_import(self):
        with tempfile.TemporaryDirectory() as temp:
            path, _ = self.fixture(Path(temp))
            digest = admission.sha(path)
            admission.preflight(path, 'contracts', digest)
            path.write_text(path.read_text() + ' ')
            with self.assertRaisesRegex(ValueError, 'manifest'):
                admission.preflight(path, 'contracts', digest)

    def test_changed_source_and_missing_native_prerequisite_reject(self):
        with tempfile.TemporaryDirectory() as temp:
            path, record = self.fixture(Path(temp))
            record['source_sha256'] = {'sleeping_machines/clock_noise_law.py': 'changed'}
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'source'):
                admission.preflight(path, 'contracts', admission.sha(path))
            record['source_sha256'] = {}
            path.write_text(json.dumps(record))
            with self.assertRaises(FileNotFoundError):
                admission.preflight(path, 'pilot', admission.sha(path))

    def test_container_rejects_before_torch_import(self):
        with patch.object(admission, 'preflight', return_value=({}, {})), \
                patch.object(Path, 'exists', return_value=True), \
                patch.object(sys, 'argv', ['clock_noise', '--manifest', 'unused', '--manifest-sha256', 'unused', '--stage', 'contracts']):
            with self.assertRaisesRegex(ValueError, 'physical-host'):
                admission.main()
        for name in ('torch', 'numpy', 'h5py'):
            self.assertNotIn(name, sys.modules)

    def test_completed_output_or_interrupted_checkpoint_rejects(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            path, record = self.fixture(directory)
            output = directory / record['stages']['contracts']['output']
            output.parent.mkdir()
            with patch.object(admission, 'ROOT', directory):
                output.write_text('{}')
                with self.assertRaisesRegex(ValueError, 'unused'):
                    admission.preflight(path, 'contracts', admission.sha(path))
                output.unlink()
                output.with_suffix('.pt').write_bytes(b'retained checkpoint')
                with self.assertRaisesRegex(ValueError, 'checkpoints'):
                    admission.preflight(path, 'contracts', admission.sha(path))


if __name__ == '__main__':
    unittest.main()
