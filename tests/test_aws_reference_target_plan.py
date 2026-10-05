import dataclasses
import importlib.util
from pathlib import Path
import sys
import unittest

spec = importlib.util.spec_from_file_location('reference_target_plan', Path(__file__).resolve().parents[1]/'experiments/aws_reference_target_plan.py')
plan = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plan
spec.loader.exec_module(plan)


class TargetProtocol(unittest.TestCase):
    def windows(self, chunk=65537):
        return list(plan.target_windows(chunk, plan.REFERENCE_TARGETS+1))

    def test_exact_denominator_across_chunks(self):
        for chunk in (16, 64, 65537, 262144, 300000):
            with self.subTest(chunk=chunk):
                result = plan.validate_plan(plan.target_windows(chunk, plan.REFERENCE_TARGETS+1))
                self.assertEqual(result['targets'], 10485760)
                self.assertEqual(result['reset_sequences'], 40)

    def test_final_lookahead_required(self):
        with self.assertRaises(ValueError): list(plan.target_windows(64, plan.REFERENCE_TARGETS))

    def test_final_target_included(self):
        windows = self.windows()
        self.assertEqual(windows[-1].target_ranges[-1][-1], 10485761)
        self.assertEqual(windows[-1].lane_load_ranges[-1][-1], 10485761)

    def test_chunk_lookahead_and_continuation(self):
        first, second = self.windows()[:2]
        for lane in range(8):
            self.assertEqual(first.target_ranges[lane][-1]-1, second.input_starts[lane])
            self.assertEqual(first.target_ranges[lane][-1], second.target_starts[lane])
        self.assertTrue(first.reset_sequence)
        self.assertFalse(second.reset_sequence)

    def test_missing_window_rejected(self):
        windows = self.windows(); del windows[1]
        with self.assertRaises(ValueError): plan.validate_plan(windows)

    def test_duplicate_window_rejected(self):
        windows = self.windows(); windows.insert(1, windows[0])
        with self.assertRaises(ValueError): plan.validate_plan(windows)

    def test_reset_inside_sequence_rejected(self):
        windows = self.windows(); windows[1] = dataclasses.replace(windows[1], reset_sequence=True)
        with self.assertRaises(ValueError): plan.validate_plan(windows)

    def test_boundary_crossing_rejected(self):
        windows = self.windows(); windows[0] = dataclasses.replace(windows[0], count=262145)
        with self.assertRaises(ValueError): plan.validate_plan(windows)

    def test_invalid_chunk(self):
        for chunk in (0, -1, True, 1.5):
            with self.assertRaises(ValueError): list(plan.target_windows(chunk, plan.REFERENCE_TARGETS+1))


if __name__ == '__main__': unittest.main()
