"""Protocol reducer/selection tests; no Torch imports or public-data access."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
import aws_reference_stream_score as scoring
from aws_reference_target_plan import target_windows, REFERENCE_TARGETS


class StreamingProtocol(unittest.TestCase):
    def windows(self, chunk=65537):
        return target_windows(chunk, REFERENCE_TARGETS+1)

    def score(self, window, state):
        if window.reset_sequence:
            self.assertIsNone(state)
        else:
            self.assertEqual(state, (window.batch, window.offset))
        # Exact synthetic per-target loss: target index / 1e6. This exercises
        # unequal last chunks and target weighting without visiting token data.
        loss = sum((begin+end-1)*(end-begin)/2e6 for begin, end in window.target_ranges)
        return loss, window.scored_targets, (window.batch, window.offset+window.count)

    def test_chunk_invariant_exact_weighted_mean(self):
        for chunk in (64, 65537, 262144):
            with self.subTest(chunk=chunk):
                result = scoring.reduce_windows(self.windows(chunk), self.score)
                self.assertEqual(result['targets'], REFERENCE_TARGETS)
                self.assertAlmostEqual(result['nll'], (REFERENCE_TARGETS+1)/2e6, places=9)
                self.assertEqual(result['reset_sequence_groups'], 5)

    def test_silent_target_skip_rejected(self):
        with self.assertRaises(ValueError):
            scoring.reduce_windows(self.windows(), lambda w,s:(1., w.scored_targets-1, None))

    def test_nonfinite_loss_rejected(self):
        with self.assertRaises(ValueError):
            scoring.reduce_windows(self.windows(), lambda w,s:(float('nan'), w.scored_targets, None))

    def test_missing_final_population_rejected(self):
        windows = list(self.windows()); windows.pop()
        with self.assertRaises(ValueError): scoring.reduce_windows(windows, self.score)

    def test_internal_chunk_omission_rejected(self):
        windows = list(self.windows()); del windows[1]
        with self.assertRaises(ValueError): scoring.reduce_windows(windows, self.score)

    def test_sequence_group_omission_rejected(self):
        windows = [w for w in self.windows() if w.batch != 2]
        with self.assertRaises(ValueError): scoring.reduce_windows(windows, self.score)


class Selection(unittest.TestCase):
    def setUp(self):
        self.result = dict(status='completed', args=dict(state_mode='carry', compiled=False, dev_offset=20971520))
        self.selected = dict(step=32, dev_nll=7.)
        self.ready = dict(status='completed', public_scoring_ready=True, gates={k:True for k in
            ('independent_seed','larger_data_quality','complete_fit_work','measured_cpu_inference','exact_history_target_protocol')})

    def test_selected_completed_evidence(self):
        scoring.validate_selection(self.result, self.selected, self.ready)

    def test_initialization_not_benchmark_selection(self):
        self.selected['step'] = 0
        with self.assertRaises(ValueError): scoring.validate_selection(self.result, self.selected, self.ready)

    def test_pending_fit(self):
        self.result['status'] = 'running'
        with self.assertRaises(ValueError): scoring.validate_selection(self.result, self.selected, self.ready)

    def test_incomplete_selection_evidence(self):
        for gate in self.ready['gates']:
            changed = copy.deepcopy(self.ready); changed['gates'][gate] = False
            with self.assertRaises(ValueError): scoring.validate_selection(self.result, self.selected, changed)

    def test_reserved_public_population(self):
        self.result['args']['dev_offset'] = 0
        with self.assertRaises(ValueError): scoring.validate_selection(self.result, self.selected, self.ready)


if __name__ == '__main__': unittest.main()
