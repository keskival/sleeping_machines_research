"""Verify flattened loss layout and unchanged scorer tape/accumulation."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from public_token_scoring import score as reference_score
from score_allocated_deep_language_v2 import score


def token_features(inputs, eos=None):
    return inputs.double().unsqueeze(-1)


class FlatReadout:
    def nll(self, x, labels):
        return (x.squeeze(-1) + labels.double()).reshape(-1)


class FakeModel:
    readout = FlatReadout()

    def eval(self):
        return self

    def forward(self, inputs, state, gen, args, deterministic=False):
        features = token_features(inputs)
        # Retain a nontrivial tape whose consumption must match across batches.
        return features + torch.rand(features.shape, generator=gen), state, None


class WindowScoringTest(unittest.TestCase):
    def test_flattened_losses_match_reference_across_partial_lane_batch(self):
        args = SimpleNamespace(seed=6, chunk=2, deterministic_eval=False)
        driver = SimpleNamespace(Model=FakeModel, torch=torch)
        tokens = np.arange(30, dtype=np.int64)
        for targets in (8, 12):
            result = score(FakeModel(), tokens, args, driver, 0, targets, 4, 2)
            reference = reference_score(FakeModel(), tokens, args, driver, 0, targets, 4, 2)
            self.assertEqual({k:v for k,v in result.items() if k!='window_losses'}, reference)
            blocks = result['window_losses']
            self.assertEqual(len(blocks), targets//4)
            self.assertAlmostEqual(sum(b['nll']*b['targets'] for b in blocks), result['loss_sum'])
            self.assertEqual([b['target_start'] for b in blocks], list(range(1, targets, 4)))
