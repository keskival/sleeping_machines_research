import copy
import unittest

import torch

from sleeping_machines.addressed_event_heads import AddressedEventHeads
from sleeping_machines.fast_native_core import fast_class
from sleeping_machines.sparse_counterfactual_episodes import token_features as shared_features
from sleeping_machines.lane_independent_races import token_features, _noise_for_lanes, all_depth_token_features, independent_forward


def model():
    torch.manual_seed(31)
    return fast_class(AddressedEventHeads)(sources=1, content_dim=7, classes=7,
        payload=4, depth=2, heads=2, pool=3).double()


def generators():
    return torch.Generator().manual_seed(19), torch.Generator().manual_seed(43)


def assert_state(test, first, second, omit=()):
    for key, value in first.items():
        if key in omit:
            continue
        if isinstance(value, list):
            for a, b in zip(value, second[key]):
                test.assertTrue(torch.equal(a, b), key)
        else:
            test.assertTrue(torch.equal(value, second[key]), key)


class IndependentRaceTest(unittest.TestCase):
    def test_single_lane_matches_legacy_features_state_all_gradients_and_tapes(self):
        first = model()
        second = copy.deepcopy(first)
        ids = torch.tensor([[0, 1, 2, 3]])
        g1, a1 = generators()
        g2, a2 = generators()
        x, sx, _ = shared_features(first, ids, generator=g1, alternative_generator=a1)
        y, sy, _ = token_features(second, ids, generator=g2, alternative_generator=a2)
        self.assertTrue(torch.equal(x, y))
        assert_state(self, sx, sy)
        self.assertTrue(torch.equal(g1.get_state(), g2.get_state()))
        self.assertTrue(torch.equal(a1.get_state(), a2.get_state()))
        x.square().sum().backward()
        y.square().sum().backward()
        for (name, p), (_, q) in zip(first.named_parameters(), second.named_parameters()):
            self.assertEqual(p.grad is None, q.grad is None, name)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, atol=1e-12, rtol=1e-12)

    def test_deterministic_multi_lane_matches_existing_function_and_gradients(self):
        first = model()
        second = copy.deepcopy(first)
        ids = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 0, 1]])
        x, sx, _ = shared_features(first, ids, generator=generators()[0],
            route_credit='none', deterministic=True)
        y, sy, _ = token_features(second, ids, generator=generators()[0],
            route_credit='none', deterministic=True)
        self.assertTrue(torch.equal(x, y))
        assert_state(self, sx, sy)
        x.square().sum().backward()
        y.square().sum().backward()
        for p, q in zip(first.parameters(), second.parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, atol=1e-12, rtol=1e-12)

    def test_chunk_partition_and_checkpoint_replay_preserve_independent_races(self):
        m = model()
        ids = torch.tensor([[0, 1, 2, 3], [3, 4, 5, 6]])
        g, a = generators()
        full, sf, _ = token_features(m, ids, generator=g, alternative_generator=a)
        g2, a2 = generators()
        prefix, state, _ = token_features(m, ids[:, :2], generator=g2, alternative_generator=a2)
        checkpoint_state = {k: [t.detach().clone() for t in v] if isinstance(v, list) else v.detach().clone() for k, v in state.items()}
        checkpoint = (checkpoint_state, g2.get_state().clone(), a2.get_state().clone())
        suffix, sc, _ = token_features(m, ids[:, 2:], state, g2, alternative_generator=a2)
        self.assertTrue(torch.equal(full, torch.cat((prefix, suffix), dim=1)))
        assert_state(self, sf, sc, omit=('race_winners', 'last_writes'))
        self.assertTrue(torch.equal(g.get_state(), g2.get_state()))
        self.assertTrue(torch.equal(a.get_state(), a2.get_state()))
        restored, gr, ar = checkpoint
        replay, sr, _ = token_features(m, ids[:, 2:], restored,
            torch.Generator().set_state(gr), alternative_generator=torch.Generator().set_state(ar))
        self.assertTrue(torch.equal(suffix, replay))
        assert_state(self, sc, sr)

    def test_identical_lane_inputs_no_longer_force_identical_exploration(self):
        m = model()
        ids = torch.zeros((32, 1), dtype=torch.long)
        _, old, _ = shared_features(m, ids, generator=generators()[0], route_credit='none')
        _, new, _ = token_features(m, ids, generator=generators()[0], route_credit='none')
        old_routes, new_routes = old['race_winners'], new['race_winners']
        self.assertTrue(torch.equal(old_routes, old_routes[:, :, :1, :].expand_as(old_routes)))
        self.assertGreater(torch.unique(new_routes[0, 0, :, 0]).numel(), 1)

    def test_supplied_ordered_content_keeps_its_gradient_path(self):
        m = model()
        ids = torch.tensor([[0, 1], [2, 3]])
        content = [torch.randn(2, 8, dtype=torch.float64, requires_grad=True) for _ in range(2)]
        x, _, _ = token_features(m, ids, generator=generators()[0],
            route_credit='none', token_content=content)
        gradients = torch.autograd.grad(x.square().sum(), content)
        self.assertTrue(all(bool(torch.isfinite(g).all()) and bool((g != 0).any()) for g in gradients))

    def test_all_depth_interface_preserves_single_lane_legacy_features_and_gradients(self):
        from sleeping_machines.multidepth_counterfactual_episodes import token_features as old
        first, second = model(), model()
        ids = torch.tensor([[0, 1, 2]])
        g, a = generators()
        x, sx, _ = old(first, ids, generator=g, alternative_generator=a)
        g, a = generators()
        y, sy, _ = all_depth_token_features(second, ids, generator=g, alternative_generator=a)
        self.assertEqual(y.shape[-1], second.depth * second.total_payload)
        self.assertTrue(torch.equal(x, y))
        assert_state(self, sx, sy)
        x.square().sum().backward()
        y.square().sum().backward()
        for p, q in zip(first.parameters(), second.parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, atol=1e-12, rtol=1e-12)

    def test_native_language_forward_adapter_is_private_and_replayable(self):
        from experiments.horizon_token_language_engine import Model
        from types import SimpleNamespace
        args = SimpleNamespace(seed=6, payload=4, depth=2, heads=2, pool=3,
            input_init='balanced', tie_pools=False, readout='adaptive',
            readout_init='frequency', minimum_tail_width=0, state_mode='carry',
            route_credit='sampled', future_site='first', free_bias=0.,
            temperature=1., compiled=False, credit_window=2)
        torch.manual_seed(6)
        m = Model(args, torch.arange(50257), torch.ones(50257, dtype=torch.long))
        # Frequency initialization deliberately zeros contextual output maps;
        # use a nonzero readout fixture to test core credit through this adapter.
        with torch.no_grad():
            for parameter in m.readout.parameters():
                if parameter.ndim == 2:
                    parameter.normal_(0., .01)
        forward = independent_forward(Model.forward)
        self.assertIsNot(forward.__globals__, Model.forward.__globals__)
        ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
        saved = m.local_generator.get_state().clone()
        x, state, _ = forward(m, ids, None, torch.Generator().manual_seed(13), args)
        m.local_generator.set_state(saved)
        y, restored, _ = forward(m, ids, None, torch.Generator().manual_seed(13), args)
        self.assertTrue(torch.equal(x, y))
        assert_state(self, state, restored)
        loss = m.readout.nll(x, ids).mean()
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(any(p.grad is not None and bool((p.grad != 0).any()) for p in m.core.parameters()))

    def test_shape_validation_and_eager_scope(self):
        with self.assertRaises(ValueError):
            _noise_for_lanes(torch.ones(2, 2, 2, 3), 2, 2, 3)
        with self.assertRaises(ValueError):
            token_features(model(), torch.zeros((2, 1), dtype=torch.long), compiled=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
