"""Hardware cost model contracts: exact parameter count and agreement with the recorded winner-only inference trace."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
import hardware_cost_model as H  # noqa: E402


def test_native_counts_match_completed_run_and_trace():
    r = H.native_counts(96, 4, 2, 2)
    assert r['parameters'] == 940875                     # curie p96/d4 6-pass result file
    assert abs(2 * r['macs_estimate'] / 1e6 - 1.3) < 0.05  # recorded winner-only 1.3 MFLOPs/position


def test_traffic_nearly_flat_in_pool():
    small, large = H.native_counts(64, 4, 2, 2), H.native_counts(64, 4, 2, 32)
    assert large['parameters'] > 10 * small['parameters']
    assert large['weights_read'] < 1.01 * small['weights_read']
