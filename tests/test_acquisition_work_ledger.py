import unittest
from sleeping_machines.acquisition_work_ledger import acquisition_work


class WorkLedgerTest(unittest.TestCase):
    def test_unknown_tail_is_never_published_as_exact(self):
        packet = dict(presentations_total=100, process_wall_s=10,
                      whole_fit_process_wall_s=11,
                      whole_fit_process_wall_lower_bound_s=30)
        result = acquisition_work(packet)
        self.assertEqual(result['whole_fit_process_wall_lower_bound_s'], 30)
        self.assertEqual(result['whole_fit_process_lower_bound_s_per_target'], .3)
        self.assertNotIn('whole_fit_process_wall_s', result)
        self.assertNotIn('whole_fit_process_s_per_target', result)

    def test_prior_interruption_is_included(self):
        result = acquisition_work(dict(presentations_total=100, process_wall_s=10,
                                       whole_fit_process_wall_s=25))
        self.assertEqual(result['whole_fit_process_wall_s'], 25)

    def test_incomplete_ledger_is_rejected(self):
        with self.assertRaises(ValueError):
            acquisition_work(dict(presentations_total=100, process_wall_s=10,
                                  whole_fit_process_wall_lower_bound_s=9))
