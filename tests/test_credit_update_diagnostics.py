import unittest
from sleeping_machines.credit_update_diagnostics import CreditUpdateDiagnostics


class CreditDiagnosticTest(unittest.TestCase):
    def test_exact_resume_preserves_full_acquisition_counts(self):
        uninterrupted=CreditUpdateDiagnostics();first=CreditUpdateDiagnostics()
        rows=[(.5,None),(2.,3.),(1.,0.),(1.5,4.)]
        for row in rows:uninterrupted.observe(*row)
        for row in rows[:2]:first.observe(*row)
        resumed=CreditUpdateDiagnostics();resumed.load_state_dict(first.state_dict())
        for row in rows[2:]:resumed.observe(*row)
        self.assertEqual(resumed.packet(),uninterrupted.packet())
        self.assertEqual(resumed.packet()['clipping_fraction'],.5)
        self.assertEqual(resumed.packet()['teacher_steps'],3)

    def test_invalid_receipt_does_not_advance_counters(self):
        stats=CreditUpdateDiagnostics()
        with self.assertRaises(ValueError):stats.observe(.5,float('nan'))
        self.assertEqual(stats.steps,0)
        with self.assertRaises(ValueError):stats.load_state_dict({})
