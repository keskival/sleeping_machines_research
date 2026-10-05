"""Admission checks run without importing Torch or executing model work."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('memory_audit', Path(__file__).resolve().parents[1]/'experiments/aws_selected_memory_path_audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class Admission(unittest.TestCase):
    def setUp(self):
        self.records = [dict(status='completed', args=dict(seed=s, credit_window=w,
            state_mode='carry', compiled=False, dev_offset=20971520, train_tokens=8192,
            tag=f'{s}_{w}'), identity=dict(train_sha256='train', dev_sha256='dev'),
            presentations_total=16368) for s in (6, 7) for w in (16, 64)]

    def test_complete_pair(self):
        audit.validate_inputs(self.records)

    def test_partial(self):
        with self.assertRaises(ValueError): audit.validate_inputs(self.records[:3])

    def test_duplicate(self):
        self.records[-1] = copy.deepcopy(self.records[0])
        with self.assertRaises(ValueError): audit.validate_inputs(self.records)

    def test_data_mismatch(self):
        self.records[-1]['identity']['dev_sha256'] = 'changed'
        with self.assertRaises(ValueError): audit.validate_inputs(self.records)

    def test_settings_mismatch(self):
        self.records[-1]['args']['train_tokens'] *= 2
        with self.assertRaises(ValueError): audit.validate_inputs(self.records)

    def test_public_interval(self):
        self.records[-1]['args']['dev_offset'] = 0
        with self.assertRaises(ValueError): audit.validate_inputs(self.records)

    def test_pending(self):
        self.records[-1]['status'] = 'running'
        with self.assertRaises(ValueError): audit.validate_inputs(self.records)

    def test_reset(self):
        self.records[-1]['args']['state_mode'] = 'reset'
        with self.assertRaises(ValueError): audit.validate_inputs(self.records)


if __name__ == '__main__': unittest.main()
