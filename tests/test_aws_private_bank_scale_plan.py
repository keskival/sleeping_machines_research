"""Synthetic admission fixtures are not benchmark evidence."""
import copy
import importlib.util
from pathlib import Path
import unittest
from experiments.aws_private_bank_scale_plan import plan
p=Path(__file__).with_name('test_aws_private_bank_comparison.py')
s=importlib.util.spec_from_file_location('bank_comparison_fixture',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


class ScalePlanTest(unittest.TestCase):
    def setUp(self):
        fixture=m.ComparisonTest();fixture.setUp();self.arms=fixture.arms
        for arm in self.arms:arm[0]["args"]["payload"]=24
    def test_resources_and_paired_seeds(self):
        r=plan(*self.arms,24*1024*1024)
        self.assertFalse(r['admitted'])
        self.assertEqual([(j['pool'],j['seed']) for j in r['jobs']],[(4,6),(4,7),(16,6),(16,7)])
        for j in r['jobs']:
            self.assertEqual(j['settings']['train_tokens'],65536)
            self.assertEqual(j['settings']['steps'],256)
            self.assertGreater(j['rss_kb'],self.arms[0][0]['max_rss_kb'])
    def test_memory_floor(self):
        with self.assertRaises(ValueError):plan(*self.arms,8*1024*1024)
    def test_missing_teacher(self):
        self.arms[1][0]['curve'][1]['future_write_teacher']=None
        with self.assertRaises(ValueError):plan(*self.arms,24*1024*1024)
    def test_no_context(self):
        self.arms[1][3]['context_gain']=0
        with self.assertRaises(ValueError):plan(*self.arms,24*1024*1024)
    def test_unbounded_timeout(self):
        self.arms[1][0]['train_tokens_per_second']=1
        with self.assertRaises(ValueError):plan(*self.arms,24*1024*1024)
