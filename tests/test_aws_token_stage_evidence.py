"""Result/selection/work binding contracts using saved completed8K evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('stage_evidence',ROOT/'experiments/aws_token_stage_evidence.py')
stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)


class Evidence(unittest.TestCase):
    def setUp(self):
        folder=ROOT/'experiments/results/token_language'
        tag='curie_fixed_batch_tokens_8k_b64_c16_s6_20261005_v1'
        self.result=json.loads((folder/(tag+'.json')).read_text())
        self.selection=json.loads((folder/(tag+'.selection.json')).read_text())
        self.work=json.loads((ROOT/'experiments/results/diagnostics/curie_fixed_batch_tokens_8k_c16_work_20261005_v1.json').read_text())

    def test_saved_complete_work(self):
        r=stage.stage_row(self.result,self.selection,self.work)
        self.assertEqual(r['fitting_targets'],16368)
        self.assertEqual(r['dev_targets'],2040)
        self.assertEqual(r['persistent_memory_scalars_per_lane'],256)
        self.assertEqual((r['scored_keys_per_target'],r['selected_writes_per_target']),(16,4))
        self.assertEqual(r['equivalent_passes'],2)
        self.assertAlmostEqual(r['whole_fit_gflops'],13.534669444)

    def test_missing_work_stays_unknown(self):
        r=stage.stage_row(self.result,self.selection)
        self.assertIsNone(r['whole_fit_gflops'])
        self.assertIsNone(r['fitting_mflops_per_target'])
        self.assertIsNone(r['inference_mflops_per_target'])

    def test_pending_fit_cannot_supply_quality(self):
        self.result['status']='running'
        with self.assertRaises(ValueError):stage.stage_row(self.result,self.selection)

    def test_final_weights_cannot_replace_selected(self):
        self.selection['selected']=dict(self.result['curve'][-1])
        with self.assertRaises(ValueError):stage.stage_row(self.result,self.selection)

    def test_work_denominator_mismatch(self):
        self.work['fitting_targets']=2040
        with self.assertRaises(ValueError):stage.stage_row(self.result,self.selection,self.work)

    def test_work_quality_mismatch(self):
        self.work['selected_dev_nll']+=.1
        with self.assertRaises(ValueError):stage.stage_row(self.result,self.selection,self.work)

    def test_unsupported_operator_rejected(self):
        self.work['fitting']['unsupported_floating_operators']={'unknown':1}
        with self.assertRaises(ValueError):stage.stage_row(self.result,self.selection,self.work)

    def test_mixed_units_rejected(self):
        self.work['fitting_arithmetic_flops_per_target']/=1e6
        with self.assertRaises(ValueError):stage.stage_row(self.result,self.selection,self.work)

    def test_other_control_rejected(self):
        self.work['control']='experiments/results/token_language/another.json'
        with self.assertRaises(ValueError):stage.stage_row(self.result,self.selection,self.work)


if __name__=='__main__':unittest.main()
