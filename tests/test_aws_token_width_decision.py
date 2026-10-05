"""Width-decision protocol rejection; synthetic P24 is never research evidence."""
import copy
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from aws_token_width_decision import decide, collect


class WidthDecision(unittest.TestCase):
    def setUp(self):
        folder=ROOT/'experiments/results/token_language'
        tag='curie_data_growth_tokens_64k_b64_c16_p16_s6_20261005_v1'
        base=json.loads((folder/(tag+'.json')).read_text())
        selected=json.loads((folder/(tag+'.selection.json')).read_text())
        utility=json.loads((ROOT/'experiments/results/diagnostics/curie_data_growth_tokens_64k_utility_20261005_v1.json').read_text())['rows'][0]
        self.records={16:base,24:copy.deepcopy(base)}
        self.selections={16:selected,24:copy.deepcopy(selected)}
        self.utilities={16:utility,24:copy.deepcopy(utility)}
        candidate=self.records[24];candidate['args']['payload']=24;candidate['args']['tag']='synthetic_p24'
        candidate['parameters']+=1000;candidate['core_parameters']+=1000
        for row in candidate['curve'][1:]:row['dev_nll']-=.01
        self.selections[24]['original_result']='/workspace/experiments/results/token_language/synthetic_p24.json'
        self.selections[24]['selected']['dev_nll']-=.01
        self.utilities[24]['tag']='synthetic_p24';self.utilities[24]['selected']['dev_nll']-=.01

    def call(self):return decide(self.records,self.selections,self.utilities)

    def test_complete_learning_pair(self):
        d=self.call();self.assertEqual(d['selected_width'],24)
        self.assertFalse(d['scaling_ready'])
        self.assertTrue(d['comparisons']['24']['selected_writes_unchanged'])
        self.assertAlmostEqual(d['comparisons']['24']['state_scalar_ratio_against_p16'],1.5)

    def test_pending_rejected(self):
        self.records[24]['status']='running'
        with self.assertRaises(ValueError):self.call()

    def test_other_data_rejected(self):
        self.records[24]['identity']['train_sha256']='other'
        with self.assertRaises(ValueError):self.call()

    def test_other_optimizer_rejected(self):
        self.records[24]['args']['lr']*=2
        with self.assertRaises(ValueError):self.call()

    def test_other_history_rejected(self):
        self.records[24]['args']['state_mode']='reset'
        with self.assertRaises(ValueError):self.call()

    def test_unequal_selection_opportunity_rejected(self):
        self.records[24]['args']['eval_every']=32
        with self.assertRaises(ValueError):self.call()

    def test_missing_utility_rejected(self):
        del self.utilities[24]
        with self.assertRaises(ValueError):self.call()

    def test_negative_context_does_not_promote(self):
        self.utilities[24]['context_gain']=-.01
        self.assertEqual(self.call()['selected_width'],16)

    def test_no_independent_width_win_claim(self):
        d=self.call();self.assertFalse(d['independent_seed_confirmed'])
        self.assertIn('single-seed',d['comparisons']['24']['verdict'])


if __name__=='__main__':unittest.main()
