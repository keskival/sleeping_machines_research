"""Synthetic bank metadata fixtures test refusal rules, not model evidence."""
import copy
import json
from pathlib import Path
import unittest
from experiments.aws_private_bank_comparison import compare, EXACT_FIELDS
from experiments.aws_private_bank_policy import replica_policy
ROOT=Path(__file__).resolve().parents[1]


class ComparisonTest(unittest.TestCase):
    def setUp(self):
        folder=ROOT/'experiments/results/token_language'
        tag='curie_fixed_batch_tokens_8k_b64_c16_s6_20261005_v1'
        raw=json.loads((folder/(tag+'.json')).read_text())
        selected=json.loads((folder/(tag+'.selection.json')).read_text())
        ledger=json.loads((ROOT/'experiments/results/diagnostics/curie_fixed_batch_tokens_8k_c16_work_20261005_v1.json').read_text())
        self.arms=[]
        for pool in (4,16):
            r,s,w=copy.deepcopy((raw,selected,ledger));r['args'].update(pool=pool,tie_pools=True,temperature=1.,free_bias=0.)
            recipe=dict(bank_policy=replica_policy(4,pool),source_sha256={'helper':'fixture'})
            r['bank_recipe']=recipe;r['source_sha256']={'fit':'fixture'}
            w.update(bank_recipe=recipe,source_sha256={'fit':'fixture','work':'fixture'},
                     final_numeric_training_state_exact=True,exact_checkpoint_fields=list(EXACT_FIELDS))
            u=dict(tag=tag,fit_targets=r['presentations_total'],dev_targets=w['inference_targets'],selected=s['selected'],
                partition_parity=True,matched_rng=True,context_gain=.1,history_gains=dict(memory=.01,message=-.02),
                checkpoint_sha256='fixture',bank_recipe=recipe)
            self.arms.append([r,s,w,u])
    def test_activity_and_capacity_separate(self):
        record=compare(*self.arms)
        self.assertEqual(record['state_capacity_ratio'],4)
        self.assertEqual(record['scored_key_ratio'],4)
        self.assertEqual(record['selected_write_ratio'],1)
        self.assertEqual(record['quality_verdict'],'tie')
        self.assertFalse(record['pareto_win'])
    def test_missing_work_rejected(self):
        self.arms[1][2]=None
        with self.assertRaises(ValueError):compare(*self.arms)
    def test_changed_credit_rejected(self):
        self.arms[1][0]['args']['credit_window']=64
        with self.assertRaises(ValueError):compare(*self.arms)
    def test_inexact_checkpoint_rejected(self):
        self.arms[1][2]['final_numeric_training_state_exact']=False
        with self.assertRaises(ValueError):compare(*self.arms)
    def test_missing_generator_rejected(self):
        self.arms[1][2]['exact_checkpoint_fields'].remove('generator')
        with self.assertRaises(ValueError):compare(*self.arms)
    def test_wrong_recipe_rejected(self):
        self.arms[1][3]['bank_recipe']={'wrong':True}
        with self.assertRaises(ValueError):compare(*self.arms)
    def test_changed_source_rejected(self):
        self.arms[1][2]['source_sha256']['fit']='different'
        with self.assertRaises(ValueError):compare(*self.arms)
