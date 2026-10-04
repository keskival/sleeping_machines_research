"""Saved-score lineage and selection-uncertainty checks; no ML imports."""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.public_benchmarks import evidence_audit as audit
from experiments.public_benchmarks.selection_policy import select_dev


def score(ids, probs, targets):
    return dict(identities=ids, probabilities=probs, targets=len(ids),
                nll=sum(-math.log(p[targets[i]]) for i, p in zip(ids, probs)) / len(ids),
                accuracy=sum(max(range(len(p)), key=p.__getitem__) == targets[i] for i, p in zip(ids, probs)) / len(ids))


class MetricsContracts(unittest.TestCase):
    def setUp(self):
        self.targets = {'a': 0, 'b': 1}
        self.score = score(['a', 'b'], [[.8, .2], [.3, .7]], self.targets)

    def test_labels_and_probabilities_reproduce_confusion(self):
        result = audit.prediction_metrics(self.score, self.targets, 2)
        self.assertEqual(result['confusion'], [[1, 0], [0, 1]])
        self.assertEqual(result['successes'], 2)
        self.assertAlmostEqual(result['nll'], self.score['nll'])

    def test_duplicate_missing_or_wrong_ids_reject(self):
        for ids in [['a', 'a'], ['a'], ['a', 'c']]:
            changed = copy.deepcopy(self.score)
            changed['identities'] = ids
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                audit.prediction_metrics(changed, self.targets, 2)

    def test_bad_normalization_nonfinite_and_underflow_reject(self):
        for probs in [[[.8, .3], [.3, .7]], [[float('nan'), .2], [.3, .7]],
                      [[0., 1.], [.3, .7]], [[1.2, -.2], [.3, .7]]]:
            changed = copy.deepcopy(self.score)
            changed['probabilities'] = probs
            with self.subTest(probs=probs), self.assertRaises(ValueError):
                audit.prediction_metrics(changed, self.targets, 2)

    def test_corrupt_saved_metrics_reject(self):
        for field in ['nll', 'accuracy', 'targets']:
            changed = copy.deepcopy(self.score)
            changed[field] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit.prediction_metrics(changed, self.targets, 2)

    def test_pairing_uses_ids_not_row_order(self):
        left = {'a': 1., 'b': 3., 'c': 2.}
        right = {'c': 1., 'a': 0., 'b': 2.}
        result = audit.paired_difference(left, right)
        self.assertEqual(result['left_minus_right'], 1.)
        self.assertEqual(result['paired_standard_error'], 0.)
        self.assertEqual(result, audit.paired_difference(dict(reversed(list(left.items()))), right))
        with self.assertRaises(ValueError):
            audit.paired_difference(left, {'a': 1., 'b': 2.})

    def test_wilson_small_sample_does_not_claim_certainty(self):
        low, high = audit.wilson(20, 20)
        self.assertLess(low, .85)
        self.assertAlmostEqual(high, 1.)
        low, high = audit.wilson(0, 20)
        self.assertAlmostEqual(low, 0.)
        self.assertGreater(high, .15)
        with self.assertRaises(ValueError):
            audit.wilson(0, 0)


class CampaignContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        raw = self.root / 'data/public_benchmarks/raw'
        raw.mkdir(parents=True)
        train = raw / 'ECG200_TRAIN.ts'
        test = raw / 'ECG200_TEST.ts'
        train.write_text('@timestamps false\n@data\n1,2:A\n2,3:B\n3,4:A\n4,5:B\n')
        test.write_text('@timestamps false\n@data\n1,2:A\n2,3:B\n')
        entry = dict(dataset='ECG200', labels={'A': 0, 'B': 1}, classes=2, train_rows=4,
                     expected_official_test_rows=2, fit_indices=[0, 1], dev_indices=[2, 3],
                     train_sha256=audit.digest(train), test_sha256=audit.digest(test))
        manifest = self.root / 'experiments/public_benchmarks/data_manifest.json'
        manifest.parent.mkdir(parents=True)
        self.write(manifest, dict(datasets=[entry]))
        directory = self.root / 'experiments/results/public_benchmarks'
        directory.mkdir(parents=True)
        dev_targets = {'TRAIN:2': 0, 'TRAIN:3': 1}
        development = score(list(dev_targets), [[.8, .2], [.3, .7]], dev_targets)
        args = dict(dataset='ECG200', payload=16, depth=2, pool=2, heads=2, lr=.003, batch=32,
                    stage='screen', seed=6, epochs=1)
        base = dict(status='completed', args=args, data_manifest_sha256=audit.digest(manifest),
                    development=development, test=None, best_epoch=1,
                    curve=[dict(epoch=1, dev=development, mean_batch_train_nll=.4)],
                    work=dict(fit_flops_per_presented_target_estimate=100., whole_fit_flops_estimate=200., scope='fixture'),
                    fitting_presentations=2, parameters=10, wall_s=1., capacity={}, source_sha256={'core.py': 'frozen'})
        candidates = []
        for i in range(3):
            record = copy.deepcopy(base)
            record['parameters'] += i
            path = directory / f'screen_{i}.json'
            self.write(path, record)
            candidates.append(dict(path=str(path.relative_to(self.root)), sha256=audit.digest(path),
                                   dev_nll=development['nll'], work=200., parameters=record['parameters']))
        self.selection = directory / 'selection.json'
        chosen = candidates[0]
        selection = dict(status='completed', args={'dataset': 'ECG200'}, test=None, candidates=candidates,
                         selected_screen=chosen['path'], selected_screen_sha256=chosen['sha256'], final_epochs=1,
                         final_seeds=[6, 7, 8])
        self.write(self.selection, selection)
        targets = {'TEST:0': 0, 'TEST:1': 1}
        for seed in [6, 7, 8]:
            final = copy.deepcopy(base)
            final['args'].update(stage='final', seed=seed, selection=chosen['path'])
            final['development'] = None
            final['test'] = score(list(targets), [[.8, .2], [.8, .2] if seed == 7 else [.3, .7]], targets)
            final['selection_parent_sha256'] = chosen['sha256']
            final['fitting_presentations'] = 4
            final['work']['whole_fit_flops_estimate'] = 400.
            final['curve'][0]['dev'] = None
            self.write(directory / f'selection_final_s{seed}.json', final)

    @staticmethod
    def write(path, record):
        path.write_text(json.dumps(record))

    def run_audit(self):
        return audit.audit_campaign(self.root, [self.selection])

    def test_complete_campaign_reuses_test_population(self):
        result = self.run_audit()
        row = result['datasets'][0]
        self.assertEqual(row['test_targets'], 2)
        self.assertAlmostEqual(row['mean_test_accuracy'], 5 / 6)
        self.assertEqual(row['all_seeds_wrong'], 0)
        self.assertEqual(len(row['seed_pairs']), 3)
        self.assertAlmostEqual(row['total_final_fit_gflops_estimate'], 1200 / 1e9)
        self.assertEqual(result['claims'], [])

    def test_missing_seed_is_not_silently_averaged(self):
        self.selection.with_name('selection_final_s7.json').unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_audit()

    def test_tampered_parent_rejects(self):
        path = self.selection.with_name('screen_0.json')
        path.write_text(path.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.run_audit()

    def test_changed_final_configuration_or_sources_reject(self):
        path = self.selection.with_name('selection_final_s6.json')
        original = json.loads(path.read_text())
        for mutation in ['config', 'source', 'manifest', 'parent', 'stage', 'presentations']:
            changed = copy.deepcopy(original)
            if mutation == 'config': changed['args']['pool'] += 1
            if mutation == 'source': changed['source_sha256']['core.py'] = 'changed'
            if mutation == 'manifest': changed['data_manifest_sha256'] = 'changed'
            if mutation == 'parent': changed['selection_parent_sha256'] = 'changed'
            if mutation == 'stage': changed['args']['stage'] = 'screen'
            if mutation == 'presentations': changed['fitting_presentations'] += 1
            self.write(path, changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.run_audit()
        self.write(path, original)

    def test_changed_dataset_bytes_reject(self):
        path = self.root / 'data/public_benchmarks/raw/ECG200_TEST.ts'
        path.write_text(path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'TEST bytes'):
            self.run_audit()

    def test_inconsistent_whole_fit_denominator_rejects(self):
        path = self.selection.with_name('selection_final_s6.json')
        record = json.loads(path.read_text())
        record['work']['whole_fit_flops_estimate'] = 40.
        self.write(path, record)
        with self.assertRaisesRegex(ValueError, 'work denominator'):
            self.run_audit()

    def test_import_does_not_load_numerical_runtimes(self):
        for name in ['torch', 'numpy', 'h5py']:
            self.assertNotIn(name, sys.modules)

    def selection_candidates(self):
        base = json.loads(self.selection.with_name('screen_0.json').read_text())
        result = {}
        for name, cost in [('cheap', 200.), ('expensive', 2000.)]:
            rows = []
            for seed in [6, 7, 8]:
                row = copy.deepcopy(base)
                row['args']['seed'] = seed
                row['work']['whole_fit_flops_estimate'] = cost
                rows.append(row)
            result[name] = rows
        return result

    def test_prospective_policy_prefers_cheaper_equal_quality(self):
        candidates = self.selection_candidates()
        result = select_dev(candidates, {'TRAIN:2': 0, 'TRAIN:3': 1}, 2, [6, 7, 8])
        self.assertEqual(result['selected_candidate'], 'cheap')
        self.assertTrue(all(r['eligible'] for r in result['candidates']))

    def test_prospective_policy_preserves_clear_quality_advantage(self):
        candidates = self.selection_candidates()
        targets = {'TRAIN:2': 0, 'TRAIN:3': 1}
        for row in candidates['cheap']:
            row['development'] = score(list(targets), [[.6, .4], [.4, .6]], targets)
        for row in candidates['expensive']:
            row['development'] = score(list(targets), [[.9, .1], [.1, .9]], targets)
        result = select_dev(candidates, targets, 2, [6, 7, 8])
        self.assertEqual(result['selected_candidate'], 'expensive')
        self.assertFalse(next(r for r in result['candidates'] if r['candidate'] == 'cheap')['eligible'])

    def test_prospective_policy_uses_paired_uncertainty_for_compute_tradeoff(self):
        candidates = self.selection_candidates()
        targets = {'TRAIN:2': 0, 'TRAIN:3': 1}
        for row in candidates['cheap']:
            row['development'] = score(list(targets), [[.99, .01], [.2, .8]], targets)
        for row in candidates['expensive']:
            row['development'] = score(list(targets), [[.9, .1], [.1, .9]], targets)
        result = select_dev(candidates, targets, 2, [6, 7, 8])
        self.assertEqual(result['minimum_nll_candidate'], 'expensive')
        self.assertEqual(result['selected_candidate'], 'cheap')
        # A frozen zero tolerance selects the empirical quality leader.
        strict = select_dev(candidates, targets, 2, [6, 7, 8], se_multiplier=0.)
        self.assertEqual(strict['selected_candidate'], 'expensive')

    def test_prospective_policy_rejects_test_access_and_missing_seeds(self):
        for mutation in ['test', 'seed', 'source', 'dataset']:
            candidates = self.selection_candidates()
            if mutation == 'test': candidates['cheap'][0]['test'] = {}
            if mutation == 'seed': candidates['cheap'].pop()
            if mutation == 'source': candidates['cheap'][0]['source_sha256']['core.py'] = 'changed'
            if mutation == 'dataset':
                for row in candidates['expensive']:
                    row['args']['dataset'] = 'another_dataset'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                select_dev(candidates, {'TRAIN:2': 0, 'TRAIN:3': 1}, 2, [6, 7, 8])


if __name__ == '__main__':
    unittest.main()
