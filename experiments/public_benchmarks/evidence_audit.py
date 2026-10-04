"""Audit completed archive evidence without running models or changing selections.

Uses saved probabilities and hash-verified archive labels. Paired DEV statistics
are descriptive: those examples already selected epochs/configurations. Nothing
here certifies a public win, estimates energy, or treats seeds as fresh TEST sets.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

from experiments.public_benchmarks.data import read_ts

ROOT = Path(__file__).resolve().parents[2]
CONFIG_KEYS = ('dataset', 'payload', 'depth', 'pool', 'heads', 'lr', 'batch')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_record(path):
    record = json.loads(Path(path).read_text())
    require(record.get('status') == 'completed', f'Incomplete evidence: {path}')
    return record


def wilson(successes, count, z=1.959963984540054):
    require(0 <= successes <= count and count > 0, 'Invalid binomial counts')
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [max(0., center - radius), min(1., center + radius)]


def prediction_metrics(score, targets, classes):
    ids, probs = score['identities'], score['probabilities']
    require(len(ids) == len(probs) == score['targets'] == len(targets), 'Prediction coverage mismatch')
    require(len(set(ids)) == len(ids) and set(ids) == set(targets), 'Duplicate or mismatched prediction IDs')
    losses, correct, predictions = {}, {}, {}
    confusion = [[0] * classes for _ in range(classes)]
    for identity, p in zip(ids, probs):
        require(len(p) == classes and all(math.isfinite(v) and 0 <= v <= 1 for v in p), 'Invalid probabilities')
        require(abs(sum(p) - 1) < 2e-6, 'Unnormalized probabilities')
        target = targets[identity]
        require(type(target) is int and 0 <= target < classes, 'Invalid target')
        # Softmax underflow can make a true-class probability exactly zero.
        # Such records cannot reproduce NLL from saved probabilities alone.
        require(p[target] > 0, 'True-class underflow: use retained logits for an NLL audit')
        pred = max(range(classes), key=p.__getitem__)
        predictions[identity] = pred
        correct[identity] = int(pred == target)
        losses[identity] = -math.log(p[target])
        confusion[target][pred] += 1
    nll = statistics.mean(losses.values())
    successes = sum(correct.values())
    require(abs(nll - score['nll']) < 2e-5, 'Saved NLL disagrees with probabilities/labels')
    require(abs(successes / len(ids) - score['accuracy']) < 1e-10, 'Saved accuracy disagrees with probabilities/labels')
    return dict(losses=losses, correct=correct, predictions=predictions, nll=nll,
                successes=successes, confusion=confusion,
                wilson95=wilson(successes, len(ids)))


def paired_difference(left, right):
    require(set(left) == set(right) and len(left) > 1, 'Paired observations must have identical IDs')
    delta = [left[k] - right[k] for k in sorted(left)]
    mean = statistics.mean(delta)
    se = statistics.stdev(delta) / math.sqrt(len(delta))
    return dict(left_minus_right=mean, paired_standard_error=se,
                descriptive_normal95=[mean - 1.959963984540054 * se, mean + 1.959963984540054 * se],
                scope='Same DEV examples, selected epochs. Ignores adaptive selection and training-seed uncertainty; not a calibrated post-selection confidence interval.')


def work_row(record):
    work = record['work']
    presentations = record['fitting_presentations']
    per, whole = work['fit_flops_per_presented_target_estimate'], work['whole_fit_flops_estimate']
    require(presentations > 0 and all(math.isfinite(v) and v > 0 for v in [per, whole]), 'Invalid fitting work')
    require(math.isclose(whole, per * presentations, rel_tol=1e-10), 'Inconsistent work denominator')
    return dict(parameters=record['parameters'], presentations=presentations,
                whole_fit_gflops_estimate=whole / 1e9,
                per_presented_target_mflops_estimate=per / 1e6,
                capacity=record['capacity'], wall_s=record['wall_s'], scope=work['scope'],
                inference_work=None, measured_energy=None)


def audit_selection(root, selection_path, entry, manifest_sha):
    """Validate lineage and recompute DEV statistics before touching TEST labels."""
    selection = load_record(selection_path)
    require(selection.get('test') is None, 'Selection accessed TEST')
    name = entry['dataset']
    train_path = root / 'data/public_benchmarks/raw' / f'{name}_TRAIN.ts'
    require(digest(train_path) == entry['train_sha256'], 'TRAIN bytes differ from frozen manifest')
    train, _ = read_ts(train_path)
    require(len(train) == entry['train_rows'], 'TRAIN count mismatch')
    mapping = entry['labels']
    fit, dev = entry['fit_indices'], entry['dev_indices']
    require(len(set(fit)) == len(fit) and len(set(dev)) == len(dev), 'Duplicate split IDs')
    require(not set(fit) & set(dev) and set(fit + dev) == set(range(len(train))), 'Invalid FIT/DEV partition')
    targets = {f'TRAIN:{i}': mapping[train[i]['label']] for i in dev}
    screens = []
    for candidate in selection['candidates']:
        path = root / candidate['path']
        require(digest(path) == candidate['sha256'], 'Candidate hash mismatch')
        record = load_record(path)
        require(record['args']['dataset'] == name and record['args']['stage'] == 'screen' and record['test'] is None, 'Invalid DEV parent')
        require(record['data_manifest_sha256'] == manifest_sha, 'Screen data lineage mismatch')
        metrics = prediction_metrics(record['development'], targets, entry['classes'])
        require(candidate['dev_nll'] == record['development']['nll'] and candidate['work'] == record['work']['whole_fit_flops_estimate'] and candidate['parameters'] == record['parameters'], 'Candidate ranking data mismatch')
        require(1 <= record['best_epoch'] <= len(record['curve']), 'Invalid selected epoch')
        best = min(record['curve'], key=lambda r: r['dev']['nll'])
        require(best['epoch'] == record['best_epoch'] and best['dev'] == record['development'], 'Selected checkpoint differs from saved DEV minimum')
        screens.append((candidate['path'], record, metrics))
    require(len(screens) == 3 and len({s[0] for s in screens}) == 3, 'Expected three unique screens')
    chosen = min(screens, key=lambda s: (s[1]['development']['nll'], s[1]['work']['whole_fit_flops_estimate'], s[1]['parameters'], s[0]))
    require(chosen[0] == selection['selected_screen'], 'Selection differs from declared rule')
    require(digest(root / chosen[0]) == selection['selected_screen_sha256'], 'Selected parent hash mismatch')
    require(chosen[1]['best_epoch'] == selection['final_epochs'], 'Final epoch lineage mismatch')
    comparisons = []
    for path, record, metrics in screens:
        if path == chosen[0]:
            continue
        comparisons.append(dict(alternative=path, **paired_difference(chosen[2]['losses'], metrics['losses']),
                                selected_over_alternative_fit_work=chosen[1]['work']['whole_fit_flops_estimate'] / record['work']['whole_fit_flops_estimate']))
    screen_rows = [dict(path=path, seed=r['args']['seed'], best_epoch=r['best_epoch'],
                        dev_accuracy=r['development']['accuracy'], dev_nll=r['development']['nll'],
                        train_mean_batch_nll_at_selected_epoch=r['curve'][r['best_epoch'] - 1]['mean_batch_train_nll'],
                        dev_wilson95_conditional=metrics['wilson95'], **work_row(r)) for path, r, metrics in screens]
    return selection, chosen[1], dict(selected=chosen[0], dev_targets=len(dev), screens=screen_rows, paired_dev=comparisons)


def audit_campaign(root, selection_paths):
    root = Path(root)
    manifest_path = root / 'experiments/public_benchmarks/data_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = digest(manifest_path)
    entries = {e['dataset']: e for e in manifest['datasets']}
    datasets = []
    for selection_path in selection_paths:
        selection_path = Path(selection_path)
        if not selection_path.is_absolute():
            selection_path = root / selection_path
        initial = load_record(selection_path)
        entry = entries[initial['args']['dataset']]
        selection, selected, development = audit_selection(root, selection_path, entry, manifest_sha)
        seeds = selection['final_seeds']
        require(len(seeds) >= 2 and len(set(seeds)) == len(seeds), 'Invalid prescribed seeds')
        finals = []
        for seed in seeds:
            path = selection_path.with_name(selection_path.stem + f'_final_s{seed}.json')
            record = load_record(path)
            require(record['args']['stage'] == 'final' and record['args']['seed'] == seed, 'Final identity mismatch')
            require(record['selection_parent_sha256'] == selection['selected_screen_sha256'] and record['args']['selection'] == selection['selected_screen'], 'Final parent mismatch')
            require(record['data_manifest_sha256'] == manifest_sha, 'Final data lineage mismatch')
            require(all(record['args'][k] == selected['args'][k] for k in CONFIG_KEYS) and record['args']['epochs'] == selection['final_epochs'], 'Final configuration mismatch')
            require(record['source_sha256'] == selected['source_sha256'], 'Changed program between screen and final')
            require(record['fitting_presentations'] == entry['train_rows'] * selection['final_epochs'], 'Final presentation count mismatch')
            finals.append((path, record))
        test_path = root / 'data/public_benchmarks/raw' / f"{entry['dataset']}_TEST.ts"
        require(digest(test_path) == entry['test_sha256'], 'TEST bytes differ from frozen manifest')
        test, _ = read_ts(test_path)
        require(len(test) == entry['expected_official_test_rows'], 'TEST count mismatch')
        targets = {f'TEST:{i}': entry['labels'][r['label']] for i, r in enumerate(test)}
        metrics = [prediction_metrics(r['test'], targets, entry['classes']) for _, r in finals]
        pairs = []
        for a, b in itertools.combinations(range(len(finals)), 2):
            ma, mb = metrics[a], metrics[b]
            pairs.append(dict(seeds=[seeds[a], seeds[b]], prediction_disagreement=sum(ma['predictions'][i] != mb['predictions'][i] for i in targets) / len(targets),
                              left_only_correct=sum(ma['correct'][i] and not mb['correct'][i] for i in targets),
                              right_only_correct=sum(mb['correct'][i] and not ma['correct'][i] for i in targets),
                              both_wrong=sum(not ma['correct'][i] and not mb['correct'][i] for i in targets)))
        rows = [dict(path=str(path.relative_to(root)), sha256=digest(path), seed=seed,
                     accuracy=r['test']['accuracy'], nll=r['test']['nll'], correct=m['successes'],
                     test_wilson95_conditional=m['wilson95'], confusion=m['confusion'],
                     train_mean_batch_nll_last_epoch=r['curve'][-1]['mean_batch_train_nll'], **work_row(r))
                for (path, r), seed, m in zip(finals, seeds, metrics)]
        accuracy = [r['accuracy'] for r in rows]
        datasets.append(dict(dataset=entry['dataset'], selection_sha256=digest(selection_path), development=development,
                             final_seeds=rows, test_targets=len(targets), mean_test_accuracy=statistics.mean(accuracy),
                             test_accuracy_seed_stddev=statistics.stdev(accuracy),
                             test_accuracy_seed_range=[min(accuracy), max(accuracy)],
                             dev_minus_mean_test_accuracy=selected['development']['accuracy'] - statistics.mean(accuracy),
                             test_class_counts=dict(Counter(targets.values())), seed_pairs=pairs,
                             all_seeds_wrong=sum(all(not m['correct'][i] for m in metrics) for i in targets),
                             total_screen_fit_gflops_estimate=sum(r['whole_fit_gflops_estimate'] for r in development['screens']),
                             total_final_fit_gflops_estimate=sum(r['whole_fit_gflops_estimate'] for r in rows)))
    return dict(status='completed_evidence_audit', data_manifest_sha256=manifest_sha, datasets=datasets,
                source_sha256={'experiments/public_benchmarks/evidence_audit.py': digest(Path(__file__)),
                               'experiments/public_benchmarks/data.py': digest(ROOT / 'experiments/public_benchmarks/data.py')},
                claims=[], scope='Read-only saved predictions, labels and lineage; no numerical model execution. Three seeds reuse each official TEST set. Wilson intervals condition on one fitted predictor and assume independent examples; serial or subject dependence and selection bias are not covered. DEV paired intervals ignore adaptive selection. No baseline parity, inference/energy or SOTA certificate. Campaign costs include screens and finals only; pilots, contracts and failed attempts remain separate.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', action='append', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = audit_campaign(ROOT, args.selection)
    path = Path(args.output)
    with path.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps([dict(dataset=d['dataset'], mean_test_accuracy=d['mean_test_accuracy'],
                           seed_stddev=d['test_accuracy_seed_stddev']) for d in result['datasets']]))


if __name__ == '__main__':
    main()
