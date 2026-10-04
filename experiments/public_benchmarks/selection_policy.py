"""Prospective multi-seed DEV selection with an explicit compute preference.

This empirical one-standard-error rule is an allocation policy, not a test of
noninferiority. Freeze it before fresh DEV replications. Historical campaign
selections and TEST results are never inputs to this function.
"""
import math
import statistics

from experiments.public_benchmarks.evidence_audit import CONFIG_KEYS, paired_difference, prediction_metrics, require


def select_dev(records_by_candidate, targets, classes, expected_seeds, se_multiplier=1.):
    require(len(expected_seeds) >= 3 and len(set(expected_seeds)) == len(expected_seeds), 'At least three prescribed unique seeds required')
    require(math.isfinite(se_multiplier) and se_multiplier >= 0, 'Invalid frozen uncertainty multiplier')
    require(len(records_by_candidate) >= 2, 'At least two candidates required')
    candidates = {}
    shared_data = shared_source = shared_dataset = None
    for name, records in records_by_candidate.items():
        require(len(records) == len(expected_seeds), 'Missing or excess development replication')
        by_seed = {r['args']['seed']: r for r in records}
        require(len(by_seed) == len(records) and set(by_seed) == set(expected_seeds), 'Prescribed seed coverage mismatch')
        metrics, costs = {}, []
        config = tuple(records[0]['args'][k] for k in CONFIG_KEYS)
        for seed in expected_seeds:
            record = by_seed[seed]
            require(record.get('status') == 'completed' and record['args']['stage'] == 'screen' and record.get('test') is None, 'Only completed DEV-only screens allowed')
            require(tuple(record['args'][k] for k in CONFIG_KEYS) == config, 'Configuration changed within a candidate')
            require(record['parameters'] == records[0]['parameters'], 'Parameter count changed within a candidate')
            if shared_data is None:
                shared_data, shared_source = record['data_manifest_sha256'], record['source_sha256']
                shared_dataset = record['args']['dataset']
            require(record['data_manifest_sha256'] == shared_data and record['source_sha256'] == shared_source, 'Different data or executable programs')
            require(record['args']['dataset'] == shared_dataset, 'Different datasets cannot share a selection')
            metrics[seed] = prediction_metrics(record['development'], targets, classes)
            cost = record['work']['whole_fit_flops_estimate']
            require(math.isfinite(cost) and cost > 0, 'Invalid fitting cost')
            costs.append(cost)
        # Average losses, not probabilities: this selects a single-seed model
        # family rather than silently selecting a paid deployment ensemble.
        losses = {identity: statistics.mean(metrics[s]['losses'][identity] for s in expected_seeds) for identity in targets}
        candidates[name] = dict(losses=losses, metrics=metrics, mean_dev_nll=statistics.mean(losses.values()),
                                mean_screen_fit_flops=statistics.mean(costs), parameters=records[0]['parameters'],
                                config=config)
    best_name = min(candidates, key=lambda name: (candidates[name]['mean_dev_nll'], candidates[name]['mean_screen_fit_flops'], name))
    best = candidates[best_name]
    rows = []
    for name, candidate in candidates.items():
        paired = paired_difference(candidate['losses'], best['losses'])
        seed_deltas = [candidate['metrics'][s]['nll'] - best['metrics'][s]['nll'] for s in expected_seeds]
        seed_se = statistics.stdev(seed_deltas) / math.sqrt(len(seed_deltas))
        # Two crossed descriptive diagnostics. max is a deliberate allocation
        # heuristic, not an estimator or upper bound on generalization variance.
        uncertainty = max(paired['paired_standard_error'], seed_se)
        eligible = paired['left_minus_right'] <= se_multiplier * uncertainty
        rows.append(dict(candidate=name, mean_dev_nll=candidate['mean_dev_nll'],
                         delta_from_best_nll=paired['left_minus_right'], example_paired_se=paired['paired_standard_error'],
                         seed_paired_se=seed_se, allocation_tolerance=se_multiplier * uncertainty,
                         eligible=eligible, mean_screen_fit_flops=candidate['mean_screen_fit_flops'],
                         parameters=candidate['parameters']))
    chosen = min((r for r in rows if r['eligible']), key=lambda r: (r['mean_screen_fit_flops'], r['parameters'], r['mean_dev_nll'], r['candidate']))
    return dict(selected_candidate=chosen['candidate'], minimum_nll_candidate=best_name, candidates=rows,
                expected_seeds=list(expected_seeds), se_multiplier=se_multiplier,
                rule='Choose lowest observed fitting cost within the frozen paired DEV allocation tolerance of minimum mean NLL; parameters, NLL and name break ties.',
                scope='Prospective allocation heuristic. Selected epochs reuse DEV; crossed seeds/examples and adaptivity prevent interpreting this tolerance as a calibrated confidence bound. Selection cost is the screened program, not measured inference or an iso-quality proof. No TEST or ensemble output.')
