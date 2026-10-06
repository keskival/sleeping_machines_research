"""Aggregate completed matched time-scale pairs without pending estimates."""
try:
    from .aws_private_bank_time_comparison import compare
except ImportError:
    from aws_private_bank_time_comparison import compare


def completed_stage(pairs):
    """Pairs contain the raw-fit/selection/work triples accepted by compare.

    Require two or more distinct matched seeds. Individual wins and losses
    remain visible; a positive average does not mean every seed won.
    """
    if len(pairs) < 2: raise ValueError('Multiple completed paired seeds required')
    records = [compare(*pair) for pair in pairs]
    seeds = [r['seed'] for r in records]
    if len(set(seeds)) != len(seeds): raise ValueError('Duplicate paired seed')
    reference = pairs[0][0][0]
    settings = {k: v for k, v in reference['args'].items() if k not in ('tag', 'seed')}
    scales = [row['memory_time_scale'] for row in records[0]['rows']]
    for pair, record in zip(pairs, records):
        result = pair[0][0]
        if {k: v for k, v in result['args'].items() if k not in ('tag', 'seed')} != settings:
            raise ValueError('Across-seed fitting protocol differs')
        if result['identity'] != reference['identity'] or result['source_sha256'] != reference['source_sha256']:
            raise ValueError('Across-seed source/data protocol differs')
        if [row['memory_time_scale'] for row in record['rows']] != scales:
            raise ValueError('Different scales across pairs')
    summaries = []
    for arm in (0, 1):
        rows = [r['rows'][arm] for r in records]
        populations = {(r['fitting_targets'], r['dev_targets']) for r in rows}
        if len(populations) != 1: raise ValueError('Across-seed target populations differ')
        summaries.append(dict(memory_time_scale=scales[arm], seeds=seeds,
            mean_selected_dev_nll=sum(r['selected_dev_nll'] for r in rows) / len(rows),
            mean_whole_fit_gflops=sum(r['whole_fit_gflops'] for r in rows) / len(rows),
            mean_fitting_mflops_per_target=sum(r['fitting_mflops_per_target'] for r in rows) / len(rows),
            mean_inference_mflops_per_target=sum(r['inference_mflops_per_target'] for r in rows) / len(rows),
            fitting_targets_per_seed=rows[0]['fitting_targets'], dev_targets_per_seed=rows[0]['dev_targets']))
    gain = summaries[0]['mean_selected_dev_nll'] - summaries[1]['mean_selected_dev_nll']
    return dict(status='completed', pairs=records, summaries=summaries,
        mean_dev_nll_gain=gain, mean_quality_verdict='win' if gain > 0 else 'loss' if gain < 0 else 'tie',
        every_seed_quality_win=all(r['dev_nll_gain'] > 0 for r in records),
        every_seed_pareto_win=all(r['pareto_win'] for r in records),
        scope='Completed matched development pairs across distinct seeds, same data/source/settings/targets. Full fitting arithmetic charged in identical units. Arithmetic and special functions separate; sampling unquantified. Average and per-seed verdicts differ explicitly. Utility and public Transformer evidence require separate completed protocols.')
