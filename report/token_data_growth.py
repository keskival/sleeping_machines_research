"""Completed data-growth fits; absent and running fits never supply scores."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pages():
    return measured_pages() + stage_pages(1048576, '1m') + stage_pages(262144, '256k') + stage_pages(65536, '64k') + repeat_pages() + message_factor_pages()


def stage_pages(budget, label):
    expected_targets = 2 * (budget - 8)
    folder = ROOT / 'experiments/results/token_language'
    rows, resources, utilities, work, completed = [], [], [], [], []
    frozen = None
    for width in (16, 24, 32):
        tag = f'curie_data_growth_tokens_{label}_b64_c16_p{width}_s6_20261005_v1'
        path = folder / (tag + '.json')
        selection_path = path.with_suffix('.selection.json')
        if not path.exists() or not selection_path.exists():
            continue
        result = json.loads(path.read_text())
        selection = json.loads(selection_path.read_text())
        if result['status'] != 'completed' or selection['status'] != 'completed':
            continue
        if result['args']['train_tokens'] != budget or result['args']['seed'] != 6 or result['args']['payload'] != width:
            raise ValueError('Data-stage identity mismatch')
        protocol = {k: v for k, v in result['args'].items() if k not in ('payload', 'tag')}
        if frozen is None:
            frozen = protocol
        elif protocol != frozen:
            raise ValueError('Capacity comparison protocol changed')
        if result['presentations_total'] != expected_targets:
            raise ValueError('Incomplete two-pass fit')
        initial = result['curve'][0]['dev_nll']
        selected = selection['selected']['dev_nll']
        expected = min(r['dev_nll'] for r in result['curve'])
        if abs(selected - expected) > 2e-6:
            raise ValueError('Initial-inclusive selection mismatch')
        rows.append([str(width), f'{initial:.6f}', f'{selected:.6f}',
                     f'{initial-selected:.6f}', str(selection['selected']['step'])])
        resources.append([str(width), str(result['parameters']),
                          str(result['core_parameters']), str(result['readout_parameters']),
                          f"{result['train_tokens_per_second']:.2f}", str(result['max_rss_kb'])])
        completed.append(width)
        work_path = ROOT / f'experiments/results/diagnostics/curie_data_growth_tokens_{label}_p{width}_work_20261005_v1.json'
        if work_path.exists():
            audit = json.loads(work_path.read_text())
            if audit['status'] != 'completed' or audit['fitting_targets'] != result['presentations_total']:
                raise ValueError('Incomplete 64K work audit')
            if Path(audit['control']).name != path.name or audit['curve_parity_max_error'] >= 2e-6:
                raise ValueError('64K work control or trajectory mismatch')
            if not all(audit[k]['formula_coverage_complete'] for k in ('fitting', 'inference')):
                raise ValueError('Unknown 64K arithmetic')
            work.append([str(width), str(audit['fitting_targets']),
                         f"{audit['fitting']['arithmetic_flops']/1e9:.6f}",
                         f"{audit['fitting_arithmetic_flops_per_target']/1e6:.6f}",
                         f"{audit['inference_arithmetic_flops_per_target']/1e6:.6f}"])
        else:
            work.append([str(width), str(result['presentations_total']), 'pending', 'pending', 'pending'])
        suffix = '' if label == '64k' and width == 16 else f'_p{width}'
        utility_path = ROOT / f'experiments/results/diagnostics/curie_data_growth_tokens_{label}{suffix}_utility_20261005_v1.json'
        if label == '1m':
            utility_path = ROOT / ('experiments/results/diagnostics/' +
                ('curie_original_1m_utility_20261006_v1.json' if width == 24 else
                 f'curie_original_1m_p{width}_utility_20261006_v1.json'))
        if utility_path.exists():
            utility = json.loads(utility_path.read_text())
            if utility['status'] != 'completed':
                continue
            row = utility['rows'][0]
            if row['tag'] != tag or row['selected'] != selection['selected']:
                raise ValueError('Utility selection mismatch')
            utilities.append([str(width), f"{row['context_gain']:.6f}",
                              *[f"{row['history_gains'][k]:.6f}" for k in ('memory', 'message', 'both')]])
    if not rows:
        return []
    bound_path = ROOT / 'experiments/results/diagnostics/modded_nanogpt_head_work_bound_20261005_v1.json'
    if bound_path.exists():
        bound = json.loads(bound_path.read_text())
        if bound['status'] != 'completed_source_bound_lower_bound' or bound['complete_reference_work']:
            raise ValueError('Invalid reference lower bound')
        work.append(['TF head ≥', str(bound['fitting_targets']),
                     f"≥{bound['fitting_head_arithmetic_flops']/1e9:.6f}",
                     f"≥{bound['fitting_head_arithmetic_flops_per_target']/1e6:.6f}",
                     f"≥{bound['inference_head_arithmetic_flops_per_target']/1e6:.6f}"])
    page = [
        ('h1', f'Appendix. Tokenized language: {label.upper()} capacity/data comparisons'),
        ('p', 'Sleeping Machines pursues a general-purpose substrate for language and reasoning, multimodal world models, embodiment, event-native analytics, continual learning, communication, self-design and hardware. This stage measures integrated tokenized learning with more data and capacity.'),
        ('p', f'Seed6, development only. GPT-2 FineWeb: {budget:,} admitted training tokens, two passes/{expected_targets:,} fitting targets, {result["args"]["steps"]} updates, 2,040 scored development targets. Payload varies; depth2/heads2/pool4, batch64, credit16 and uniform-site K4 actual alternative-write credit are fixed. Every temporal, sparse and persistent-state mechanism is retained. Initialization is eligible for selection, with four evaluation checkpoints over the two passes. Public validation untouched.'),
        ('table', (['Payload', 'Initial NLL', 'Selected NLL', 'Gain', 'Selected step'], rows, [60, 115, 115, 110, 105])),
        ('table', (['Payload', 'All parameters', 'Core + input parameters', 'Readout parameters', 'Targets/s', 'RSS KiB'], resources, [45, 110, 105, 120, 85, 80])),
        ('table', (['Member', 'Fit targets', 'Whole-fit GFLOPs', 'Fit MFLOPs/target', 'Eval MFLOPs/target'], work, [65, 85, 120, 120, 120])),
        ('p', 'TF head ≥ is a source-derived arithmetic lower bound: forward output projection plus its two explicit backward matrix contractions; evaluation has one projection. Width768 and padded50,304classes give231.800832MFLOPs/fitting target and77.266944MFLOPs/evaluation target, at two FLOPs per multiply-add. Other reference computation and optimizer work are excluded. Its695,992,320training targets and public NLL3.2774 use a different data/quality population from these native development fits; these columns show raw work, not a matched-quality or iso-FLOP win. FP8 GPU arithmetic and native CPUfloat32 have different hardware costs.'),
        ('p', 'Work cells require a complete replay with trajectory parity and full arithmetic formula coverage. Fitting includes discovery, actual alternative-write replay, readout, backward, clipping, optimizer and in-step diagnostics; preprocessing, evaluation and serialization are separate. Special functions are counted separately, random sampling work remains unquantified. Pending cells contain no extrapolation from a smaller fit.'),
        ('p', 'All parameters include the token interface and readout; the core/input column includes lexical input parameters. Selected activity remains four writes and sixteen scored keys per token, while vector width grows. Equal data and passes are not equal fitting FLOPs. These ordinary throughput measurements exclude instrumented arithmetic tracing.'),
        ('p', 'The 8K, 64K, 256K and 1M stages score the same development population and use two passes. Train-frequency priors and evaluation cadence differ; report absolute loss and within-fit learning separately. The learning gate requires a 0.02 NLL improvement over initialization. No scaling exponent or matched-compute Transformer win is inferred from these cells.')]
    if utilities:
        page += [
            ('table', (['Payload', 'Context gain', 'Memory erase delta', 'Message erase delta', 'Both erase delta'], utilities, [55, 105, 130, 130, 130])),
            ('p', 'Context gain is constant TRAIN-mean feature NLL minus intact NLL through the same frozen readout. Erasure deltas are intervention NLL minus intact NLL: positive means erasure hurts prediction, negative means it helps. Source, matched-RNG and partition checks pass. The constant-feature control is not an optimally refitted unigram; full-message erasure removes payload, arrival metadata and presence together, changing the normalization branch and read-clock policy. Frozen erasures are not retrained architecture comparisons. The completed payload-only diagnostic is shown separately.')]
    missing = set(completed) - {int(r[0]) for r in utilities}
    if missing:
        page.append(('p', 'Selected-checkpoint utility pending for payload ' + ', '.join(map(str, sorted(missing))) + '; no utility value is predicted.'))
    return [page]


def repeat_pages():
    receipt_path = ROOT / 'experiments/queue/curie_data_growth_64k_repeat_admission_20261005_v1.json'
    if not receipt_path.exists():
        return []
    receipt = json.loads(receipt_path.read_text())
    width = receipt['selected_payload']
    rows = []
    repeat_protocol = None
    for seed in (6, 7):
        tag = f'curie_data_growth_tokens_64k_b64_c16_p{width}_s{seed}_20261005_v1'
        path = ROOT / f'experiments/results/token_language/{tag}.json'
        selection_path = path.with_suffix('.selection.json')
        if not path.exists() or not selection_path.exists():
            return []
        result = json.loads(path.read_text())
        selection = json.loads(selection_path.read_text())
        if result['status'] != 'completed' or selection['status'] != 'completed':
            return []
        if result['args']['seed'] != seed or result['args']['payload'] != width:
            raise ValueError('Repeat identity mismatch')
        if result['presentations_total'] != 131056:
            raise ValueError('Repeat data exposure mismatch')
        protocol = {k: v for k, v in result['args'].items() if k not in ('seed', 'tag')}
        if repeat_protocol is None:
            repeat_protocol = protocol
        elif repeat_protocol != protocol:
            raise ValueError('Repeat protocol changed')
        initial = result['curve'][0]['dev_nll']
        selected = selection['selected']['dev_nll']
        if abs(selected - min(x['dev_nll'] for x in result['curve'])) > 2e-6:
            raise ValueError('Repeat selection mismatch')
        rows.append([str(seed), f'{initial:.6f}', f'{selected:.6f}',
                     f'{initial-selected:.6f}', str(selection['selected']['step'])])
    if abs(float(rows[0][2]) - receipt['seed6_selected_nll']) > 2e-6:
        raise ValueError('Repeat parent selection changed')
    page = [('h1', 'Appendix. Independent seed for selected 64K capacity'),
            ('p', f'Payload{width}, selected from the three completed seed6 capacity fits. Seeds6/7 share GPT-2 FineWeb, 65,536 admitted training tokens, 131,056 fitting targets/two passes and 2,040 development targets; initialization-inclusive selection every64 updates. Public validation untouched.'),
            ('table', (['Seed', 'Initial NLL', 'Selected NLL', 'Gain', 'Selected step'], rows, [55, 115, 115, 110, 105])),
            ('p', 'This repeat measures selected-member learning reliability and seed variation. The quality gain over P16 is a seed6 capacity comparison; it is not a paired two-seed capacity win.')]
    utility_path = ROOT / f'experiments/results/diagnostics/curie_data_growth_tokens_64k_p{width}_s7_utility_20261005_v1.json'
    if utility_path.exists():
        utility = json.loads(utility_path.read_text())
        if utility['status'] != 'completed':
            raise ValueError('Incomplete repeat utility')
        row = utility['rows'][0]
        if row['tag'] != tag or row['selected'] != selection['selected']:
            raise ValueError('Repeat utility checkpoint mismatch')
        values = [[f"{row['context_gain']:.6f}",
                   *[f"{row['history_gains'][k]:.6f}" for k in ('memory', 'message', 'both')]]]
        page += [('table', (['Context gain', 'Memory erase delta', 'Message erase delta', 'Both erase delta'], values, [115, 140, 140, 140])),
                 ('p', 'Frozen interventions use the same readout and matched RNG, with intact partition parity. Positive erasure delta means the intervention hurts prediction. Full-message erasure bundles payload, arrival metadata and presence, including changes to normalization and the read clock. Constant TRAIN-mean features are not an optimally refitted unigram; no retrained ablation claim.')]
    else:
        page.append(('p', 'Independent-seed utility pending; no predicted utility is reported.'))
    return [page]


def message_factor_pages():
    path = ROOT / 'experiments/results/diagnostics/curie_token_message_factors_64k_20261005_v1.json'
    if not path.exists():
        return []
    result = json.loads(path.read_text())
    if result['status'] != 'completed':
        return []
    rows = []
    assert {row['seed'] for row in result['rows']} == {6, 7}
    for row in result['rows']:
        selection = json.loads((ROOT / ('experiments/results/token_language/' + row['tag'] + '.selection.json')).read_text())
        assert row['selected'] == selection['selected']
        assert row['matched_rng'] and row['partition_parity'] and row['scored_targets'] == 2040
        assert abs(row['scores']['intact'] - row['selected']['dev_nll']) < 2e-6
        rows.append([str(row['seed']), f"{row['scores']['intact']:.6f}", f"{row['deltas']['payload']:.6f}", f"{row['deltas']['full_message']:.6f}"])
    return [[('h1', 'Appendix. Tokenized messages: replicated payload contribution'),
        ('p', 'P24 selected64Kcheckpoints, two seeds, 2,040development targets. Erasure delta is intervention NLL minus intact NLL. Payload-only erasure hurts prediction in both seeds by0.105206/0.101270NLL; the message information path contributes under this intervention.'),
        ('table', (['Seed', 'Intact NLL', 'Payload erase delta', 'Full message erase delta'], rows, [65, 130, 160, 185])),
        ('p', result['scope'])]]


def measured_pages():
    name = 'token_language_measured_scaling_20261006_v4'
    path = ROOT / ('report/figures/' + name + '.json')
    if not path.exists():
        return []
    receipt = json.loads(path.read_text())
    assert receipt['status'] == 'completed_measured_visualization' and not receipt['curve_fitted']
    import hashlib
    for source, digest in receipt['input_sha256'].items():
        assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == digest
    return [[('h1', 'Appendix. Measured language quality, capacity and compute'),
        ('p', 'Sleeping Machines pursues a general-purpose substrate spanning language/reasoning, multimodal world models, embodiment, event-native analytics, continual learning, communication, self-design and hardware. Completed integrated language measurements show quality improving with data and a useful intermediate capacity choice.'),
        ('figure', (name, 174)),
        ('p', 'P16 selected NLL improves8.297491→8.099440 from8Kto64K TRAINtokens; P24 improves8.033311→7.741714→7.251503 from64Kto256Kto1M. The256Kto1M gain is0.490211NLL, seed6 at fixed recipe. At64K/P24 beats P16/P32 by0.066129/0.057108NLL, seed6. The independent P24seed7point is8.078991. At1M/P24 beats P32 by0.012258NLL; both select step2,048. P24 seed7 at1M selects7.263557 at finalstep4,096; two-seed mean7.257530. Each plotted point is a completed fit.'),
        ('p', receipt['scope'])]]
