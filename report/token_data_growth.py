"""Completed data-growth fits; absent and running fits never supply scores."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pages():
    folder = ROOT / 'experiments/results/token_language'
    rows, resources, utilities, work, completed = [], [], [], [], []
    frozen = None
    for width in (16, 24, 32):
        tag = f'curie_data_growth_tokens_64k_b64_c16_p{width}_s6_20261005_v1'
        path = folder / (tag + '.json')
        selection_path = path.with_suffix('.selection.json')
        if not path.exists() or not selection_path.exists():
            continue
        result = json.loads(path.read_text())
        selection = json.loads(selection_path.read_text())
        if result['status'] != 'completed' or selection['status'] != 'completed':
            continue
        protocol = {k: v for k, v in result['args'].items() if k not in ('payload', 'tag')}
        if frozen is None:
            frozen = protocol
        elif protocol != frozen:
            raise ValueError('Capacity comparison protocol changed')
        if result['presentations_total'] != 131056:
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
        work_path = ROOT / f'experiments/results/diagnostics/curie_data_growth_tokens_64k_p{width}_work_20261005_v1.json'
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
        suffix = '' if width == 16 else f'_p{width}'
        utility_path = ROOT / f'experiments/results/diagnostics/curie_data_growth_tokens_64k{suffix}_utility_20261005_v1.json'
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
    page = [
        ('h1', 'Appendix. Tokenized language: reserved 64K capacity comparisons'),
        ('p', 'Sleeping Machines pursues a general-purpose substrate for language and reasoning, multimodal world models, embodiment, event-native analytics, continual learning, communication, self-design and hardware. This stage measures integrated tokenized learning with more data and capacity.'),
        ('p', 'Seed6, development only. GPT-2 FineWeb: 65,536 admitted training tokens, two passes/131,056 fitting targets, 256 updates, 2,040 scored development targets. Payload varies; depth2/heads2/pool4, batch64, credit16 and uniform-site K4 actual alternative-write credit are fixed. Every temporal, sparse and persistent-state mechanism is retained. Initialization is eligible for selection, with evaluation every64 updates. Public validation untouched.'),
        ('table', (['Payload', 'Initial NLL', 'Selected NLL', 'Gain', 'Selected step'], rows, [60, 115, 115, 110, 105])),
        ('table', (['Payload', 'All parameters', 'Core + input parameters', 'Readout parameters', 'Targets/s', 'RSS KiB'], resources, [45, 110, 105, 120, 85, 80])),
        ('table', (['Payload', 'Fit targets', 'Whole-fit GFLOPs', 'Fit MFLOPs/target', 'Eval MFLOPs/target'], work, [45, 90, 125, 125, 125])),
        ('p', 'Work cells require a complete replay with trajectory parity and full arithmetic formula coverage. Fitting includes discovery, actual alternative-write replay, readout, backward, clipping, optimizer and in-step diagnostics; preprocessing, evaluation and serialization are separate. Special functions are counted separately, random sampling work remains unquantified. Pending cells contain no extrapolation from a smaller fit.'),
        ('p', 'All parameters include the token interface and readout; the core/input column includes lexical input parameters. Selected activity remains four writes and sixteen scored keys per token, while vector width grows. Equal data and passes are not equal fitting FLOPs. These ordinary throughput measurements exclude instrumented arithmetic tracing.'),
        ('p', 'The 8K and 64K fits score the same development population and both use two passes. Train-frequency priors and evaluation cadence differ; report absolute loss and within-fit learning separately. The learning gate requires a 0.02 NLL improvement over initialization. No scaling exponent or matched-compute Transformer win is inferred from these cells.')]
    if utilities:
        page += [
            ('table', (['Payload', 'Context gain', 'Memory erase delta', 'Message erase delta', 'Both erase delta'], utilities, [55, 105, 130, 130, 130])),
            ('p', 'Context gain is constant TRAIN-mean feature NLL minus intact NLL through the same frozen readout. Erasure deltas are intervention NLL minus intact NLL: positive means erasure hurts prediction, negative means it helps. Source, matched-RNG and partition checks pass. The constant-feature control is not an optimally refitted unigram; frozen erasures change routing and readout features and are not retrained architecture comparisons.')]
    missing = set(completed) - {int(r[0]) for r in utilities}
    if missing:
        page.append(('p', 'Selected-checkpoint utility pending for payload ' + ', '.join(map(str, sorted(missing))) + '; no utility value is predicted.'))
    return [page]
