"""Completed data-growth fits; absent and running fits never supply scores."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = 'curie_data_growth_tokens_64k_b64_c16_p16_s6_20261005_v1'


def pages():
    folder = ROOT / 'experiments/results/token_language'
    path = folder / (TAG + '.json')
    if not path.exists():
        return []
    result = json.loads(path.read_text())
    if result['status'] != 'completed':
        return []
    selection = json.loads(path.with_suffix('.selection.json').read_text())
    if selection['status'] != 'completed':
        raise ValueError('64K selection incomplete')
    initial = result['curve'][0]['dev_nll']
    selected = selection['selected']['dev_nll']
    rows = [['16', '6', str(result['presentations_total']),
             f'{initial:.6f}', f'{selected:.6f}', f'{initial-selected:.6f}',
             str(selection['selected']['step'])]]
    page = [
        ('h1', 'Appendix. Tokenized language: reserved 64K data exposure'),
        ('p', 'Sleeping Machines pursues a general-purpose substrate for language and reasoning, multimodal world models, embodiment, event-native analytics, continual learning, communication, self-design and hardware. This stage measures integrated tokenized learning with more data.'),
        ('p', 'Single seed, development only. GPT-2 FineWeb: 65,536 admitted training tokens, two passes, 256 updates, 2,040 scored development targets. P16/D2/H2/U4, batch64, credit16 and uniform-site K4 actual alternative-write credit; all temporal, sparse and persistent-state mechanisms retained. Initialization is eligible for selection, with evaluation every64 updates. Public validation untouched.'),
        ('table', (['Payload', 'Seed', 'Fit targets', 'Initial NLL', 'Selected NLL', 'Gain', 'Selected step'], rows, [50, 40, 85, 95, 95, 85, 75])),
        ('p', f"Measured ordinary fitting throughput: {result['train_tokens_per_second']:.2f} targets/s; peak RSS: {result['max_rss_kb']} KiB. The learning gate requires a 0.02 NLL improvement over initialization."),
        ('p', 'The 8K and 64K fits score the same development population and both use two passes. Their train-frequency priors and evaluation cadence differ; report absolute loss and within-fit learning separately. This single width/data step supplies no fitted scaling exponent or matched-compute Transformer win.')]
    utility_path = ROOT / 'experiments/results/diagnostics/curie_data_growth_tokens_64k_utility_20261005_v1.json'
    if utility_path.exists():
        utility = json.loads(utility_path.read_text())
        if utility['status'] != 'completed':
            raise ValueError('64K utility incomplete')
        row = utility['rows'][0]
        if row['tag'] != TAG:
            raise ValueError('64K utility tag mismatch')
        page.append(('p', f"Frozen selected-checkpoint context gain: {row['context_gain']:.6f} NLL. History interventions: {json.dumps(row['history_gains'], sort_keys=True)}. These matched-RNG interventions use the same frozen readout; the constant TRAIN-mean feature control is not an optimally refitted unigram, and erasures are not retrained ablations."))
    else:
        page.append(('p', 'Selected-checkpoint context and state utility diagnostic pending; no utility value is predicted.'))
    return [page]
