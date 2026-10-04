"""Completed public results, including negative evidence and resource scope."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pages():
    rows = []
    research_work = 0
    for path in (ROOT / 'experiments/results/public_benchmarks').glob('*.json'):
        r = json.loads(path.read_text())
        if r.get('status') == 'completed':
            research_work += r.get('work', {}).get('whole_fit_flops_estimate', 0)
    for dataset in ('ECG200', 'JapaneseVowels', 'PenDigits'):
        paths = sorted((ROOT / 'experiments/results/public_benchmarks').glob(
            f'aws_public_{dataset}_selection_20261004T000100Z_final_s*.json'))
        if len(paths) != 3:
            continue
        records = [json.loads(p.read_text()) for p in paths]
        assert all(r['status'] == 'completed' and r['args']['stage'] == 'final' for r in records)
        scores = [r['test']['accuracy'] * 100 for r in records]
        rows.append([dataset, f'{statistics.mean(scores):.2f}',
                     f'{statistics.stdev(scores):.2f}',
                     f"{statistics.mean(r['work']['whole_fit_flops_estimate'] for r in records)/1e9:,.2f}",
                     f"{statistics.mean(r['work']['fit_flops_per_presented_target_estimate'] for r in records)/1e6:.3f}"])
    if not rows:
        return []
    return [[('h1', 'Completed public archive benchmark campaign'),
             ('p', '<b>No public benchmark win is established by this campaign.</b> '
              'Three fixed TRAIN-only development screens per dataset selected minimum DEV negative log likelihood. '
              'Selected configurations were refitted on full official TRAIN at the selected epoch budget, '
              'then scored once on official TEST for each seed6/7/8. All nine refits completed; '
              'negative results and their checkpoints remain preserved.'),
             ('table', (['Dataset', 'Test %', 'Seed SD', 'Fit GF/run', 'Fit MF/series'], rows,
                        [40, 25, 25, 39, 40])),
             ('p', 'ECG200 and JapaneseVowels selected payload32/depth4/pool4 with two heads, '
              'at13 and22 epochs respectively; PenDigits selected payload16/depth2/pool2 at36 epochs. '
              'These models use temporal races, separate keys/values, sparse addressed private-state writes '
              'and linear local message route credit. Full counterfactual suffix replay is absent. '
              'Inputs use ordinal synchronous steps; PenDigits coordinates are spatial resampling, not physical event times.'),
             ('p', 'Seed SD measures variation across training seeds on the same TEST examples, not a generalization '
              'confidence interval. Matched nearest-neighbor controls were DEV-only and do not supply matched TEST '
              'or resource comparisons. Published protocol variants require verification before frontier comparisons. '
              'Inference work and energy remain unmeasured; these fitting estimates establish no resource advantage.'),
             ('small', f'Fit GF/run is mean whole-fit work per final seed; MF/series divides by fitting presentations, '
              f'not unique examples. Completed pilots, screens and final refits together account for '
              f'{research_work/1e9:,.2f}GF estimated fitting work. Failed/contract/evaluation/compilation/preprocessing '
              'work is additional. Estimates extrapolate one eager optimizer window per epoch; variable padding '
              'prevents exact whole-fit accounting. Sources: experiments/results/public_benchmarks/*.json.')]]
