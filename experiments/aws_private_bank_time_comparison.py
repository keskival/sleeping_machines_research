"""Completed positive-time scale quality/work comparison; stdlib only."""
import math

try:
    from .aws_token_stage_evidence import stage_row
    from .aws_private_bank_comparison import EXACT_FIELDS
except ImportError:
    from aws_token_stage_evidence import stage_row
    from aws_private_bank_comparison import EXACT_FIELDS


def compare(control, scaled):
    """Each arm is (completed raw fit, selection, complete work receipt).

    One paired seed; actual entire fitting exposure is charged. This neither
    admits training nor selects a public Transformer reference.
    """
    rows = []
    for result, selection, work in (control, scaled):
        if work is None: raise ValueError('Complete measured work required')
        row = stage_row(result, selection, work)
        scale = result.get('memory_time_scale')
        if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError('Positive declared memory time scale required')
        recipe = result.get('bank_recipe', {})
        if recipe.get('memory_time_scale') != scale or work.get('memory_time_scale') != scale:
            raise ValueError('Fit/work time scale differs')
        if work.get('bank_recipe') != recipe or not recipe.get('time_source_sha256'):
            raise ValueError('Source-bound scale recipe required')
        if work.get('final_numeric_training_state_exact') is not True:
            raise ValueError('Exact full numerical replay required')
        if set(work.get('exact_checkpoint_fields', [])) != EXACT_FIELDS:
            raise ValueError('Full model/optimizer/state/RNG replay required')
        sources = result.get('source_sha256', {})
        if not sources or any(work.get('source_sha256', {}).get(k) != v for k, v in sources.items()):
            raise ValueError('Fit/work source binding differs')
        row['memory_time_scale'] = scale
        rows.append(row)
    a, b = rows
    if a['memory_time_scale'] != 1 or b['memory_time_scale'] == 1:
        raise ValueError('Scale-one control and changed positive scale required')
    left, right = control[0], scaled[0]
    if {k: v for k, v in left['args'].items() if k != 'tag'} != {
            k: v for k, v in right['args'].items() if k != 'tag'}:
        raise ValueError('Paired fitting settings required')
    if left['identity'] != right['identity'] or left['source_sha256'] != right['source_sha256']:
        raise ValueError('Paired full data/source identity required')
    recipes = [{k: v for k, v in r['bank_recipe'].items() if k != 'memory_time_scale'}
               for r in (left, right)]
    if recipes[0] != recipes[1]: raise ValueError('Only temporal scale may change')
    if a['fitting_targets'] != b['fitting_targets']:
        raise ValueError('Equal complete fitting exposure required')
    gain = a['selected_dev_nll'] - b['selected_dev_nll']
    fit_ratio = b['whole_fit_gflops'] / a['whole_fit_gflops']
    inference_ratio = b['inference_mflops_per_target'] / a['inference_mflops_per_target']
    return dict(status='completed', rows=rows, seed=a['seed'], dev_nll_gain=gain,
        quality_verdict='win' if gain > 0 else 'loss' if gain < 0 else 'tie',
        whole_fit_work_ratio=fit_ratio, inference_work_ratio=inference_ratio,
        pareto_win=gain > 0 and fit_ratio <= 1 and inference_ratio <= 1,
        scope='One source/data/settings-matched seed, completed GPT-2/FineWeb development fits. Entire fitting work charged, selected-checkpoint inference on identical targets. Arithmetic and special functions separate; sampling work unquantified. No independent-seed or public-reference claim; memory usefulness requires its own matched intervention evidence.')
