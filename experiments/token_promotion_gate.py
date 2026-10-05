"""Evidence-driven development selection; no predicted scaling scores."""
import math


def assess(record, minimum_gain=.02):
    if record.get('status') != 'completed':
        raise ValueError('Completed trajectory required')
    curve = record['curve']
    if not curve or curve[0]['step'] != 0:
        raise ValueError('Initialization must be scored')
    if any(not math.isfinite(row['dev_nll']) for row in curve):
        raise ValueError('Finite development loss required')
    selected = min(curve, key=lambda row: (row['dev_nll'], row['step']))
    initial = curve[0]['dev_nll']
    gain = initial - selected['dev_nll']
    return dict(selected_step=selected['step'], selected_dev_nll=selected['dev_nll'],
                initial_dev_nll=initial, contextual_learning_gain=gain,
                small_fit_promotable=selected['step'] > 0 and gain >= minimum_gain,
                practical_gain_threshold=minimum_gain,
                scaling_ready=False,
                remaining_gates=['independent seed', 'complete resource accounting',
                                 'larger-data predictive-capacity comparison'])
