"""Preserve exact versus lower-bound process work in downstream scoring."""
import math


def acquisition_work(packet):
    targets = packet['presentations_total']
    if targets <= 0:
        raise ValueError('Positive acquired target count required')
    result = {'interruption_work_scope': packet.get('whole_fit_work_scope',
              'Single process including setup, fitting, checkpointing and DEV evaluation.')}
    if 'whole_fit_process_wall_lower_bound_s' in packet:
        seconds = packet['whole_fit_process_wall_lower_bound_s']
        key = 'whole_fit_process_wall_lower_bound_s'
        per_target = 'whole_fit_process_lower_bound_s_per_target'
        result['process_work_precision'] = 'lower_bound_unknown_interruption_tail'
    else:
        seconds = packet.get('whole_fit_process_wall_s', packet['process_wall_s'])
        key = 'whole_fit_process_wall_s'
        per_target = 'whole_fit_process_s_per_target'
        result['process_work_precision'] = 'recorded_process_intervals'
    if not math.isfinite(seconds) or seconds < packet['process_wall_s']:
        raise ValueError('Whole-fit interval must include the completed process')
    result[key] = seconds
    result[per_target] = seconds / targets
    return result
