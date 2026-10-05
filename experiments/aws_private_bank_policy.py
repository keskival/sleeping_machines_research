"""Outcome-independent replica initialization policy; no model execution."""
import math


def replica_policy(canonical_pool, target_pool):
    if type(canonical_pool) is not int or type(target_pool) is not int:
        raise ValueError('Integer pool sizes required')
    if canonical_pool < 2 or target_pool < canonical_pool or target_pool % canonical_pool:
        raise ValueError('Target must be a positive multiple of the canonical pool')
    replicas = target_pool // canonical_pool
    return dict(version=1, canonical_pool=canonical_pool, target_pool=target_pool,
                replicas=replicas, receiver_order='canonical-major',
                clock_bias_shift=-math.log(replicas),
                shared_fields=['input','output','gate_w','gate_b','control_w','control_b'],
                private_fields=['key','key_read','clock_bias','raw_rate','frequency'],
                initial_law_scope='Zero memory, temperature one, neither score clipping boundary crossed')


def grouped_initial_rates(scores, policy):
    """Check the actual clipped race law; fail rather than assert false equivalence."""
    if len(scores) != policy['canonical_pool'] or any(not math.isfinite(x) for x in scores):
        raise ValueError('Finite canonical scores required')
    shift = policy['clock_bias_shift']
    if any(not (-12 < x < 12 and -12 < x+shift < 12) for x in scores):
        raise ValueError('Clipping invalidates replica race equivalence')
    return [policy['replicas'] * math.exp(x+shift) for x in scores]
