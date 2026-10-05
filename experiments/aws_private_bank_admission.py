"""Read-only prerequisites for guarded private-bank smoke fits."""


def validate_contract(record, expected_sources):
    if record.get('status') != 'completed': raise ValueError('Construction contract incomplete')
    for field in ('construction_rng_exact','canonical_parameter_parity','shared_rule_parity',
                  'private_replica_storage','cross_recipe_checkpoint_rejected','factual_gradient_finite'):
        if record.get(field) is not True: raise ValueError('Contract failed: '+field)
    if record.get('source_sha256') != expected_sources: raise ValueError('Contract source binding mismatch')
    rows=record.get('rows',[])
    if [r.get('pool') for r in rows] != [4,16]: raise ValueError('Missing paired bank contracts')
    if any(r.get('selected_writes') != 32 for r in rows): raise ValueError('Selected activity changed')
    return True
