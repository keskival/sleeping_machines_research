"""Fail closed on a silently restricted non-stationary comparison."""
def validate(contract):
    required={'battle','decision','distribution','arms','resource_boundary'}
    missing=required-set(contract)
    if missing:raise ValueError('Missing capability declaration: '+str(sorted(missing)))
    changing=contract['distribution'] in ('nonstationary','chronological_stream')
    if not contract['arms']:raise ValueError('Comparison arms required')
    for name,arm in contract['arms'].items():
        for field in ('online_weight_updates','state_policy','retained_mechanisms','restrictions'):
            if field not in arm:raise ValueError(name+' lacks '+field)
        if changing and not arm['online_weight_updates']:raise ValueError(name+' disables online adaptation in the primary stream comparison')
        if not arm['state_policy'] or not arm['retained_mechanisms']:raise ValueError(name+' must declare actual mechanisms and state')
        for restriction in arm['restrictions']:
            if not restriction.get('reason') or not restriction.get('effect'):raise ValueError(name+' has unexplained restriction')
    if changing and contract.get('forecast_order')!='issue_then_reveal_then_update':raise ValueError('Prequential ordering required')
    return True
