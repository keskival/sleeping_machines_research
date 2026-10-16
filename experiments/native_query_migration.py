"""Migrate an acquired packed native learner without refitting shared state."""
import copy
import torch
from sleeping_machines.queryable_local_deep_race import QueryableLocalDeepRace
from sleeping_machines.packed_queryable_local_deep_race import PackedQueryableLocalDeepRace


def migrate(model,optimizer,state,*,publication_capacity=16):
    old=model.core
    if hasattr(old,'memory_queries'):raise ValueError('already queryable')
    if not hasattr(old,'banks'):raise ValueError('packed native parent required')
    named=dict(model.named_parameters())
    moments={name:copy.deepcopy(optimizer.state[p]) for name,p in named.items() if p in optimizer.state}
    groups=[(dict((k,copy.deepcopy(v)) for k,v in group.items() if k!='params'),
        [next(name for name,p in named.items() if p is param) for param in group['params']]) for group in optimizer.param_groups]
    # New nonzero operators deliberately change the function. Shared native
    # parameters are restored exactly; their optimizer clocks must not restart.
    reference=QueryableLocalDeepRace(width=old.width,depth=old.depth,heads=old.heads,pool=old.pool,
        publication_capacity=publication_capacity).to(dtype=old.mix.dtype,device=old.mix.device)
    core=PackedQueryableLocalDeepRace(reference)
    missing=core.load_state_dict(old.state_dict(),strict=False)
    assert not missing.unexpected_keys and all(name.startswith(('memory_queries.','memory_keys.','assemblers.')) for name in missing.missing_keys)
    model.core=core;current=dict(model.named_parameters());new_names=set(current)-set(named)
    assert new_names and all(name.startswith(('core.memory_queries.','core.memory_keys.','core.assemblers.')) for name in new_names)
    params=[]
    for settings,names in groups:params.append(dict(settings,params=[current[name] for name in names]))
    # New parameters use the existing first group's explicit hyperparameters.
    params[0]['params']+= [p for name,p in current.items() if name in new_names]
    updated=torch.optim.AdamW(params)
    for name,value in moments.items():updated.state[current[name]]=value
    for name,p in named.items():assert torch.equal(p,current[name])
    migrated=core.new_state();migrated.update(state)
    assert migrated['layers'] is state['layers'] and migrated['cause_bank'] is state['cause_bank']
    return model,updated,migrated,dict(shared_parameters=len(named),new_parameters=sorted(new_names),
        restored_optimizer_banks=len(moments),parent_events=state['events'],
        history='All receiver layers/eligibility,episodic bank,cause inbox and next-input constraint retained. New query-publication roots start empty;no invented old publications or corpus replay.',
        scope='Shared parameters and Adam moments preserved;new nonzero query/joint maps change predictions. Changing-parameter eligibility remains approximate;not prediction-preserving growth.')
