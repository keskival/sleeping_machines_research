"""R1 production-world to finite-horizon receipt integration.

Horizon outcomes are per-time-position losses averaged over registered lanes;
physical scoring and own-gradient targets retain the full lane multiplier.
"""
import hashlib
import torch
from nested_horizon_audit import NestedHorizonAudit, PairedOutcome


def record_native_horizon(reference, intervention, receipt, level, weights,
                          chunks, *, initial_force=None):
    """Advance owned paired clones at original full-window boundaries.

    Callers must provide disposable issued worlds, never the live actor. This
    API records fixed corpus futures and does not handle endogenous observations.
    """
    if reference is intervention or reference.model is intervention.model:
        raise ValueError('Independent owned learner worlds required')
    if reference.next_step != intervention.next_step or vars(reference.args)!=vars(intervention.args):
        raise ValueError('Paired future policies must match')
    audit=NestedHorizonAudit(receipt,level,weights)
    work=[0,0,0];windows=0
    for inputs,targets in chunks:
        if audit.count==audit.upper_horizon:break
        n=inputs.shape[1]
        if audit.count+n>audit.upper_horizon:
            raise ValueError('Registered horizon must end on an original update boundary')
        losses=[]
        for world,force in ((reference,None),(intervention,initial_force if windows==0 else None)):
            before=tuple(world.teacher_work)
            losses.append(world.advance(inputs,targets,force_site=force))
            work=[v+b-a for v,a,b in zip(work,before,world.teacher_work)]
        for j in range(n):
            identity=hashlib.sha256(torch.stack((inputs[:,j],targets[:,j])).cpu().numpy().tobytes()).hexdigest()
            audit.append(PairedOutcome(audit.count+1,identity,identity,
                float(losses[0][:,j].mean()),float(losses[1][:,j].mean()),
                receipt.objective_id,receipt.world_id,receipt.learner_version))
        windows+=1
    target=audit.target()
    return dict(target=target,lower_value=audit.lower_value,upper_value=audit.total,
                production_windows_per_world=windows,executed_updates=2*windows,
                factual_scored_targets=2*audit.count*reference.args.lanes,
                own_gradient_targets=2*audit.count*reference.args.lanes,
                teacher_calls=work[0],teacher_replayed_targets=work[1],teacher_scored_targets=work[2])
