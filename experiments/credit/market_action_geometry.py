"""Four-action optimizer metric; full global clip boundary retained."""
import market_actual_update_teacher as teacher
torch=teacher.torch

def parameter_metric(model,opt,grads,credit):
    def proposal(action):
        ps,_=teacher.adam_parameters(model,opt,teacher.guided(model,grads,action))
        return torch.cat([p.reshape(-1) for p in ps.values()])
    columns=[]
    for unit in torch.eye(credit.numel(),dtype=credit.dtype):
        _,column=torch.func.jvp(proposal,(credit.detach(),),(unit,));columns.append(column.detach())
    jac=torch.stack(columns,1)
    return jac.T@jac

def teaching_direction(metric,credit_gradient,scale,radius=.1):
    # Work in critic normalized output coordinates, not raw clock units.
    metric=metric*scale[:,None]*scale[None,:]
    demand=credit_gradient.detach()*scale
    damping=max(float(torch.linalg.eigvalsh(metric).max())*.01,1e-12)
    direction=-torch.linalg.solve(metric+damping*torch.eye(scale.numel(),dtype=scale.dtype),demand)
    return direction*(radius/direction.norm().clamp_min(radius))
