"""Shared forward gradients teach approximate alternative Adam trajectories."""
from copy import deepcopy
import torch
from token_forward_credit import bonuses,projected_costs
from token_issued_credit_geometry import route_mean_shifts
from token_block_curvature import block_basis,block_action,block_covariances


def replay_transport(p,receivers,actions,lr,curvature='spectral',diagnostics=False,initial_learner=None,spectral_cache=None):
    if curvature not in ['zero','diagonal','spectral']:raise ValueError('Unknown transport')
    if curvature=='spectral' and spectral_cache is None:spectral_cache=block_basis(p.embedding,min(8,p.embedding.shape[1]))
    beta=bonuses(p,actions);r=torch.tensor(receivers,dtype=torch.long);W=len(r);D=p.embedding.shape[1]
    def learners(snapshot=None):
        value=p.embedding.new_zeros(D) if snapshot is None else snapshot['z'].detach()
        if value.shape!=(D,):raise ValueError('Incoming fast state shape')
        reference=torch.nn.Parameter(value.clone())
        alternatives=torch.nn.Parameter(value[None].expand(W,D).clone())
        refopt=torch.optim.Adam([reference],lr=lr)
        altopt=torch.optim.Adam([alternatives],lr=lr)
        if snapshot is not None:
            state=deepcopy(snapshot['optimizer'])
            if len(state['param_groups'])!=1 or state['param_groups'][0]['lr']!=lr:
                raise ValueError('Incoming optimizer must match issued learning rule')
            refopt.load_state_dict(state)
            alternative_state=deepcopy(snapshot['optimizer'])
            for fields in alternative_state['state'].values():
                for name,v in fields.items():
                    if isinstance(v,torch.Tensor) and v.ndim:
                        if v.shape!=(D,):raise ValueError('Incoming optimizer tensor shape')
                        fields[name]=v[None].expand(W,D).clone()
            altopt.load_state_dict(alternative_state)
        return reference,refopt,alternatives,altopt
    zb,ob,za,oa=learners(initial_learner);predictions=[];reference=[]
    for j,target in enumerate(p.targets.tolist()):
        if bool(p.reset[j]):zb,ob,za,oa=learners()
        with torch.no_grad():
            logits=p.log_probabilities[j]+p.embedding@zb
            probability=logits.softmax(0)
            shift,dp,mean=route_mean_shifts(logits,p.embedding,receivers,beta[:,j])
            g=mean-p.embedding[target];ga=g[None]+shift
            displacement=za-zb[None]
            costs=projected_costs(logits,receivers,beta[:,j],target)
            predictions.append(costs+(ga*displacement).sum(1))
            reference.append(-logits.log_softmax(0)[target])
            if curvature in ['diagonal','spectral']:
                second=probability@p.embedding.square()
                pr=probability[r]
                second_except=(second[None]-pr[:,None]*p.embedding[r].square())/(1-pr)[:,None]
                second_a=second[None]+dp[:,None]*(p.embedding[r].square()-second_except)
                diagonal=(second_a-(mean[None]+shift).square()).clamp_min(0)
                if curvature=='spectral':
                    covariance=block_covariances(probability,receivers,dp,spectral_cache)
                    ga=ga+block_action(diagonal,covariance,spectral_cache['u'],displacement)
                else:ga=ga+diagonal*displacement
            zb.grad=g.clone();za.grad=ga.clone()
        ob.step();oa.step()
    if p.final_eos:zb,ob,za,oa=learners()
    result=dict(scores=torch.stack(predictions,1),reference_scores=torch.stack(reference),z=za.detach().clone(),reference_z=zb.detach().clone(),actual_worlds=1,transported_worlds=W,reference_updates=len(p.targets),transported_updates=W*len(p.targets))
    result['reference_learner']=dict(z=zb.detach().clone(),optimizer=deepcopy(ob.state_dict()))
    if diagnostics:result.update(reference_optimizer=ob.state_dict(),alternative_optimizer=oa.state_dict())
    return result
