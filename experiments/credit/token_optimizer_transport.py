"""Shared forward gradients teach approximate alternative Adam trajectories."""
import torch
from token_forward_credit import bonuses,projected_costs
from token_issued_credit_geometry import route_mean_shifts


def replay_transport(p,receivers,actions,lr,curvature='zero',diagnostics=False):
    if curvature not in ['zero','diagonal']:raise ValueError('Unknown transport')
    beta=bonuses(p,actions);r=torch.tensor(receivers,dtype=torch.long);W=len(r);D=p.embedding.shape[1]
    def learners():
        reference=torch.nn.Parameter(p.embedding.new_zeros(D))
        alternatives=torch.nn.Parameter(p.embedding.new_zeros(W,D))
        return reference,torch.optim.Adam([reference],lr=lr),alternatives,torch.optim.Adam([alternatives],lr=lr)
    zb,ob,za,oa=learners();predictions=[];reference=[]
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
            if curvature=='diagonal':
                second=probability@p.embedding.square()
                pr=probability[r]
                second_except=(second[None]-pr[:,None]*p.embedding[r].square())/(1-pr)[:,None]
                second_a=second[None]+dp[:,None]*(p.embedding[r].square()-second_except)
                diagonal=(second_a-(mean[None]+shift).square()).clamp_min(0)
                ga=ga+diagonal*displacement
            zb.grad=g.clone();za.grad=ga.clone()
        ob.step();oa.step()
    if p.final_eos:zb,ob,za,oa=learners()
    result=dict(scores=torch.stack(predictions,1),reference_scores=torch.stack(reference),z=za.detach().clone(),reference_z=zb.detach().clone(),actual_worlds=1,transported_worlds=W,reference_updates=len(p.targets),transported_updates=W*len(p.targets))
    if diagnostics:result.update(reference_optimizer=ob.state_dict(),alternative_optimizer=oa.state_dict())
    return result
