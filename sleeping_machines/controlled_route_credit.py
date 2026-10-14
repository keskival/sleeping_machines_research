"""Forward-taught baselines can reduce audit noise without changing raw credit."""
import torch


def controlled_route_credit(probabilities,winner,alternative,proposal_probability,loss_difference,baseline):
    if probabilities.ndim!=2 or baseline.shape!=probabilities.shape:
        raise ValueError('One issued baseline per lane and candidate required')
    if any(x.shape!=probabilities.shape[:1] for x in (winner,alternative,proposal_probability,loss_difference)):
        raise ValueError('Matched lane receipts required')
    if bool((winner==alternative).any()):raise ValueError('Audit must select a nonwinner')
    q=proposal_probability.detach();b=baseline.detach()
    if not bool(torch.isfinite(q).all() and (q>0).all() and (q<=1).all() and torch.isfinite(b).all() and torch.isfinite(loss_difference).all()):
        raise ValueError('Finite issued baseline and positive proposal support required')
    pj=probabilities.gather(1,alternative[:,None]).squeeze(1)
    bj=b.gather(1,alternative[:,None]).squeeze(1)
    eligible=1-torch.nn.functional.one_hot(winner,probabilities.shape[1]).to(probabilities.dtype)
    residual=(pj-pj.detach())*(loss_difference.detach()-bj)/q
    correction=((probabilities-probabilities.detach())*b*eligible).sum(-1)
    return (residual+correction).mean()


def check_native_control_mean(probabilities,winners,baseline):
    """Native propensities, synthetic contrasts: finite-sum gradient contract."""
    from .paired_route_credit import paired_route_credit
    maximum=0.
    with torch.enable_grad():
        for lane in range(probabilities.shape[0]):
            logits=probabilities[lane:lane+1].detach().double().log().requires_grad_()
            pi=logits.softmax(-1);winner=winners[lane:lane+1]
            eligible=1-torch.nn.functional.one_hot(winner,pi.shape[1]).double()
            q=.9*pi.detach()*eligible/(pi.detach()*eligible).sum()+.1*eligible/(pi.shape[1]-1)
            labels=torch.arange(pi.shape[1],dtype=torch.float64).sin()[None]
            raw=controlled=0.
            for j in range(pi.shape[1]):
                if j==int(winner[0]):continue
                alt=torch.tensor([j]);proposal=q[:,j];label=labels[:,j]
                raw=raw+proposal[0]*paired_route_credit(pi,alt,proposal,label)
                controlled=controlled+proposal[0]*controlled_route_credit(pi,winner,alt,proposal,label,baseline[lane:lane+1])
            a=torch.autograd.grad(raw,logits,retain_graph=True)[0]
            b=torch.autograd.grad(controlled,logits)[0]
            maximum=max(maximum,float((a-b).abs().max()))
            assert torch.allclose(a,b,atol=1e-12,rtol=1e-12)
    return dict(status='passed',maximum_raw_mean_gradient_error=maximum,
                lanes=probabilities.shape[0],candidates=probabilities.shape[1],
                scope='Finite-sum synthetic contrast contract at actual native route propensities. No learned baseline, actual continuation variance reduction or optimizer utility measured.')
