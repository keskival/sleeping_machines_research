"""Neutral, action-scaled reciprocal value learning with separate audit policy.

Native temporal/queryable feature learning is retained. A small regularized
least-squares response learns in actual loss units instead of assigning random
large advantages to unobserved actions. Uncertainty is an admission heuristic,
not a frequentist coverage certificate under changing representations.
"""
import math
import torch
from .native_future_update_credit import NativeFutureUpdateCredit


class CalibratedNativeFutureCredit(NativeFutureUpdateCredit):
    def __init__(self, width, depth=8, heads=4, pool=4):
        super().__init__(width, depth, heads, pool)
        del self.action, self.value
        size=2*(width+1)
        self.register_buffer('response',torch.zeros(size,dtype=torch.float64))
        self.register_buffer('covariance',10*torch.eye(size,dtype=torch.float64))
        self.register_buffer('residual_variance',torch.tensor(1e-4,dtype=torch.float64))
        self.register_buffer('observations',torch.tensor(0,dtype=torch.long))
        self.begin_decision()

    def begin_decision(self):
        self.features=[]

    def quote(self,published,producer_input,gradient_summary,address,step_rms,
              state,generator,alternative_generator,*,timestamp=None):
        if any(x.shape!=(self.width,) for x in [published,producer_input,gradient_summary]):
            raise ValueError('actual state-interface vectors required')
        if not math.isfinite(step_rms) or step_rms<=0:raise ValueError('positive action scale required')
        stamp=float(state['events']) if timestamp is None else float(timestamp)
        clocks=state.setdefault('interface_observation_times',{})
        elapsed=stamp-clocks.get(address,stamp)
        if not math.isfinite(stamp) or elapsed<0:raise ValueError('causal observation time required')
        clocks[address]=stamp
        context=torch.cat([published.detach(),producer_input.detach(),gradient_summary.detach(),
                           published.new_tensor([address,math.log1p(step_rms),math.log1p(elapsed)])])
        views,state,packets,_=self.core.consume(self.context(context),state,generator,alternative_generator,learn=True)
        z=torch.cat([views.new_ones(1),views.mean(0)/math.sqrt(self.width)])
        radius=step_rms*self.width  # square width-by-width step with RMS step_rms
        signs=published.new_tensor([-1.,0.,1.])[:,None]
        features=torch.cat([signs*radius*z,signs.abs()*radius*z],-1)
        self.features.append(features)
        quote=(features.double()@self.response.detach().clone()).to(views.dtype)
        return quote,state,packets

    def choose_service(self,quotes,*,uncertainty_price=.5):
        features=torch.cat(self.features).detach().double()
        if quotes.shape!=(len(features),):raise ValueError('one value per actual action required')
        variance=torch.einsum('nd,df,nf->n',features,self.covariance,features).clamp_min(0)
        uncertainty=(variance*self.residual_variance).sqrt()
        costs=quotes.detach().double()+uncertainty_price*uncertainty
        selected=int(costs.argmin())
        # No-op has exactly zero cost and uncertainty. Positive uncertain costs
        # cannot be mistaken for a reason to intervene.
        return selected,float(uncertainty[selected])

    def choose_audit(self,generator):
        features=torch.cat(self.features)
        eligible=torch.arange(len(features))[torch.arange(len(features))%3!=1]
        index=int(eligible[torch.randint(len(eligible),(1,),generator=generator)])
        return index,1/len(eligible)

    def learn_audit(self,quotes,index,target,packets,state,optimizer):
        target=torch.as_tensor(target,dtype=torch.float64).detach()
        if target.ndim or not bool(torch.isfinite(target)):raise ValueError('finite revealed action advantage required')
        prediction=quotes[index]
        feature=torch.cat(self.features)[index].detach().double()
        raw_error=(prediction.detach().double()-target).square()
        optimizer.zero_grad(set_to_none=True)
        # First observation trains the neutral response; only a nonzero known
        # response can teach its native feature producer. No zero-signal decay.
        norm=0.
        if bool(self.response.abs().any()):
            loss=(prediction.double()-target).square()/self.residual_variance.detach().clamp_min(1e-8)
            loss.backward();self.core.deliver_credit(packets,state)
            norm=float(torch.nn.utils.clip_grad_norm_(self.parameters(),1.,error_if_nonfinite=True))
            optimizer.step()
        with torch.no_grad():
            projected=self.covariance@feature
            denominator=1+feature@projected
            residual=target-feature@self.response
            self.response.add_(projected*(residual/denominator))
            self.covariance.sub_(torch.outer(projected,projected)/denominator)
            self.covariance.copy_((self.covariance+self.covariance.T)/2)
            self.observations.add_(1)
            self.residual_variance.mul_(.95).add_(.05*raw_error).clamp_(min=1e-8)
        return dict(predicted_audit_advantage=float(prediction.detach()),actual_audit_advantage=float(target),
                    audit_squared_error=float(raw_error),credit_preclip_norm=norm,
                    response_observations=int(self.observations),residual_variance=float(self.residual_variance))
