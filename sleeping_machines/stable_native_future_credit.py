"""Native persistent critic with a fixed feature coordinate system per epoch.

Feature parameters stay fixed while recursive values are fitted. Explicit
recalibration resets the response before a new feature epoch. Forgetting is
separate from coordinate stability and handles changing actor action utility.
"""
import math
import torch
from .calibrated_native_future_credit import CalibratedNativeFutureCredit

class StableNativeFutureCredit(CalibratedNativeFutureCredit):
    def __init__(self,*args,forgetting=1.,**kwargs):
        super().__init__(*args,**kwargs)
        if not 0<forgetting<=1:raise ValueError('forgetting must lie in (0,1]')
        self.register_buffer('forgetting',torch.tensor(float(forgetting),dtype=torch.float64))
        self.register_buffer('feature_epoch',torch.tensor(0,dtype=torch.long))
        self._feature_versions=None

    def begin_decision(self):
        super().begin_decision()
        versions=tuple(p._version for p in self.parameters())
        if getattr(self,'_feature_versions',None) is not None and versions!=self._feature_versions:
            raise RuntimeError('feature encoder changed: explicitly start a new calibration epoch')
        self._feature_versions=versions

    def load_state_dict(self,*args,**kwargs):
        result=super().load_state_dict(*args,**kwargs);self._feature_versions=None
        self.begin_decision();return result

    @torch.no_grad()
    def recalibrate(self):
        self.response.zero_();self.covariance.copy_(10*torch.eye(len(self.response),device=self.response.device,dtype=self.response.dtype))
        self.residual_variance.fill_(1e-4);self.observations.zero_();self.feature_epoch.add_(1)
        self._feature_versions=None;self.begin_decision()

    def quote(self,*args,actual_action_norm=None,**kwargs):
        if actual_action_norm is not None and (not math.isfinite(actual_action_norm) or actual_action_norm<0):raise ValueError('finite nonnegative actual action norm required')
        quote,state,packets=super().quote(*args,**kwargs)
        if actual_action_norm is not None:
            step=args[4] if len(args)>4 else kwargs['step_rms']
            factor=actual_action_norm/(step*self.width)
            self.features[-1]=self.features[-1]*factor
            quote=quote*factor
        return quote,state,packets

    def choose_service(self,quotes,*,uncertainty_price=.5):
        if not math.isfinite(uncertainty_price) or uncertainty_price<0:raise ValueError('nonnegative finite risk price required')
        if not bool(torch.isfinite(quotes).all()):raise ValueError('finite quoted action costs required')
        choice,uncertainty=super().choose_service(quotes,uncertainty_price=uncertainty_price)
        # Zero-cost interventions tie with an exact no-op; prefer the no-op.
        if float(quotes[choice].detach())+uncertainty_price*uncertainty>=0:return 1,0.
        return choice,uncertainty

    def learn_audit(self,quotes,index,target,packets,state,optimizer=None):
        if tuple(p._version for p in self.parameters())!=self._feature_versions:raise RuntimeError('feature encoder changed during a decision')
        target=torch.as_tensor(target,dtype=torch.float64,device=self.response.device).detach()
        if target.ndim or not bool(torch.isfinite(target)):raise ValueError('finite revealed action value required')
        features=torch.cat(self.features).detach().double()
        if not isinstance(index,int) or not 0<=index<len(features):raise ValueError('valid quoted action required')
        feature=features[index];prediction=quotes[index].detach().double()
        if not bool(torch.isfinite(feature).all()) or not bool(torch.isfinite(prediction)):raise ValueError('finite features and prediction required')
        with torch.no_grad():
            projected=self.covariance@feature;denominator=self.forgetting+feature@projected
            if not bool(torch.isfinite(denominator)) or float(denominator)<=0:raise RuntimeError('invalid recursive information matrix')
            residual=target-feature@self.response;error=(prediction-target).square()
            self.response.add_(projected*residual/denominator)
            self.covariance.sub_(torch.outer(projected,projected)/denominator).div_(self.forgetting)
            self.covariance.copy_((self.covariance+self.covariance.T)/2)
            self.observations.add_(1);self.residual_variance.mul_(.95).add_(.05*error).clamp_(min=1e-8)
        # No encoder backward/optimizer step. Native numerical state evolves;
        # old packets do not carry updates across feature epochs.
        return dict(predicted_audit_advantage=float(prediction),actual_audit_advantage=float(target),audit_squared_error=float(error),credit_preclip_norm=0.,response_observations=int(self.observations),feature_epoch=int(self.feature_epoch),residual_variance=float(self.residual_variance))
