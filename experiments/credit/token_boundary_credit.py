"""Connected downstream sensitivity to transported complete learner differences."""
import torch
from token_learner_state_features import learner_features

class BoundaryCredit(torch.nn.Module):
    def __init__(self,context_size,dimension,width=64):
        super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(context_size,width),torch.nn.Tanh(),torch.nn.Linear(width,width),torch.nn.Tanh(),torch.nn.Linear(width,3*dimension))
        torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.zeros_(self.net[-1].bias)
    def forward(self,context,differences):
        return .001*(differences*self.net(context)[None]).sum(1)

def boundary_features(transport,record,embedding):
    ref=transport['reference_learner'];zeros=torch.zeros_like(ref['z']);zero_worlds=torch.zeros_like(transport['z'])
    rfields=next(iter(ref['optimizer']['state'].values()),dict(exp_avg=zeros,exp_avg_sq=zeros));fields=next(iter(transport['alternative_optimizer']['state'].values()),dict(exp_avg=zero_worlds,exp_avg_sq=zero_worlds))
    differences=torch.cat([(transport['z']-ref['z'])/.01,(fields['exp_avg']-rfields['exp_avg'])/.001,(fields['exp_avg_sq'].sqrt()-rfields['exp_avg_sq'].sqrt())/.001],1)
    context=torch.cat([record['context'],record['boundary_geometry'],learner_features(ref,record['context'],embedding.shape[1]),embedding[record['tokens'][64:72]].mean(0)])
    return context,differences
