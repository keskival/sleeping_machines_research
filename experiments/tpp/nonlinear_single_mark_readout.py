"""B4 supporting nonlinear correction atop the frozen native-core readout."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from single_mark_readout import SingleMarkReadout,distributions

class NonlinearSingleMarkReadout(SingleMarkReadout):
    def __init__(self,*args):
        super().__init__(*args);d=len(self.feature_mean)
        self.residual=nn.Sequential(nn.Linear(d,64),nn.Tanh(),nn.Linear(64,self.core.K)).to(dtype=self.feature_mean.dtype)
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
        for p in self.core.parameters():p.requires_grad_(False)
        for p in self.readout.parameters():p.requires_grad_(False)
    @torch.no_grad()
    def event_distributions(self,t,m,mask):
        time,lp,features=distributions(self.core,t,m,mask);z=(features-self.feature_mean)/self.feature_std
        return time,F.log_softmax(lp+self.readout(z)+self.residual(z),-1)
