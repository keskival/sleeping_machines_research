"""Information-matched B10 encoders with the original normalized RaceTPP head."""
import math
import torch
from torch import nn
from b10_depth_tpp_v1 import Race

class ExplicitLSTM(nn.Module):
    # Explicit gates expose backward arithmetic to the operator ledger.
    def __init__(self):
        super().__init__()
        self.input = nn.ModuleList(nn.Linear(32,128) for _ in range(2))
        self.hidden = nn.ModuleList(nn.Linear(32,128,bias=False) for _ in range(2))
    def forward(self,x):
        for a,b in zip(self.input,self.hidden):
            h=x.new_zeros(x.shape[0],32); c=torch.zeros_like(h); out=[]
            projected=a(x)
            for i in range(x.shape[1]):
                ig,fg,cg,og=(projected[:,i]+b(h)).chunk(4,-1)
                c=torch.sigmoid(fg)*c+torch.sigmoid(ig)*torch.tanh(cg)
                h=torch.sigmoid(og)*torch.tanh(c); out.append(h)
            x=torch.stack(out,1)
        return x

class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__(); self.qkv=nn.Linear(32,96);self.output=nn.Linear(32,32)
        self.norm1=nn.LayerNorm(32); self.norm2=nn.LayerNorm(32)
        self.ff=nn.Sequential(nn.Linear(32,128),nn.GELU(),nn.Linear(128,32))
    def forward(self,x):
        B,L,_=x.shape
        q,k,v=[z.reshape(B,L,4,8).transpose(1,2) for z in self.qkv(x).chunk(3,-1)]
        mask=torch.ones(L,L,dtype=torch.bool,device=x.device).triu(1)
        a=(q@k.transpose(-1,-2)/math.sqrt(8)).masked_fill(mask,-math.inf).softmax(-1)
        h=self.norm1(x+self.output((a@v).transpose(1,2).reshape(B,L,32)))
        return self.norm2(h+self.ff(h))

class MatchedRace(Race):
    def __init__(self,gaps,family):
        super().__init__(gaps,32,8,2); self.family=family
        if family=='native':return
        # Remove unused native encoder parameters, retain identical addressed
        # mark memory and all clock/mark output parameters.
        self.m.layers=nn.ModuleList()
        self.m.gap=nn.Linear(1,32)
        self.encoder=ExplicitLSTM() if family=='lstm' else nn.Sequential(AttentionBlock(),AttentionBlock())
        if family not in ('lstm','transformer'):raise ValueError(family)
        self.m.encode=self.encode
    def encode(self,t,m,mask):
        dt=torch.diff(t,dim=1,prepend=t[:,:1]).clamp_min(0)/self.m.scale
        x=self.m.embed(m)+self.m.gap(torch.log1p(dt).unsqueeze(-1))
        if self.family=='transformer':
            # Two disjoint sinusoidal subspaces: elapsed physical time / TRAIN
            # median and within-window index. Exact encoding fixed before fits.
            frequency=torch.exp(torch.arange(8,dtype=t.dtype,device=t.device)*(-math.log(10000)/8))
            elapsed=(t-t[:,:1])/self.m.scale
            position=torch.arange(t.shape[1],dtype=t.dtype,device=t.device).expand_as(t)
            def pe(z):
                a=z.unsqueeze(-1)*frequency
                return torch.stack((a.sin(),a.cos()),-1).flatten(-2)
            x=x+torch.cat((pe(elapsed),pe(position)),-1)
        h=self.encoder(x)
        return h,self.m.marks(h,m,dt)
