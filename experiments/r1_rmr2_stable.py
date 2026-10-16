"""RMR2 with causal standing-memory decay factors bounded by one chunk.

Drops impossible future writes before exponentiation. The predictive law and
parameterization are unchanged; avoids 0*inf and inf*0 on long lanes.
"""
import torch
import torch.nn.functional as F
from r1_rmr2 import RMR2


class StableRMR2(RMR2):
    def standing_coeff(self, a_st, ws_list, rows, T, own_from=None):
        rate=F.softplus(self.st_decay).clamp(max=0.5)
        c0=int(rows[0]);end=int(rows[-1])+1
        # Caller supplies contiguous chunks: every positive exponent <= rate*(chunk-1).
        assert bool((rows==torch.arange(c0,end,device=rows.device)).all())
        assert end-c0<=128, 'Bounded exponential factor requires chunks <=128'
        left=a_st*torch.exp(-rate[None,None,:]*(rows-c0).to(a_st.dtype)[None,:,None])
        out=torch.zeros(*a_st.shape[:2],ws_list[0][0].shape[-1],device=a_st.device,dtype=a_st.dtype) if ws_list else 0.0
        for ds,ws,lim in ws_list:
            stop=min(end,T) if lim=='le' else min(int(lim),end,T)
            if stop<=0:continue
            tt=torch.arange(stop,device=rows.device)
            right=ws[:,:stop]*torch.exp(-rate[None,None,:]*(c0-tt).to(ws.dtype)[None,:,None])
            coef=left@right.transpose(1,2)
            if lim=='le':coef=coef.masked_fill(tt[None,:]>rows[:,None],0.0)
            out=out+coef@ds[:,:stop]
        return out
