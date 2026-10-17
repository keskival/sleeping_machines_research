import json
import numpy as np
import torch
from experiments.r1_cerl import LocalClip,lanes_nll
from experiments.r1_rmr import lanes_nll_carry

class Minimal(torch.nn.Module):
    cause_msgs=False
    def __init__(self):
        super().__init__();self.base=torch.nn.Linear(1,1);self.levels=torch.nn.ModuleList([torch.nn.Linear(1,1)])
    def lane_nll(self,x,y,rows,stats=False,**kw):
        a,b=rows;out=torch.ones(x.shape[0],b-a)
        return (out,{'bg_only_nll':2.,'clocks':[.25,.75]}) if stats else out
    def carry_out(self,keep):return None

def test_inactive_bank_clipping_calibrates_and_restores():
    m=Minimal();c=LocalClip(m,calib=2)
    for _ in range(3):
        m.zero_grad();m.base.weight.sum().backward();c(m)
        assert m.levels[0].weight.grad is None
    assert all(np.isfinite(list(c.caps.values())))
    restored=LocalClip(m,calib=2);restored.load(c.state());assert restored.state()==c.state()

def test_vector_statistics_aggregate_in_reset_and_carried_scoring():
    m=Minimal();block=np.arange(14).reshape(2,7)
    for result in [lanes_nll(m,block,2,True),lanes_nll_carry(m,block,2,3,True)]:
        assert result['targets']==12 and result['native']['clocks']==[.25,.75]
        assert result['native']['bg_only_nll']==2.
        json.dumps(result,allow_nan=False)
