import copy,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/tpp'))
import numpy as np
import torch
import race_tpp_b4x2 as X


def test_microbatch_preserves_complete_native_optimizer_update_without_dropout_or_target_jitter():
    previous=torch.get_default_dtype();torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(1031)
        a=X.PointerTPP(5,8,4,1,1,2,2,0.,1.,[-1.,1.],.05,0,None,4)
        b=copy.deepcopy(a);oa=torch.optim.AdamW(a.parameters(),lr=.003);ob=torch.optim.AdamW(b.parameters(),lr=.003)
        seqs=[(np.array([0.,.4,1.,2.]),np.array([4,0,1,2])),(np.array([0.,.3,.9,1.5,2.,3.]),np.array([4,1,2,1,0,3]))]
        batch=next(X.base.batches(seqs,2,False,random.Random(0)));X.rep.CAPS=None
        X.micro_update(a,oa,batch,0.,[],0,0,1000)
        X.micro_update(b,ob,batch,0.,[],0,0,6)
        assert all(torch.allclose(v,b.state_dict()[k],atol=1e-11,rtol=1e-11) for k,v in a.state_dict().items())
    finally:torch.set_default_dtype(previous)
