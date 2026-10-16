import copy,random,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/tpp'))
import b4x_resumable_micro as M


def make(dropout=0):
    torch.manual_seed(1049);return M.X.PointerTPP(5,8,4,1,1,2,2,dropout,1.,[-1.,1.],.05,0,None,4)


def data():return [(np.array([0.,.4,1.,2.]),np.array([4,0,1,2])),(np.array([0.,.3,.9,1.5,2.,3.]),np.array([4,1,2,1,0,3]))]


def test_resumable_micro_matches_full_update_without_stochastic_regularization():
    old=torch.get_default_dtype();torch.set_default_dtype(torch.float64)
    try:
        a=make();b=copy.deepcopy(a);oa=torch.optim.AdamW(a.parameters(),lr=.003,weight_decay=0);ob=torch.optim.AdamW(b.parameters(),lr=.003,weight_decay=0)
        batch=next(M.X.base.batches(data(),2,False,random.Random(0)));M.X.rep.CAPS=None
        M.X.micro_update(a,oa,batch,0.,[],0,0,1000);M.micro_update(b,ob,batch,0.,[],0,0,6)
        assert all(torch.allclose(v,b.state_dict()[k],atol=1e-11,rtol=1e-11) for k,v in a.state_dict().items())
    finally:torch.set_default_dtype(old)


def test_exact_model_optimizer_rng_recovery_with_jitter_and_dropout(tmp_path):
    old=torch.get_default_dtype();torch.set_default_dtype(torch.float64)
    try:
        a=make(.1);oa=torch.optim.AdamW(a.parameters(),lr=.003,weight_decay=0);batch=next(M.X.base.batches(data(),2,False,random.Random(0)));M.X.rep.CAPS=None
        M.micro_update(a,oa,batch,.05,[],0,0,6);path=tmp_path/'state.pt'
        M.save(path,dict(model=a.state_dict(),optimizer=oa.state_dict(),rng=torch.get_rng_state()))
        M.micro_update(a,oa,batch,.05,[],1,0,6);expected=torch.get_rng_state()
        b=make(.1);ob=torch.optim.AdamW(b.parameters(),lr=.003,weight_decay=0);saved=torch.load(path,weights_only=False)
        b.load_state_dict(saved['model']);ob.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng']);M.micro_update(b,ob,batch,.05,[],1,0,6)
        assert torch.equal(torch.get_rng_state(),expected)
        assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    finally:torch.set_default_dtype(old)


def test_single_sequence_validation_preserves_the_official_statistic():
    old=torch.get_default_dtype();torch.set_default_dtype(torch.float64)
    try:
        model=make(.1);a=M.X.per_sequence(model,data(),batch=1);b=M.X.per_sequence(model,data(),batch=32)
        assert all(abs(a[k]-b[k])<1e-11 for k in ['L_T','L_M','total'])
    finally:torch.set_default_dtype(old)
