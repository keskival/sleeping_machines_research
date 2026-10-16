import numpy as np
import torch
from experiments.r1_local_native_deep import LocalNativeDeepLM,word_classes


def test_all_native_depth_mixture_has_finite_local_credit():
    torch.manual_seed(918);c,w=word_classes(np.arange(1,18),4)
    model=LocalNativeDeepLM(17,c,w,width=8,depth=8,heads=2,pool=3,concept_units=16,topk=2,explore=1,dk=4)
    tokens=torch.tensor([1,2,1,3,1,2,4]);state=model.new_state()
    g=torch.Generator().manual_seed(11);a=torch.Generator().manual_seed(12)
    nll,objective,state,packets,report=model.nll(tokens[:-1],tokens[1:],state,g,a,stats=True)
    assert torch.isfinite(nll).all() and abs(sum(report['depth_responsibility'])-1)<1e-6
    assert len(model.causes.tm)==len(model.causes.mlp)==0
    objective.backward();model.deliver_credit(packets,state)
    for layer in model.core.units:
        assert sum(float(p.grad.square().sum()) for p in layer.parameters() if p.grad is not None)>0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert report['persistent_storage']['temporal_graph_entries']==0
