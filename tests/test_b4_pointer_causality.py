import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/tpp'))
from race_tpp_pointer import RaceTPP


def model():
    torch.manual_seed(1061);m=RaceTPP(5,8,4,1,1,2,2,0.,1.,[-1.,1.],.05,0,None,4).double().eval()
    with torch.no_grad():m.p_null.zero_();m.p_rec.fill_(2);m.p_ind.fill_(2)
    return m


def test_pointer_predictive_mark_law_normalizes_with_active_nonnull_routes():
    m=model();t=torch.tensor([[0.,.4,1.,2.,3.]],dtype=torch.float64);mask=torch.ones_like(t,dtype=torch.bool);history=torch.tensor([[4,0,1,2,3]])
    probabilities=[];times=[]
    with torch.no_grad():
        for target in range(5):
            marks=history.clone();marks[0,3]=target
            time,joint,_=m.event_terms(t,marks,mask);probabilities.append((joint[0,2]-time[0,2]).exp());times.append(time[0,2])
    assert torch.allclose(torch.stack(probabilities).sum(),torch.tensor(1.,dtype=torch.float64),atol=1e-12)
    assert all(torch.allclose(x,times[0],atol=1e-12) for x in times)


def test_pointer_future_marks_and_times_do_not_change_prefix_likelihoods():
    m=model();t=torch.tensor([[0.,.4,1.,2.,3.]],dtype=torch.float64);marks=torch.tensor([[4,0,1,2,3]]);mask=torch.ones_like(t,dtype=torch.bool)
    with torch.no_grad():
        first=m.event_terms(t,marks,mask)[:2]
        future_t=t.clone();future_t[0,4]=100.;future_m=marks.clone();future_m[0,4]=0
        changed=m.event_terms(future_t,future_m,mask)[:2]
    assert all(torch.allclose(a[:,:3],b[:,:3],atol=1e-11,rtol=1e-11) for a,b in zip(first,changed))
