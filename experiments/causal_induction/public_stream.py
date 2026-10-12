"""Public-only causal event/query/feedback interface; no oracle-file access."""
import numpy as np

def records(path):
    with np.load(path,allow_pickle=False) as data:
        assert set(data.files)=={'events','queries','feedback','capacity'},data.files
        entries=[]
        for t,n,v,ext in data['events']:entries.append((float(t),0,dict(kind='event',time=float(t),node=int(n),value=int(v),exogenous=bool(ext))))
        for t,n,h,q in data['queries']:entries.append((float(t),1,dict(kind='query',time=float(t),node=int(n),horizon=float(h),query_id=int(q))))
        for t,q,y in data['feedback']:entries.append((float(t),2,dict(kind='feedback',time=float(t),query_id=int(q),target=int(y))))
    for _,_,row in sorted(entries,key=lambda x:(x[0],x[1])):yield row

def persistence(path,prior=(1.,1.),adapt=True):
    correct,total=prior;state={};issued={};scores=[];trace=[];updates=0;events=0;max_slots=0
    for r in records(path):
        if r['kind']=='event':state[r['node']]=r['value'];events+=1;max_slots=max(max_slots,len(state))
        elif r['kind']=='query':
            assert r['query_id'] not in issued and 'target' not in r
            value=state.get(r['node'],0);p=correct/total if total>correct else .5;p=max(.001,min(.999,p));issued[r['query_id']]=(r['time'],value,p)
            trace.append((r['time'],r['query_id'],value,p))
        else:
            when,value,p=issued.pop(r['query_id']);assert when<r['time'];match=int(value==r['target']);scores.append(-np.log(p if match else 1-p))
            if adapt:correct+=match;total+=1;updates+=1
    assert not issued
    return dict(nll=float(np.mean(scores)),targets=len(scores),online_calibration_updates=updates,events=events,max_observed_slots=max_slots,issued_predictions=trace), (correct,total)
