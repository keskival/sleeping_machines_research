"""AWS B5 review: strict per-query causal memory; inherited native race readout.

Repair the batch-stale visibility identified on wiki. Retain sparse addressed
clock banks and sampled losing-candidate credit. Ties are excluded across batch
and split boundaries. No benchmark TEST admitted by the development pipeline.
"""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import race_link_review as base

ROOT=Path(__file__).resolve().parents[2]


class CausalState(base.State):
    def __init__(self,n):
        self.n=n
        self.pop=np.zeros((n,len(base.TAUS)));self.dcnt=np.zeros(n);self.dlast=np.full(n,np.nan)
        self.scnt=np.zeros(n);self.slast=np.full(n,np.nan)
        self.pidx={};cap=1<<15
        self.pa=np.zeros((cap,len(base.TAUS)));self.pc=np.zeros(cap);self.pl=np.zeros(cap)
        self.seen_dst=np.zeros(n,bool);self.seen_list=[]
        self.pending=[];self.latest=-np.inf

    def advance(self,t):
        assert t>=self.latest,'Nonchronological query'
        if self.pending and self.pending[0][2]<t:
            s,c,ts=zip(*self.pending)
            assert max(ts)<t
            self.add(np.asarray(s),np.asarray(c),np.asarray(ts))
            self.pending=[]

    def push(self,s,c,t):
        assert t>=self.latest,'Nonchronological input'
        self.pending.append((int(s),int(c),float(t)));self.latest=float(t)

    def history(self,s,c,t):
        for u,v,stamp in zip(s,c,t):self.advance(float(stamp));self.push(u,v,stamp)


def causal_batch(st,s,c,t,candidates=None,neg=20,rng=None):
    features=[];candidate_rows=[]
    for q in range(len(s)):
        st.advance(float(t[q]))
        if candidates is None:
            competitors=st.sample_negatives(neg,rng,1)[0]
            # Exclude the positive and duplicate competitors; a duplicated
            # destination is not an independent competing clock.
            used={int(c[q])}
            for j in range(len(competitors)):
                while int(competitors[j]) in used:competitors[j]=rng.integers(st.n)
                used.add(int(competitors[j]))
            cand=np.concatenate(([c[q]],competitors)).astype(np.int64)
        else:cand=candidates[q]
        features.append(st.features(np.asarray([s[q]]),cand[None],np.asarray([t[q]]))[0])
        candidate_rows.append(cand)
        st.push(s[q],c[q],float(t[q]))
    return np.stack(candidate_rows),np.stack(features)


def evaluate(ds,ev,model,st,mask,split,max_q=0):
    d=ds.full_data;src=d['sources'][mask];dst=d['destinations'][mask];rt=d['timestamps'][mask]
    if max_q:src,dst,rt=src[:max_q],dst[:max_q],rt[:max_q]
    model.eval();out=[];checked=0;start=time.time();profile={k:[] for k in ('repeat_pair','new_pair_known_source','new_source')}
    with torch.no_grad():
        for b in range(0,len(src),200):
            s,c,r=src[b:b+200],dst[b:b+200],rt[b:b+200]
            negatives=ds.negative_sampler.query_batch(s,c,r,split_mode=split)
            candidates=np.stack([np.concatenate(([c[q]],np.asarray(negatives[q]))) for q in range(len(s))]).astype(np.int64)
            candidates,features=causal_batch(st,s,c,r.astype(float),candidates)
            sc=model(torch.from_numpy(s.astype(np.int64)),torch.from_numpy(candidates),torch.from_numpy(features)).numpy()
            for q in range(len(s)):
                value=base.rr(sc[q]);out.append(value)
                category='repeat_pair' if features[q,0,5]>0 else ('new_pair_known_source' if features[q,0,14]>0 else 'new_source')
                profile[category].append(value)
                if checked<50:
                    official=ev.eval(dict(y_pred_pos=sc[q,:1],y_pred_neg=sc[q,None,1:],eval_metric=['mrr']))['mrr']
                    assert abs(float(official)-value)<1e-6;checked+=1
    return float(np.mean(out)),len(out),time.time()-start,{k:dict(queries=len(v),mrr=float(np.mean(v)) if v else None) for k,v in profile.items()}


def contracts():
    s=np.array([0,0,0,1,0,0]);c=np.array([1,2,1,2,1,3]);t=np.array([1.,1.,2.,2.,3.,4.])
    candidates=np.array([[1,2,3]]*len(s))
    reference=[]
    for q in range(len(s)):
        prefix=base.State(5);mask=t<t[q];prefix.add(s[mask],c[mask],t[mask])
        reference.append(prefix.features(s[q:q+1],candidates[q:q+1],t[q:q+1])[0])
    expected=np.stack(reference);maximum=0.
    for batch in (1,2,3,6):
        st=CausalState(5);pieces=[]
        for b in range(0,len(s),batch):pieces.append(causal_batch(st,s[b:b+batch],c[b:b+batch],t[b:b+batch],candidates[b:b+batch])[1])
        maximum=max(maximum,float(np.max(np.abs(np.concatenate(pieces)-expected))))
    assert maximum<1e-7
    # Prefix-only reference excludes ties even when the split boundary bisects them.
    st=CausalState(5);st.history(s[:1],c[:1],t[:1])
    _,boundary=causal_batch(st,s[1:],c[1:],t[1:],candidates[1:])
    boundary_error=float(np.max(np.abs(boundary-expected[1:])));assert boundary_error<1e-7
    # A batch-stale implementation misses an earlier event in this same batch.
    stale=base.State(5).features(s,candidates,t)
    stale_difference=float(np.max(np.abs(stale-expected)));assert stale_difference>1
    return dict(maximum_causal_prefix_error=maximum,split_tie_boundary_error=boundary_error,
                stale_batch_difference=stale_difference,batch_sizes=[1,2,3,6],strict_timestamp_cut=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--contract',action='store_true')
    ap.add_argument('--dataset',default='tgbl-review');ap.add_argument('--root',default='../../../data/tgb_aws/root')
    for name,default in [('seed',0),('epochs',3),('neg',20),('dim',16),('hidden',64),('max-train',0),('max-val',0)]:ap.add_argument('--'+name,type=int,default=default)
    ap.add_argument('--lr',type=float,default=.003);ap.add_argument('--weight-decay',type=float,default=.0001)
    a=ap.parse_args();start=time.time();torch.set_num_threads(1)
    out=ROOT/'experiments/results/tgb'/f'{a.tag}.json';assert not out.exists(),'Use a fresh tag'
    if a.contract:result=dict(status='completed',checks=contracts(),scope='Numerical causal-state contracts; no fitting')
    else:
        torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed);base.safe_tgb.patch_tgb()
        from tgb.linkproppred.dataset import LinkPropPredDataset
        from tgb.linkproppred.evaluate import Evaluator
        ds=LinkPropPredDataset(name=a.dataset,root=a.root,preprocess=True,download=False);ev=Evaluator(name=a.dataset)
        ds.load_val_ns();d=ds.full_data;n=int(max(d['sources'].max(),d['destinations'].max()))+1
        tr=np.flatnonzero(ds.train_mask);limited=tr[:a.max_train] if a.max_train else tr
        model=base.Model(n,a.dim,a.hidden);opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
        ckpt=out.with_suffix('.pt');best=-1.;best_ep=-1;history=[]
        for epoch in range(a.epochs):
            st=CausalState(n);model.train();epoch_start=time.time();total=0.;batches=0
            for b in range(0,len(limited),200):
                ix=limited[b:b+200];s=d['sources'][ix];c=d['destinations'][ix];t=d['timestamps'][ix].astype(float)
                cand,feat=causal_batch(st,s,c,t,neg=a.neg,rng=rng)
                sc=model(torch.from_numpy(s.astype(np.int64)),torch.from_numpy(cand),torch.from_numpy(feat))
                loss=F.cross_entropy(sc,torch.zeros(len(s),dtype=torch.long));assert torch.isfinite(loss)
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                total+=float(loss.detach());batches+=1
            train_wall=time.time()-epoch_start
            if a.max_train:
                rest=tr[a.max_train:];st.history(d['sources'][rest],d['destinations'][rest],d['timestamps'][rest].astype(float))
            replay_wall=time.time()-epoch_start-train_wall
            mrr,nq,val_wall,profile=evaluate(ds,ev,model,st,ds.val_mask,'val',a.max_val)
            row=dict(epoch=epoch,train_loss=total/max(batches,1),val_mrr=mrr,val_queries=nq,
                     train_wall_s=train_wall,remaining_history_replay_wall_s=replay_wall,val_wall_s=val_wall,
                     epoch_s=time.time()-epoch_start,query_profile=profile)
            history.append(row);print(json.dumps(row),flush=True)
            if mrr>best:best=mrr;best_ep=epoch;torch.save(model.state_dict(),ckpt)
        result=dict(status='completed',args=vars(a),dataset=a.dataset,nodes=n,parameters=sum(p.numel() for p in model.parameters()),
                    val_mrr=best,best_epoch=best_ep,history=history,checkpoint=str(ckpt.relative_to(ROOT)),
                    trained_events=len(limited)*a.epochs,competitors_per_train_query=a.neg,
                    protocol='Per-query features from events strictly earlier than query timestamp; pending ties carried across batches and splits; official validation negatives/Evaluator; no TEST',
                    evidence_level='first-pass v2 development repair; bounded prefixes are diagnostics, not full validation claims')
    sources=[Path(__file__).resolve(),Path(base.__file__).resolve(),ROOT/'experiments/tgb/safe_tgb.py']
    result.update(tag=a.tag,battle='B5 tgbl-review development',wall_s=time.time()-start,
                  max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('history','args')},indent=2),flush=True)


if __name__=='__main__':main()
