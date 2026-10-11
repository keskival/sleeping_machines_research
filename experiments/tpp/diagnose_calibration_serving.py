"""B4 TRAIN-only scalar conditional-mark confidence, saved native predictors."""
import argparse
import hashlib
import json
import random
import resource
import time
import numpy as np
import torch
import race_tpp_b4g as adapter
from race_tpp_logspread import RaceTPP as Stable
from race_tpp_logspread_marks import RaceTPP as MarkModel

ROOT=adapter.ROOT


def mixture(x,y,alpha):
    if alpha==0:return x
    if alpha==1:return y
    return np.logaddexp(np.log1p(-alpha)+x,np.log(alpha)+y)


def fit_confidence(x,y):
    # Convex NLL in the probability-mixture coefficient. Labels fit a single
    # global parameter; the inference weight cannot depend on the target mark.
    def derivative(alpha):
        mixed=mixture(x,y,alpha)
        responsibility=np.exp(np.log(alpha)+y-mixed)
        return float(np.sum((1-responsibility)/(1-alpha)-responsibility/alpha))
    lo,hi=1e-8,1-1e-8
    for _ in range(80):
        mid=(lo+hi)/2
        if derivative(mid)>0:hi=mid
        else:lo=mid
    candidates=[0.,1.,(lo+hi)/2]
    costs=[-float(mixture(x,y,a).sum()) for a in candidates]
    return candidates[int(np.argmin(costs))],dict(candidates=candidates,train_nll=costs,numerical_interval=[1e-8,1-1e-8])


@torch.no_grad()
def score_sequences(model,seqs):
    model.eval();rows=[]
    for seq in seqs:
        t,m,mask=next(adapter.base.batches([seq],1,False,random.Random(0)))
        time_lp,joint_lp,_=model.event_terms(t,m,mask)
        valid=mask[:,1:];lp=(joint_lp-time_lp)[valid].numpy()
        assert np.isfinite(lp).all() and np.max(lp)<1e-9
        rows.append(dict(time=float(time_lp[valid].sum()),mark=lp))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);p.add_argument('--baseline',required=True);p.add_argument('--marks',required=True);a=p.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    tags=[a.baseline,a.marks]
    records=[json.loads((ROOT/'experiments/results/tpp_b4'/f'{tag}.json').read_text()) for tag in tags]
    assert records[0]['seed']==records[1]['seed'] and records[0]['split']==records[1]['split']==1
    for r in records:
        assert r['status']=='completed'
        for s,h in r['source_sha256'].items():assert hashlib.sha256((ROOT/s).read_bytes()).hexdigest()==h
        for part,h in r['data_sha256'].items():assert hashlib.sha256((adapter.DATA/'github_filtered'/'split_1'/f'{part}.json').read_bytes()).hexdigest()==h
    train,K=adapter.load('github_filtered',1,'train');dev,_=adapter.load('github_filtered',1,'val');cfg=records[0]['config']
    pos=np.concatenate([np.diff(t) for t,m in train]);pos=pos[pos>0];qs=np.log(np.quantile(pos,np.linspace(.1,.9,cfg['n_lognormal']))).tolist()
    pilot=json.loads((ROOT/'experiments/results/tpp_b4/aws_b4_logspread_pilot_20261010T2003Z.json').read_text())
    args=(K+1,cfg['d'],cfg['modes'],cfg['layers'],cfg['n_exp'],cfg['n_lognormal'],cfg['dv'],cfg['dropout'],pilot['scale'],qs,pilot['recording_cell'],0,None,cfg['state_modes'])
    models=[Stable(*args),MarkModel(*args)];cached=[];forward_s=0.
    paths=[ROOT/r['best_checkpoint'] for r in records]
    for model,path in zip(models,paths):
        model.load_state_dict(torch.load(path,weights_only=True));tick=time.monotonic()
        cached.append(dict(train=score_sequences(model,train),dev=score_sequences(model,dev)));forward_s+=time.monotonic()-tick
    x,y=[np.concatenate([s['mark'] for s in c['train']]) for c in cached]
    tick=time.monotonic();alpha,optimization=fit_confidence(x,y);fit_s=time.monotonic()-tick
    # alpha is locked before DEV outcomes are combined/reported.
    def aggregate(split,weight):
        rows=[]
        for b,j in zip(cached[0][split],cached[1][split]):
            assert b['mark'].shape==j['mark'].shape
            rows.append(dict(L_T=-b['time'],L_M=-float(mixture(b['mark'],j['mark'],weight).sum())))
        lt=float(np.mean([r['L_T'] for r in rows]));lm=float(np.mean([r['L_M'] for r in rows]))
        return dict(L_T=lt,L_M=lm,total=lt+lm,sequences=len(rows))
    tick=time.monotonic()
    for model in models:score_sequences(model,dev)
    serving_s=time.monotonic()-tick
    base_score=aggregate('dev',0);score=aggregate('dev',alpha);joint_marks=aggregate('dev',1)
    expected=next(h['val'] for h in records[0]['history'] if h['epoch']==records[0]['best_epoch'])
    assert abs(base_score['total']-expected['total'])<1e-9
    gain=base_score['L_M']-score['L_M']
    sources=['experiments/tpp/diagnose_calibration_serving.py','experiments/tpp/race_tpp_logspread.py','experiments/tpp/race_tpp_logspread_marks.py',
             'experiments/tpp/race_tpp_v19.py','experiments/tpp/race_tpp_v20.py','experiments/tpp/race_tpp_b4g.py']
    result=dict(status='completed',tag=a.tag,battle='B4 development',stage='conditional_mark_confidence_diagnostic',
                stage_detail='serving_cost_and_sequence_error',sequence_mark_gains=[float(mixture(b['mark'],j['mark'],alpha).sum()-b['mark'].sum()) for b,j in zip(cached[0]['dev'],cached[1]['dev'])],seed=records[0]['seed'],input_records={tag:hashlib.sha256((ROOT/'experiments/results/tpp_b4'/f'{tag}.json').read_bytes()).hexdigest() for tag in tags},alpha_train_locked=alpha,optimization=optimization,dev=score,baseline_dev=base_score,joint_marks_repaired_time_dev=joint_marks,
                train=aggregate('train',alpha),dev_mark_gain=gain,diagnostic_gate=gain>=1,
                rule='One TRAIN-fitted global probability-mixture coefficient; DEV mark gain>=1nat/sequence with repaired timing fixed. Diagnostic only.',
                cached_scored_targets=dict(train=len(x),dev=sum(len(s['mark']) for s in cached[0]['dev'])),
                predictor_forward_s=forward_s,scalar_fit_s=fit_s,dev_two_predictor_forward_s=serving_s,dev_seconds_per_target=serving_s/sum(len(s['mark']) for s in cached[0]['dev']),serving_protocol='One warm sequential DEV pass through both full predictors, single thread float64; no energy/FLOP estimate',
                prerequisite_fitting_s=sum(r['fitting_wall_s'] for r in records),parameters=sum(p.numel() for m in models for p in m.parameters())+1,
                source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
                checkpoint_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                data_sha256=records[0]['data_sha256'],wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                decision='Use diagnostic to investigate conditional calibration; no adoption, automatic fit or sealed scoring.',
                scope='Saved native two-predictor diagnostic,TRAIN-only scalar fitting,34 DEV sequences. Same alpha for every candidate mark and history; no target-dependent inference selection. '
                      'Both full predictors and prerequisite fits charged. No integrated-substrate,efficiency or TEST win.')
    out=ROOT/'experiments/results/tpp_b4'/f'{a.tag}.json'
    with out.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
