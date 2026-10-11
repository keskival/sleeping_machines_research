"""B4 DEV-only frozen-checkpoint mark-read ablation after a failed window gate."""
import argparse
import hashlib
import json
import random
import resource
import time
import numpy as np
import torch
import race_tpp_b4g as adapter
from race_tpp_logspread_marks import RaceTPP

ROOT=adapter.ROOT


class Ablated(RaceTPP):
    disable_read=False
    def encode(self,*args):
        encoded=super().encode(*args)
        if self.disable_read:self._extra=None
        return encoded


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);a=p.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    fit_path=ROOT/'experiments/results/tpp_b4/aws_b4_marks_develop_20261010T2300Z.json';r=json.loads(fit_path.read_text())
    decision_path=ROOT/'experiments/results/tpp_b4/aws_b4_marks_decision_20261010T2301Z.json';d=json.loads(decision_path.read_text())
    assert r['status']=='completed' and d['status']=='completed' and not d['development_gate']
    for s,h in r['source_sha256'].items():assert hashlib.sha256((ROOT/s).read_bytes()).hexdigest()==h
    train,K=adapter.load('github_filtered',1,'train');dev,_=adapter.load('github_filtered',1,'val')
    pilot=json.loads((ROOT/'experiments/results/tpp_b4/aws_b4_logspread_pilot_20261010T2003Z.json').read_text())
    cfg=r['config'];pos=np.concatenate([np.diff(t) for t,m in train]);pos=pos[pos>0]
    qs=np.log(np.quantile(pos,np.linspace(.1,.9,cfg['n_lognormal']))).tolist()
    model=Ablated(K+1,cfg['d'],cfg['modes'],cfg['layers'],cfg['n_exp'],cfg['n_lognormal'],cfg['dv'],cfg['dropout'],
                  pilot['scale'],qs,pilot['recording_cell'],0,None,cfg['state_modes'])
    ckpt=ROOT/r['best_checkpoint'];model.load_state_dict(torch.load(ckpt,weights_only=True));model.eval()
    scores=[];timing=[]
    for disable in (False,True):
        model.disable_read=disable;tick=time.monotonic();rows=[]
        with torch.no_grad():
            for seq in dev:
                batch=next(adapter.base.batches([seq],1,False,random.Random(0)))
                tl,ml,count,_=model.loglik(*batch)
                rows.append(dict(L_T=-float(tl),L_M=-float(ml),total=-float(tl+ml),events=int(count)))
        scores.append(rows);timing.append(time.monotonic()-tick)
    gains=np.array([off['L_M']-on['L_M'] for on,off in zip(*scores)])
    time_error=max(abs(on['L_T']-off['L_T']) for on,off in zip(*scores));assert time_error<1e-10
    mean=float(gains.mean());baseline=d['selected_repaired_baseline']['L_M']
    off_mean=float(np.mean([s['L_M'] for s in scores[1]]))
    selected_mean=float(np.mean([s['L_M'] for s in scores[0]]))
    assert abs(selected_mean-d['selected_mark_memory']['L_M'])<1e-9
    sources=['experiments/tpp/diagnose_marks_development.py','experiments/tpp/race_tpp_logspread_marks.py',
             'experiments/tpp/race_tpp_logspread.py','experiments/tpp/race_tpp_v20.py','experiments/tpp/race_tpp_v19.py','experiments/tpp/race_tpp_b4g.py']
    result=dict(status='completed',tag=a.tag,battle='B4 development',stage='mark_read_ablation',
                on_mean_mark_loss=selected_mean,off_mean_mark_loss=off_mean,direct_read_mark_gain=mean,
                other_fitted_parameter_mark_difference=baseline-off_mean,max_timing_difference=time_error,
                sequence_gains=gains.tolist(),sequences_helped=int((gains>0).sum()),sequences_harmed=int((gains<0).sum()),
                per_sequence_on=scores[0],per_sequence_off=scores[1],evaluation_wall_s=timing,
                wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (fit_path,decision_path)},
                source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
                decision='Diagnose direct branch benefit and cross-fit differences before another fit; failed window gate remains failed.',
                scope='Frozen selected checkpoint,34 DEV sequences,mark read enabled/disabled with all computation charged; no updates or TEST. '
                      'Cross-fit difference includes indirect learning and RNG changes; it is not an identified causal effect.')
    out=ROOT/'experiments/results/tpp_b4'/f'{a.tag}.json'
    with out.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
