"""B4 fixed-core residual mark fit: task-valued DEV gate, unchanged timing model."""
import argparse
import hashlib
import json
import math
import random
import resource
import time
import numpy as np
import torch
import race_tpp_b4g as adapter
from race_tpp_logspread_marks import RaceTPP

ROOT=adapter.ROOT


class FrozenCore(RaceTPP):
    def train(self,mode=True):
        super().train(False) # Frozen core also retains evaluation dropout semantics.
        self.mark_layers.train(mode);self.mark_extra.train(mode)
        return self


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);a=p.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(0)
    random.seed(0);np.random.seed(0);rng=random.Random(0);start=time.monotonic()
    base_path=ROOT/'experiments/results/tpp_b4/aws_b4_logspread_develop_20261010T2112Z.json'
    diag_path=ROOT/'experiments/results/tpp_b4/aws_b4_marks_diagnosis_20261011T0303Z.json'
    base=json.loads(base_path.read_text());diag=json.loads(diag_path.read_text())
    assert base['status']=='completed' and base['skipped_updates']==0 and diag['direct_read_mark_gain']>0
    for r in (base,diag):
        for s,h in r['source_sha256'].items():assert hashlib.sha256((ROOT/s).read_bytes()).hexdigest()==h
    data=adapter.DATA/'github_filtered'/'split_1'
    for part,h in base['data_sha256'].items():assert hashlib.sha256((data/f'{part}.json').read_bytes()).hexdigest()==h
    train,K=adapter.load('github_filtered',1,'train');dev,_=adapter.load('github_filtered',1,'val')
    pilot=json.loads((ROOT/'experiments/results/tpp_b4/aws_b4_logspread_pilot_20261010T2003Z.json').read_text());cfg=base['config']
    pos=np.concatenate([np.diff(t) for t,m in train]);pos=pos[pos>0]
    qs=np.log(np.quantile(pos,np.linspace(.1,.9,cfg['n_lognormal']))).tolist()
    model=FrozenCore(K+1,cfg['d'],cfg['modes'],cfg['layers'],cfg['n_exp'],cfg['n_lognormal'],cfg['dv'],cfg['dropout'],
                     pilot['scale'],qs,pilot['recording_cell'],0,None,cfg['state_modes'])
    checkpoint=ROOT/base['best_checkpoint'];missing=model.load_state_dict(torch.load(checkpoint,weights_only=True),strict=False)
    assert not missing.unexpected_keys and all(k.startswith(('mark_layers.','mark_extra.')) for k in missing.missing_keys)
    for name,param in model.named_parameters():param.requires_grad_(name.startswith(('mark_layers.','mark_extra.')))
    frozen={k:v.detach().clone() for k,v in model.state_dict().items() if not k.startswith(('mark_layers.','mark_extra.'))}
    baseline=adapter.per_sequence(model,dev)
    assert abs(baseline['total']-base['best_dev_total'])<1e-9
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['lr'],weight_decay=0.)
    out=ROOT/'experiments/results/tpp_b4';path=out/f'{a.tag}.json';best_path=out/f'{a.tag}.pt';resume=out/f'{a.tag}.resume.pt'
    assert not path.exists() and not best_path.exists() and not resume.exists()
    history=[];skips=[];best=math.inf;best_epoch=-1;fit_s=eval_s=0.;failure=None;stop='100-pass cap'
    train_targets=sum(len(m)-1 for t,m in train)
    for epoch in range(100):
        tick=time.monotonic();adapter.train_epoch(model,opt,train,cfg,rng,pilot['recording_cell'],skips,epoch);fit_s+=time.monotonic()-tick
        if skips:failure='Nonfinite residual update';break
        for name,value in frozen.items():assert torch.equal(value,model.state_dict()[name]),name
        tick=time.monotonic();v=adapter.per_sequence(model,dev);eval_s+=time.monotonic()-tick
        assert all(math.isfinite(x) for x in v.values())
        assert abs(v['L_T']-baseline['L_T'])<1e-9,'Frozen timing changed'
        if v['total']<best:best=v['total'];best_epoch=epoch;torch.save(model.state_dict(),best_path)
        row=dict(epoch=epoch,val=v,mark_gain=baseline['L_M']-v['L_M'],total_gain=baseline['total']-v['total']);history.append(row);print(json.dumps(row),flush=True)
        temp=resume.with_suffix('.tmp');torch.save(dict(epoch=epoch,model=model.state_dict(),optimizer=opt.state_dict(),
                    torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                    batch_rng=rng.getstate(),history=history),temp);temp.replace(resume)
        if epoch-best_epoch>=30:stop='30-pass total-DEV patience';break
        if time.monotonic()-start>7200:stop='7200s between-pass budget';break
    selected=next((x for x in history if x['epoch']==best_epoch),None)
    window=history[-20:];window_gain=float(np.mean([x['mark_gain'] for x in window])) if window else None
    passed=failure is None and len(window)==20 and selected['mark_gain']>=1 and selected['total_gain']>=1 and window_gain>=1
    sources=['experiments/tpp/b4_frozen_mark_residual.py','experiments/tpp/race_tpp_logspread_marks.py',
             'experiments/tpp/race_tpp_logspread.py','experiments/tpp/race_tpp_v20.py','experiments/tpp/race_tpp_v19.py','experiments/tpp/race_tpp_b4g.py']
    result=dict(status='completed' if failure is None else 'failed',tag=a.tag,battle='B4 development',stage='frozen_core_mark_residual',
                baseline=baseline,selected=selected,history=history,epochs=len(history),skipped_updates=len(skips),failure=failure,stop=stop,
                development_gate=passed,mean_last20_mark_gain=window_gain,
                rule='At least1 mark and total nat/sequence at selected checkpoint and last20-epoch mean,zero skips,bitwise frozen core and unchanged timing.',
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                total_parameters=sum(p.numel() for p in model.parameters()),train_targets_per_pass=train_targets,
                residual_train_targets=len(history)*train_targets,residual_fit_s=fit_s,evaluation_s=eval_s,
                warm_start_fit_s=base['fitting_wall_s'],cumulative_fit_s=fit_s+base['fitting_wall_s'],
                wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                best_checkpoint=str(best_path.relative_to(ROOT)) if best_path.exists() else None,resume_checkpoint=str(resume.relative_to(ROOT)),
                inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (base_path,diag_path)},
                warm_start_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),data_sha256=base['data_sha256'],
                source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
                scope='Fixed Github split1 seed0 TRAIN/DEV residual fit from selected repaired checkpoint; only mark layers/read train. '
                      'Baseline pretraining charged. Joint-fit gate unchanged; no TEST,matched-compute or family verdict.')
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
