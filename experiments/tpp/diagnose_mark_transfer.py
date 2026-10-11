"""B4 saved-parameter factorial diagnosis, TRAIN/DEV only, no fitting."""
import argparse
import hashlib
import json
import resource
import time
import numpy as np
import torch
import race_tpp_b4g as adapter
from race_tpp_logspread_marks import RaceTPP

ROOT=adapter.ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);a=p.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    tags=['aws_b4_logspread_develop_20261010T2112Z','aws_b4_marks_develop_20261010T2300Z','aws_b4_frozen_marks_20261011T0307Z']
    records=[json.loads((ROOT/'experiments/results/tpp_b4'/f'{tag}.json').read_text()) for tag in tags]
    for r in records:
        assert r['status']=='completed'
        for source,sha in r['source_sha256'].items():assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==sha
        for part,sha in r['data_sha256'].items():assert hashlib.sha256((adapter.DATA/'github_filtered'/'split_1'/f'{part}.json').read_bytes()).hexdigest()==sha
    base,joint,residual=records;assert not residual['development_gate']
    train,K=adapter.load('github_filtered',1,'train');dev,_=adapter.load('github_filtered',1,'val')
    pos=np.concatenate([np.diff(t) for t,m in train]);pos=pos[pos>0];cfg=base['config']
    pilot=json.loads((ROOT/'experiments/results/tpp_b4/aws_b4_logspread_pilot_20261010T2003Z.json').read_text())
    qs=np.log(np.quantile(pos,np.linspace(.1,.9,cfg['n_lognormal']))).tolist()
    model=RaceTPP(K+1,cfg['d'],cfg['modes'],cfg['layers'],cfg['n_exp'],cfg['n_lognormal'],cfg['dv'],cfg['dropout'],
                  pilot['scale'],qs,pilot['recording_cell'],0,None,cfg['state_modes'])
    paths=[ROOT/r['best_checkpoint'] for r in records]+[ROOT/residual['resume_checkpoint']]
    states=[torch.load(p,weights_only=True) for p in paths[:3]]
    # Trusted own resume packet contains optimizer/RNG; only its model is used.
    final=torch.load(paths[3],weights_only=False)['model']
    core_states={'repaired':states[0],'joint':states[1]}
    branch_states={'off':None,'joint':states[1],'residual_best':states[2],'residual_final':final}
    rows=[]
    for core_name,core_state in core_states.items():
        for branch_name,branch_state in branch_states.items():
            state={k:v for k,v in core_state.items() if not k.startswith(('mark_layers.','mark_extra.'))}
            if branch_state is not None:
                state.update({k:v for k,v in branch_state.items() if k.startswith(('mark_layers.','mark_extra.'))})
                model.load_state_dict(state)
            else:
                model.load_state_dict(state,strict=False)
                with torch.no_grad():model.mark_extra.weight.zero_();model.mark_extra.bias.zero_()
            tick=time.monotonic();scores={split:adapter.per_sequence(model,seqs) for split,seqs in (('train',train),('dev',dev))}
            rows.append(dict(core=core_name,branch=branch_name,scores=scores,evaluation_s=time.monotonic()-tick))
            print(json.dumps(rows[-1]),flush=True)
    for core in core_states:
        timing=[r['scores']['dev']['L_T'] for r in rows if r['core']==core]
        assert max(timing)-min(timing)<1e-9
    sources=['experiments/tpp/diagnose_mark_transfer.py','experiments/tpp/race_tpp_logspread_marks.py','experiments/tpp/race_tpp_logspread.py',
             'experiments/tpp/race_tpp_v20.py','experiments/tpp/race_tpp_v19.py','experiments/tpp/race_tpp_b4g.py']
    result=dict(status='completed',tag=a.tag,battle='B4 development',stage='mark_checkpoint_factorial',rows=rows,
                checkpoint_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                input_sha256={tag:hashlib.sha256((ROOT/'experiments/results/tpp_b4'/f'{tag}.json').read_bytes()).hexdigest() for tag in tags},
                data_sha256=base['data_sha256'],source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
                wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                scope='Two saved cores by four saved mark-read settings, repeated TRAIN/DEV evaluation; no parameter updates or TEST. '
                      'Checkpoint swaps diagnose transfer and fit behavior; they do not identify the causal effect of joint training.',
                decision='Use transfer and TRAIN/DEV divergence to select a bounded improvement plan; failed development gates remain failed.')
    out=ROOT/'experiments/results/tpp_b4'/f'{a.tag}.json'
    with out.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
