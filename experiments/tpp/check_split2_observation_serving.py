"""B4 standalone one-core readout bundles, serialization and causal serving."""
import argparse,hashlib,json,random,resource,time
from pathlib import Path
import numpy as np
import torch
import race_tpp_b4g as adapter
from race_tpp_logspread import RaceTPP as Stable
from nonlinear_single_mark_readout import NonlinearSingleMarkReadout
ROOT=adapter.ROOT

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def restore(bundle):
    core=Stable(*bundle['core_args']).eval();s=bundle['weights'];mean,std=s['feature_mean'],s['feature_std'];coef=torch.cat([s['readout.weight'].T,s['readout.bias'][None]],0)
    model=NonlinearSingleMarkReadout(core,mean,std,coef).eval();model.load_state_dict(s);return model
@torch.no_grad()
def evaluate(model,seqs):
    values=[];normalization=0.;tick=time.monotonic()
    for seq in seqs:
        t,m,mask=next(adapter.base.batches([seq],1,False,random.Random(0)));valid=mask[:,1:]
        time_lp,lp=model.event_distributions(t,m,mask);normalization=max(normalization,float((lp[valid].exp().sum(-1)-1).abs().max()))
        selected=lp.gather(-1,m[:,1:,None]).squeeze(-1);values.append([-float(time_lp[valid].sum()),-float(selected[valid].sum())])
    lt,lm=np.mean(values,0);return dict(L_T=float(lt),L_M=float(lm),total=float(lt+lm),normalization_error=normalization,wall_s=time.monotonic()-tick)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    decision_path=ROOT/'experiments/results/tpp_b4/aws_b4_split2_observation_readout_20261011.json';d=json.loads(decision_path.read_text());assert d['status']=='completed' and d['gate']
    for source,h in d['source_sha256'].items():assert digest(ROOT/source)==h
    for path,h in d['input_sha256'].items():assert digest(ROOT/path)==h
    pilot=json.loads((ROOT/'experiments/results/tpp_b4/aws_b4_split2_logspread_pilot_20261011_v2.json').read_text());dev,K=adapter.load('github_filtered',2,'val');rows=[];inputs={str(decision_path.relative_to(ROOT)):digest(decision_path)}
    for seed in range(1):
        r=json.loads((ROOT/'experiments/results/tpp_b4'/'aws_b4_split2_observation_readout_20261011.json').read_text())
        for source,h in r['source_sha256'].items():assert digest(ROOT/source)==h
        original=ROOT/r['checkpoint'];assert digest(original)==r['checkpoint_sha256'];inputs[str(original.relative_to(ROOT))]=digest(original)
        # Configuration is metadata only: no teacher/core checkpoint is opened.
        cfg=pilot['config'];core_args=[K+1,cfg['d'],cfg['modes'],cfg['layers'],cfg['n_exp'],cfg['n_lognormal'],cfg['dv'],cfg['dropout'],pilot['scale'],[0.]*cfg['n_lognormal'],pilot['recording_cell'],0,None,cfg['state_modes']]
        weights=torch.load(original,weights_only=True);bundle=dict(schema_version=1,kind='NonlinearSingleMarkReadout',core_args=core_args,weights=weights)
        path=ROOT/'experiments/results/tpp_b4'/f'{a.tag}.s{seed}.bundle.pt';assert not path.exists();torch.save(bundle,path)
        model=restore(torch.load(path,weights_only=True));scored=evaluate(model,dev)
        error=max(abs(scored[k]-r['student_dev'][k]) for k in ['L_T','L_M','total']);assert error<1e-9 and scored['normalization_error']<1e-12
        t,m,mask=next(adapter.base.batches([dev[0]],1,False,random.Random(0)));_,lp=model.event_distributions(t,m,mask)
        changed=m.clone();changed[:,1:]=(changed[:,1:]+1)%K;_,alternate=model.event_distributions(t,changed,mask);causal=float((lp[:,0]-alternate[:,0]).abs().max());assert causal<1e-12
        rows.append(dict(seed=seed,score=scored,serialized_score_error=error,causal_error=causal,bundle=str(path.relative_to(ROOT)),bundle_sha256=digest(path),bundle_bytes=path.stat().st_size,parameters=sum(p.numel() for p in model.parameters())))
    sources=['experiments/tpp/check_split2_observation_serving.py','experiments/tpp/nonlinear_single_mark_readout.py','experiments/tpp/single_mark_readout.py','experiments/tpp/race_tpp_logspread.py','experiments/tpp/race_tpp_v19.py','experiments/tpp/race_tpp_b4g.py']
    result=dict(status='completed',tag=a.tag,battle='B4 one-core serving contract',rows=rows,source_sha256={s:digest(ROOT/s) for s in sources},input_sha256=inputs,data_sha256=dict(val=digest(adapter.DATA/'github_filtered'/'split_2/val.json')),wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='One fixed Github split2 DEV seed; CPU float64. Serving bundle contains one core and normalization/readout metadata; no teacher checkpoints loaded and no TRAIN loaded to restore. Normalized all-mark probabilities, time/score parity and first-target suffix invariance. No TEST or deployment to external service.')
    with (ROOT/'experiments/results/tpp_b4'/f'{a.tag}.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
