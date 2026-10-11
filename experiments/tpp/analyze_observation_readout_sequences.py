"""B4 DEV sequence-level one-core readout failure localization; no fitting."""
import argparse,json,time,resource,random
import numpy as np
import torch
from check_single_readout_serving import adapter,ROOT,digest,restore
from single_mark_readout import distributions
@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    proof=ROOT/'experiments/results/tpp_b4/aws_observation_readout_serving_20261011.json';r=json.loads(proof.read_text());assert r['status']=='completed'
    dev,K=adapter.load('github_filtered',1,'val');prior_path=ROOT/'experiments/results/tpp_b4/aws_single_readout_serving_20261011_v1.json';prior=json.loads(prior_path.read_text());assert prior['status']=='completed';rows=[];inputs={str(proof.relative_to(ROOT)):digest(proof)}
    for item in r['rows']:
        path=ROOT/item['bundle'];assert digest(path)==item['bundle_sha256'];inputs[item['bundle']]=digest(path);model=restore(torch.load(path,weights_only=True));old=prior['rows'][item['seed']];oldpath=ROOT/old['bundle'];assert digest(oldpath)==old['bundle_sha256'];inputs[old['bundle']]=digest(oldpath);oldmodel=restore(torch.load(oldpath,weights_only=True));sequences=[]
        for index,seq in enumerate(dev):
            t,m,mask=next(adapter.base.batches([seq],1,False,random.Random(0)));valid=mask[:,1:];time_lp,base_lp=oldmodel.event_distributions(t,m,mask);student_time,student_lp=model.event_distributions(t,m,mask)
            assert torch.equal(time_lp,student_time)
            selected=m[:,1:,None];gain=(student_lp.gather(-1,selected)-base_lp.gather(-1,selected)).squeeze(-1)[valid]
            sequences.append(dict(index=index,targets=int(valid.sum()),mark_gain=float(gain.sum()),per_target_gain=float(gain.mean()),targets_helped=int((gain>0).sum()),largest_target_loss=float(-gain.min())))
        mean=float(np.mean([s['mark_gain'] for s in sequences]));assert abs(mean-(item['score']['L_M']*-1+old['score']['L_M']))<1e-9
        rows.append(dict(seed=item['seed'],mean_mark_gain=mean,sequences_helped=sum(s['mark_gain']>0 for s in sequences),sequences=sequences,worst_sequences=sorted(sequences,key=lambda s:s['mark_gain'])[:5]))
    sources=['experiments/tpp/analyze_observation_readout_sequences.py','experiments/tpp/check_single_readout_serving.py','experiments/tpp/single_mark_readout.py','experiments/tpp/nonlinear_single_mark_readout.py'];out=dict(status='completed',tag=a.tag,battle='B4',rows=rows,input_sha256=inputs,source_sha256={s:digest(ROOT/s) for s in sources},wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Same Github split1 DEV, three fixed seeds, no fitting/TEST. Localizes observed-label-anchor gains versus prior compact readout before further development; no independent confirmation.')
    with (ROOT/'experiments/results/tpp_b4'/f'{a.tag}.json').open('x') as f:json.dump(out,f,indent=2)
    print(json.dumps(out))
if __name__=='__main__':main()
